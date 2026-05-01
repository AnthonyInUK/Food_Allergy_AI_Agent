import os
import time
import base64
import hashlib
import asyncio
import json
import redis
import re
import random
import unicodedata
from io import BytesIO
import queue
import threading
from typing import List, TypedDict, Annotated, Union, Generator, Dict, Any, Optional, NotRequired
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph, START
from langgraph.types import RetryPolicy
from sqlalchemy import text as sql_text

from agent_logic import get_sql_agent, get_db, get_llm, query_text as sql_query_text
from app_config import get_runtime_settings
from db import get_engine
from llm_invoke import invoke_with_retry

load_dotenv(override=True)
try:
    from langsmith_config import configure_langsmith_tracing_compat

    configure_langsmith_tracing_compat()
except ImportError:
    pass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --- 1. 全局资源缓存 ---

_llm_cache = None
_vision_llm_cache = None
_vectorstore_cache = None
_sql_agent_cache = None

def get_fast_llm():
    """获取轻量级LLM实例（缓存）"""
    global _llm_cache
    if _llm_cache is None:
        _base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
        _llm_cache = ChatOpenAI(model="gpt-4o-mini", temperature=0, **({"base_url": _base_url} if _base_url else {}))
    return _llm_cache


def get_vision_llm():
    """多模态生成（有用户附图时）：默认 gpt-4o，可用环境变量 VISION_LLM_MODEL 覆盖。"""
    global _vision_llm_cache
    if _vision_llm_cache is None:
        model = os.getenv("VISION_LLM_MODEL", "gpt-4o").strip() or "gpt-4o"
        _base_url_v = os.getenv("OPENAI_BASE_URL", "").strip() or None
        _vision_llm_cache = ChatOpenAI(model=model, temperature=0, **({"base_url": _base_url_v} if _base_url_v else {}))
    return _vision_llm_cache


def _image_bytes_to_data_url(image_bytes: bytes) -> str:
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    if len(image_bytes) >= 8 and image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        mime = "image/png"
    elif len(image_bytes) >= 3 and image_bytes[:3] == b"\xff\xd8\xff":
        mime = "image/jpeg"
    elif len(image_bytes) >= 6 and image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        mime = "image/gif"
    elif len(image_bytes) >= 12 and image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        mime = "image/jpeg"
    return f"data:{mime};base64,{b64}"


def get_vectorstore():
    """获取向量存储实例（缓存）；数据落在 CHROMA_PERSIST_DIRECTORY 磁盘目录。"""
    global _vectorstore_cache
    if _vectorstore_cache is None:
        settings = get_runtime_settings()
        persist_dir = settings.chroma_persist_directory
        os.makedirs(persist_dir, exist_ok=True)
        # food_products collection currently uses 384-d sentence-transformers vectors.
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        _vectorstore_cache = Chroma(
            persist_directory=persist_dir,
            collection_name=settings.chroma_text_collection_name,
            embedding_function=embeddings,
        )
    return _vectorstore_cache


def warm_retrieval_stack() -> None:
    """
    检索预热：加载 HuggingFace 嵌入模型 + 打开 Chroma，并执行一次极小查询。
    把「冷启动」成本挪到进程启动，首个真实用户问题的「向量检索」段通常明显变短。
    """
    settings = get_runtime_settings()
    if not settings.enable_retrieval_warmup:
        return
    t0 = time.perf_counter()
    try:
        vs = get_vectorstore()
        vs.similarity_search("warmup", k=1)
        dt = time.perf_counter() - t0
        print(f"🔥 Retrieval warm-up done in {dt:.2f}s", flush=True)
    except Exception as e:
        print(f"⚠️ Retrieval warm-up failed (non-fatal): {e}", flush=True)


# --- 2. 缓存系统（内存式，不依赖Streamlit） ---

class CacheManager:
    """全局缓存管理器"""
    def __init__(self):
        self.response_cache: Dict[str, Any] = {}
        self.retrieval_cache: Dict[str, Any] = {}
        self.generation_cache: Dict[str, Any] = {}
        self.grade_cache: Dict[str, Any] = {}
        self.hallucination_cache: Dict[str, Any] = {}
        self.answer_grade_cache: Dict[str, Any] = {}
        
        # 时间戳映射用于TTL管理
        self.response_cache_ts: Dict[str, float] = {}
        self.retrieval_cache_ts: Dict[str, float] = {}
        self.generation_cache_ts: Dict[str, float] = {}
        self.grade_cache_ts: Dict[str, float] = {}
        self.hallucination_cache_ts: Dict[str, float] = {}
        self.answer_grade_cache_ts: Dict[str, float] = {}
        
        self.cache_stats = {"hits": 0, "misses": 0}

    def get_cache_stats(self):
        """获取缓存统计"""
        total = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total * 100) if total > 0 else 0
        return {
            "hit_rate": hit_rate,
            "total_queries": total,
            **self.cache_stats
        }

    def clear_all(self):
        """清空所有缓存"""
        self.response_cache.clear()
        self.retrieval_cache.clear()
        self.generation_cache.clear()
        self.grade_cache.clear()
        self.hallucination_cache.clear()
        self.answer_grade_cache.clear()
        
        self.response_cache_ts.clear()
        self.retrieval_cache_ts.clear()
        self.generation_cache_ts.clear()
        self.grade_cache_ts.clear()
        self.hallucination_cache_ts.clear()
        self.answer_grade_cache_ts.clear()
        
        self.cache_stats = {"hits": 0, "misses": 0}


# 全局缓存实例
_cache_manager = CacheManager()

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", 3600))
RESPONSE_CACHE_TTL_SECONDS = int(
    os.getenv("RESPONSE_CACHE_TTL_SECONDS", 3 * 24 * 60 * 60)
)
NODE_RETRY_MAX_ATTEMPTS = int(os.getenv("NODE_RETRY_MAX_ATTEMPTS", "3"))
NODE_RETRY_INITIAL_INTERVAL = float(os.getenv("NODE_RETRY_INITIAL_INTERVAL", "0.5"))
NODE_RETRY_BACKOFF_FACTOR = float(os.getenv("NODE_RETRY_BACKOFF_FACTOR", "2.0"))
NODE_RETRY_MAX_INTERVAL = float(os.getenv("NODE_RETRY_MAX_INTERVAL", "8.0"))
_settings = get_runtime_settings()
CACHE_TTL_SECONDS = _settings.inproc_cache_ttl_seconds
RESPONSE_CACHE_TTL_SECONDS = _settings.response_cache_ttl_seconds
NODE_RETRY_MAX_ATTEMPTS = _settings.node_retry_max_attempts
NODE_RETRY_INITIAL_INTERVAL = _settings.node_retry_initial_interval
NODE_RETRY_BACKOFF_FACTOR = _settings.node_retry_backoff_factor
NODE_RETRY_MAX_INTERVAL = _settings.node_retry_max_interval


def _node_retry_policy() -> RetryPolicy:
    return RetryPolicy(
        initial_interval=NODE_RETRY_INITIAL_INTERVAL,
        backoff_factor=NODE_RETRY_BACKOFF_FACTOR,
        max_interval=NODE_RETRY_MAX_INTERVAL,
        max_attempts=NODE_RETRY_MAX_ATTEMPTS,
        jitter=True,
    )


def get_semantic_hash(text: str) -> str:
    """将文本转换为MD5哈希"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


@lru_cache(maxsize=1)
def _load_brand_aliases() -> Dict[str, List[str]]:
    default_aliases: Dict[str, List[str]] = {
        "Lee Kum Kee": ["lee kum kee", "likumkee", "lkk", "李锦记"],
        "Haday": ["haday", "hai tian", "haitian", "海天"],
        "Master Kong": ["master kong", "康师傅"],
    }
    path = os.path.join(os.getcwd(), "data", "brand_aliases.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return default_aliases
        out: Dict[str, List[str]] = {}
        for canonical, aliases in raw.items():
            if not isinstance(canonical, str):
                continue
            alias_list = aliases if isinstance(aliases, list) else []
            cleaned = [a.strip() for a in alias_list if isinstance(a, str) and a.strip()]
            if cleaned:
                out[canonical.strip()] = cleaned
        return out or default_aliases
    except Exception:
        return default_aliases


def _normalize_question(question: str) -> str:
    q = unicodedata.normalize("NFKC", (question or "").strip().lower())
    q = re.sub(r"[“”\"'`]+", " ", q)
    q = re.sub(r"[，。！？、,:;!?()\\[\\]{}<>]+", " ", q)
    q = re.sub(
        r"\b(please|pls|tell me|can you|what is|ingredients|ingredient|allergens?)\b",
        " ",
        q,
    )
    q = re.sub(r"(请问|告诉我|一下|是什么|有哪些|成分|配料|过敏原)", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def _cache_key_normalize_question(question: str) -> str:
    """整回答/检索缓存用：在语义规范化基础上去掉空格，避免「旺仔qq糖」与「旺仔 QQ 糖」拆成不同 key。"""
    return re.sub(r"\s+", "", _normalize_question(question))


_FILENAME_ATTACHMENT_RX = [
    re.compile(r"\[\s*附图\s*[：:]\s*([^\]]+?)\s*\]", re.I),
    re.compile(r"\b附图\s*[：:]\s*([\w./-]+\.(?:jpe?g|png|gif|webp))\b", re.I),
    re.compile(r"\b(?:attachment|image|file)\s*[：:]\s*([\w./-]+\.(?:jpe?g|png|gif|webp))\b", re.I),
]


def _filename_stems_for_retrieval(question: str) -> List[str]:
    """从用户文案里抽附图文件名主干（如 rousong.jpeg → rousong），不依赖视觉模型，用于补强文本向量检索。"""
    raw = question or ""
    stems: List[str] = []
    for rx in _FILENAME_ATTACHMENT_RX:
        for m in rx.finditer(raw):
            name = (m.group(1) or "").strip()
            if not name:
                continue
            base = name.replace("\\", "/").rsplit("/", 1)[-1]
            stem = re.sub(r"\.(?:jpe?g|png|gif|webp)$", "", base, flags=re.I)
            stem = re.sub(r"[^\w\u4e00-\u9fff-]+", "", stem, flags=re.UNICODE)
            if len(stem) >= 2:
                stems.append(stem)
    return stems


def _build_retrieval_query(question: str) -> tuple[str, List[str]]:
    """拼向量检索 query：主串用去空格归一化（与缓存 key 一致），别名仍用原文 + norm 匹配。"""
    norm_q = _normalize_question(question)
    embed_q = _cache_key_normalize_question(question)
    if not embed_q:
        embed_q = re.sub(r"\s+", "", norm_q) if norm_q else ""
    aliases = _load_brand_aliases()
    matched_terms: List[str] = []
    for canonical, alias_list in aliases.items():
        all_terms = [canonical] + alias_list
        if any((t.lower() in norm_q) or (t in question) for t in all_terms):
            matched_terms.extend(all_terms)
    seen: set[str] = set()
    terms: List[str] = []
    for t in [embed_q] + matched_terms:
        key = t.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        terms.append(t.strip())
    for stem in _filename_stems_for_retrieval(question):
        key = stem.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        terms.append(stem.strip())
    return " ".join(terms[:20]).strip(), matched_terms


def _response_cache_key(question: str, image_bytes: Optional[bytes]) -> str:
    """带图问题时响应缓存键需包含图片内容，避免误命中纯文本缓存。
    文本部分用 _cache_key_normalize_question，弱化空格/大小写差异。
    """
    qn = _cache_key_normalize_question(question)
    if image_bytes:
        return get_semantic_hash(qn + "\n" + hashlib.md5(image_bytes).hexdigest())
    return get_semantic_hash(qn)


def _retrieval_cache_key(question: str, image_bytes: Optional[bytes]) -> str:
    qn = _cache_key_normalize_question(question)
    if image_bytes:
        return get_semantic_hash("retrieve:" + qn + "|img:" + hashlib.md5(image_bytes).hexdigest())
    return get_semantic_hash("retrieve:" + qn)


_clip_model_singleton = None


def _get_clip_encoder():
    """Lazy-load CLIP / sentence-transformers image encoder (only when image retrieval enabled)."""
    global _clip_model_singleton
    if _clip_model_singleton is None:
        from sentence_transformers import SentenceTransformer

        _clip_model_singleton = SentenceTransformer(_settings.clip_model_name)
    return _clip_model_singleton


def _query_similar_products_by_image(image_bytes: bytes, k: int) -> List[str]:
    """Chroma product_images 集合：以图搜货。集合不存在或为空时返回 []。"""
    if not _settings.enable_image_vector_retrieval or not image_bytes:
        return []
    try:
        import chromadb
        from PIL import Image

        client = chromadb.PersistentClient(path=_settings.chroma_persist_directory)
        coll = client.get_collection(_settings.chroma_image_collection_name)
    except Exception as e:
        _debug_log("H23", "image vector collection unavailable", {"error": str(e)})
        return []
    try:
        n_total = int(coll.count())
        if n_total <= 0:
            return []
    except Exception:
        n_total = 1
    try:
        model = _get_clip_encoder()
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        qemb = model.encode(img, convert_to_numpy=True)
        qlist = qemb.tolist() if hasattr(qemb, "tolist") else list(qemb)
    except Exception as e:
        _debug_log("H23", "clip encode query image failed", {"error": str(e)})
        return []
    n_results = min(max(k, 1), max(n_total, 1))
    try:
        res = coll.query(query_embeddings=[qlist], n_results=n_results)
    except Exception as e:
        _debug_log("H23", "chroma image query failed", {"error": str(e)})
        return []
    docs = (res.get("documents") or [[]])[0]
    return [d for d in docs if isinstance(d, str) and d.strip()]


def _merge_retrieval_docs(image_docs: List[str], text_docs: List[str], max_total: int) -> List[str]:
    """图检索结果优先，再拼文本检索；按 product_id 或内容去重。"""
    seen = set()
    out: List[str] = []
    for chunk in image_docs + text_docs:
        if not chunk or not str(chunk).strip():
            continue
        s = str(chunk).strip()
        m = re.search(r"\[product_id:([^\]]+)\]", s)
        key = m.group(1) if m else hashlib.md5(s[:400].encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= max_total:
            break
    return out


def purge_expired_caches():
    """删除过期的缓存项"""
    now = time.time()
    cache_pairs = {
        "response_cache": "response_cache_ts",
        "retrieval_cache": "retrieval_cache_ts",
        "generation_cache": "generation_cache_ts",
        "grade_cache": "grade_cache_ts",
        "hallucination_cache": "hallucination_cache_ts",
        "answer_grade_cache": "answer_grade_cache_ts",
    }
    
    for cache_name, ts_name in cache_pairs.items():
        cache_dict = getattr(_cache_manager, cache_name)
        ts_dict = getattr(_cache_manager, ts_name)
        
        expired_keys = [k for k, ts in ts_dict.items() if now - ts > CACHE_TTL_SECONDS]
        for key in expired_keys:
            cache_dict.pop(key, None)
            ts_dict.pop(key, None)


# --- 3. 状态类型定义 ---

class GraphState(TypedDict):
    """图的状态"""
    question: str
    generation: str
    documents: List[str]
    web_results: List[str]
    document_relevance_score: str
    hallucination_score: str
    answer_is_helpful: str
    grade_documents: str
    answer_grade: str
    score: float
    node: str
    image_bytes: bytes
    generation_source: str
    retrieval_top_score: float
    # plan_route 输出：规则 / 灰区 JSON 一次调用
    need_web: bool
    task: str
    plan_sql_hit: str
    # HTTP 流式回答：仅 /api/chat/stream 注入；generate 内 chain.stream 写入文本块
    stream_delta_queue: NotRequired[Any]


# --- 4. 核心节点函数 ---

def _normalize_sql_result(sql_result: Any) -> str:
    if isinstance(sql_result, tuple):
        return sql_result[0] if len(sql_result) > 0 else ""
    if isinstance(sql_result, str):
        return sql_result
    if sql_result is None:
        return ""
    return str(sql_result)


def _is_local_db_miss(text_value: str) -> bool:
    text_lower = (text_value or "").strip().lower()
    if not text_lower:
        return True
    miss_markers = [
        "数据库中没有找到",
        "没有找到与",
        "请确认产品名称",
        "not found in database",
        "no matching product",
        "no relevant product",
    ]
    return any(marker in text_lower for marker in miss_markers)


def _target_lang_from_question(question: str) -> str:
    # 简单语种判定：含中文字符则按中文输出，否则按英文输出
    if any("\u4e00" <= ch <= "\u9fff" for ch in (question or "")):
        return "zh"
    return "en"


def _extract_product_hint(question: str) -> str:
    """从用户问题中提取更像产品名的片段，降低 SQL 模糊匹配噪音。"""
    q = (question or "").strip()
    if not q:
        return ""
    noise_words = [
        "成分",
        "成份",
        "配料",
        "过敏原",
        "长什么样",
        "图片",
        "外观",
        "有没有",
        "可以吃吗",
        "安全吗",
        "ingredients",
        "allergen",
        "allergens",
        "what is in",
        "look like",
        "image",
        "picture",
    ]
    low = q.lower()
    for w in noise_words:
        low = low.replace(w, " ")
    cleaned = " ".join(low.split()).strip()
    if cleaned:
        return cleaned
    return q


def _documents_cover_question_product(question: str, documents: List[str]) -> bool:
    """检索片段是否真正覆盖用户问的产品名；用于避免「同类零食」被判 relevant 却不走联网。"""
    hint = (_extract_product_hint(question) or "").strip()
    if not hint:
        return True
    blob = "\n".join(documents).lower()
    h = hint.lower()
    if h and h in blob:
        return True
    parts = re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9]{3,}", h)
    if not parts:
        return True
    return all(p in blob for p in parts)


def _grey_zone_json_route(
    question: str,
    documents: List[str],
    *,
    top_score: float,
    overlap: int,
    doc_count: int,
) -> tuple[bool, str]:
    """
    灰区：一次结构化 JSON 调用，决定 need_web + task。
    解析失败时默认 need_web=True（偏保守，避免漏联网）。
    """
    try:
        llm = get_fast_llm()
        preview = "\n\n".join(documents[:3])[:2400]
        prompt = ChatPromptTemplate.from_template(
            """You route a food Q&A assistant. Reply with ONLY a compact JSON object (no markdown) with exactly two keys:
- "need_web" (boolean): true if web search would likely improve the answer because local excerpts are weak or ambiguous for this question.
- "task" (string): one of ingredient, allergy, compare, general

Signals:
- retrieval_top_score: {top_score}
- keyword_overlap_with_first_doc: {overlap}
- doc_count: {doc_count}

Question:
{question}

Retrieved excerpt preview:
{preview}
"""
        )
        chain = prompt | llm
        out = invoke_with_retry(
            lambda: chain.invoke(
                {
                    "question": question,
                    "preview": preview,
                    "top_score": top_score,
                    "overlap": overlap,
                    "doc_count": doc_count,
                }
            )
        )
        raw = (out.content if hasattr(out, "content") else str(out)).strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(raw)
        need_web = bool(data.get("need_web", True))
        task = str(data.get("task", "general") or "general").lower().strip()
        if task not in ("ingredient", "allergy", "compare", "general"):
            task = "general"
        return need_web, task
    except Exception as e:
        _debug_log("H23", "grey_zone_json_route failed", {"error": str(e)})
        return True, "general"


def plan_route(state: GraphState) -> GraphState:
    """
    规则 + 至多一次 JSON：先定 need_web 与 task。
    主干：retrieve → plan_route → (websearch?) → generate
    """
    question = (state.get("question") or "").strip()
    documents = state.get("documents") or []
    top_score = float(state.get("retrieval_top_score") or 0.0)
    state["web_results"] = []
    state["plan_sql_hit"] = ""
    state["task"] = "general"
    state["need_web"] = False

    if not documents:
        hit = _deterministic_local_sql_lookup(question)
        if hit:
            state["plan_sql_hit"] = hit
            state["need_web"] = False
            state["task"] = "sql_local"
        else:
            state["need_web"] = True
            state["task"] = "web_assisted"
        state["node"] = "plan_route"
        _debug_log(
            "H11",
            "plan_route",
            {"branch": "empty_docs", "need_web": state["need_web"], "task": state["task"]},
        )
        return state

    hint_ok = _documents_cover_question_product(question, documents)
    first_doc = documents[0] if documents else ""
    overlap = _token_overlap(question, first_doc)
    doc_count = len(documents)

    if not hint_ok:
        state["need_web"] = True
        state["task"] = "web_assisted"
    elif top_score < 0.45 or overlap == 0 or doc_count >= 3:
        state["need_web"] = True
        state["task"] = "web_assisted"
    elif top_score >= 0.82 and overlap >= 1:
        state["need_web"] = False
        state["task"] = "rag_high"
    else:
        if _env_bool("GREY_ZONE_JSON_ROUTE", True):
            nw, tk = _grey_zone_json_route(
                question,
                documents,
                top_score=top_score,
                overlap=overlap,
                doc_count=doc_count,
            )
            if _settings.quality_first and not nw:
                nw = True
                _debug_log(
                    "H23",
                    "quality_first: grey_zone JSON wanted skip-web; overridden to need_web=True",
                    {"task": tk},
                )
            state["need_web"] = nw
            state["task"] = tk
        else:
            state["need_web"] = True
            state["task"] = "web_assisted"

    state["node"] = "plan_route"
    _debug_log(
        "H11",
        "plan_route",
        {
            "branch": "has_docs",
            "need_web": state["need_web"],
            "task": state["task"],
            "doc_count": doc_count,
            "top_score": top_score,
        },
    )
    return state


def _plan_to_web_edge(state: GraphState) -> str:
    return "websearch" if state.get("need_web") else "generate"


def _deterministic_local_sql_lookup(question: str) -> str:
    """不用 LLM，直接查 products，避免未命中时产生“编造成分”。"""
    hint = _extract_product_hint(question)
    if not hint:
        return ""
    like_hint = f"%{hint}%"
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            sql_text(
                """
                SELECT name, brand, ingredients, allergens
                FROM products
                WHERE lower(coalesce(name,'')) LIKE lower(:kw)
                   OR lower(coalesce(brand,'')) LIKE lower(:kw)
                ORDER BY length(coalesce(ingredients,'')) DESC
                LIMIT 1
                """
            ),
            {"kw": like_hint},
        ).first()
    if not row:
        return ""
    name, brand, ingredients, allergens = row
    ingredients = (ingredients or "").strip()
    if not ingredients:
        return ""
    if _target_lang_from_question(question) == "zh":
        return (
            f"产品：{name}（品牌：{brand or '未知'}）\n"
            f"配料：{ingredients}\n"
            f"过敏原：{(allergens or '未标注')}"
        )
    return (
        f"Product: {name} (Brand: {brand or 'Unknown'})\n"
        f"Ingredients: {ingredients}\n"
        f"Allergens: {allergens or 'Not specified'}"
    )


def _rewrite_sql_result_to_question_lang(question: str, sql_text: str) -> str:
    if not (sql_text or "").strip():
        return sql_text
    target_lang = _target_lang_from_question(question)
    llm = get_fast_llm()
    prompt = ChatPromptTemplate.from_template(
        """You are a faithful formatter. Rewrite the database answer into the target language while preserving all facts.

Target language: {target_lang}
User question: {question}
Database answer:
{sql_text}

Requirements:
- Keep facts, numbers, ingredients, allergen names unchanged in meaning.
- Do not invent any new information.
- Keep concise and readable.
"""
    )
    chain = prompt | llm
    result = invoke_with_retry(
        lambda: chain.invoke(
            {"target_lang": "Chinese" if target_lang == "zh" else "English", "question": question, "sql_text": sql_text}
        )
    )
    rewritten = result.content if hasattr(result, "content") else str(result)
    return rewritten or sql_text


def _extract_web_contents(raw_results: Any) -> List[str]:
    """兼容不同 Tavily 返回结构，统一提取可用文本。"""
    if raw_results is None:
        return []
    if isinstance(raw_results, list):
        contents: List[str] = []
        for item in raw_results:
            if isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str) and content.strip():
                    contents.append(content)
            elif isinstance(item, str) and item.strip():
                contents.append(item)
        return contents
    if isinstance(raw_results, dict):
        # 兼容 {"results":[...]} 以及单条 {"content":"..."}
        if isinstance(raw_results.get("results"), list):
            return _extract_web_contents(raw_results.get("results"))
        content = raw_results.get("content")
        return [content] if isinstance(content, str) and content.strip() else []
    if isinstance(raw_results, str):
        text_value = raw_results.strip()
        if not text_value:
            return []
        try:
            parsed = json.loads(text_value)
            return _extract_web_contents(parsed)
        except Exception:
            return [text_value]
    return []


def _looks_like_web_error(raw_results: Any) -> bool:
    text_value = str(raw_results or "").lower()
    markers = [
        "httperror(",
        "401",
        "403",
        "unauthorized",
        "forbidden",
        "invalid api key",
        "api key",
    ]
    return any(m in text_value for m in markers)


def _tavily_key_fingerprint() -> dict:
    key = (os.getenv("TAVILY_API_KEY") or "").strip()
    return {
        "present": bool(key),
        "length": len(key),
        "prefix": key[:8] if key else "",
        "suffix": key[-4:] if len(key) >= 4 else "",
    }


def retrieve_documents(state: GraphState) -> GraphState:
    """检索相关文档：文本向量 + 可选图向量（CLIP 集合）合并去重。"""
    question = state["question"]
    image_bytes: bytes = state.get("image_bytes") or b""
    ib: Optional[bytes] = image_bytes if len(image_bytes) > 0 else None
    key = _retrieval_cache_key(question, ib)
    search_query, alias_hits = _build_retrieval_query(question)
    if not search_query:
        search_query = _cache_key_normalize_question(question) or (question or "").strip()
    _debug_log(
        "H9",
        "retrieve start",
        {
            "question_hash": key,
            "question_preview": question[:120],
            "alias_hit_count": len(alias_hits),
            "alias_hits_preview": alias_hits[:8],
            "search_query_preview": search_query[:160],
            "has_image": bool(ib),
            "image_retrieval_enabled": _settings.enable_image_vector_retrieval,
        },
    )

    if key in _cache_manager.retrieval_cache:
        _cache_manager.cache_stats["hits"] += 1
        state["documents"] = _cache_manager.retrieval_cache[key]
        state["node"] = "retrieve"
        return state

    _cache_manager.cache_stats["misses"] += 1

    try:
        k_eff = _settings.effective_vector_top_k()

        def _text_retrieve_branch() -> tuple[List[str], float]:
            """文本向量检索（与原逻辑一致）。"""
            vectorstore = get_vectorstore()
            top_score_l = 0.0
            text_docs: List[str] = []
            try:
                scored = vectorstore.similarity_search_with_relevance_scores(
                    search_query, k=k_eff
                )
                if scored:
                    text_docs = [doc.page_content for doc, _ in scored]
                    score_vals = [
                        float(score)
                        for _, score in scored
                        if isinstance(score, (int, float))
                    ]
                    top_score_l = max(score_vals) if score_vals else 0.0
            except Exception:
                docs = vectorstore.similarity_search(search_query, k=k_eff)
                text_docs = [doc.page_content for doc in docs]
            return text_docs, top_score_l

        def _image_retrieve_branch() -> List[str]:
            if not ib:
                return []
            return _query_similar_products_by_image(ib, _settings.image_vector_top_k)

        # 有图时文本检索与 CLIP 图检索互不依赖，并行以缩短墙钟时间；合并顺序与结果与串行一致。
        if ib and _settings.enable_image_vector_retrieval:
            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_text = pool.submit(_text_retrieve_branch)
                fut_img = pool.submit(_image_retrieve_branch)
                text_documents, top_score = fut_text.result()
                image_documents = fut_img.result()
        else:
            text_documents, top_score = _text_retrieve_branch()
            image_documents = _image_retrieve_branch()

        max_merged = max(_settings.effective_vector_top_k() * 2, 10)
        documents = _merge_retrieval_docs(image_documents, text_documents, max_total=max_merged)

        _debug_log(
            "H9",
            "retrieve result",
            {
                "question_hash": key,
                "text_doc_count": len(text_documents),
                "image_doc_count": len(image_documents),
                "merged_doc_count": len(documents),
                "top_score": top_score,
                "has_lee_kum_kee": any("Lee Kum Kee" in d for d in documents),
                "first_doc_preview": (documents[0][:180] if documents else ""),
            },
        )

        _cache_manager.retrieval_cache[key] = documents
        _cache_manager.retrieval_cache_ts[key] = time.time()

        state["documents"] = documents
        state["retrieval_top_score"] = top_score
    except Exception as e:
        _debug_log("H15", "retrieve failed, raise for retry", {"error": str(e)})
        raise

    state["node"] = "retrieve"
    return state


def _token_overlap(question: str, doc_text: str) -> int:
    q_tokens = set(re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]{2,}", (question or "").lower()))
    if not q_tokens:
        return 0
    d = (doc_text or "").lower()
    return sum(1 for t in q_tokens if t in d)


_UNIFIED_ANSWER_PROMPT = """You are a food product Q&A assistant (ingredients, allergens, brands). Prefer answering in the same language as the user's question.

User question:
{question}

Routing task hint: {task}

=== Local retrieval excerpts (may be empty, wrong product, or incomplete; ignore if not clearly the same product as the question) ===
{local_block}

=== Web snippets (may be empty; use when present for freshness and product-specific facts) ===
{web_block}

Instructions:
- Ground claims in the evidence blocks. If evidence is insufficient for a definitive product-specific answer, say so clearly.
- If local excerpts and web snippets conflict, prefer corroborated facts and state uncertainty briefly.
- Be concise and helpful.
"""


_UNIFIED_VISION_ANSWER_TEXT = """You are a food product Q&A assistant (ingredients, allergens, brands).

The user attached an IMAGE (food packaging or product). Use it as the **primary** source for **which product** this is (read visible brand/product text when possible).

Use the user's **written question** for **intent** (e.g. allergens, ingredients, what it is).

=== Local retrieval excerpts (may describe a different product than the image; if clearly mismatched, ignore for identity and say so briefly) ===
{local_block}

=== Web snippets (secondary; may not match the image) ===
{web_block}

User question:
{question}

Routing task hint: {task}

Instructions:
- Identify the product from the **image** first, then answer the user's intent.
- Cite local/web text only when it plausibly matches the same product as in the image; otherwise do not pretend it matches.
- Prefer the user's question language. Be concise.
"""


def _lc_stream_chunk_text(chunk: Any) -> str:
    """从 LangChain chat model stream chunk 取出纯文本增量。"""
    c = getattr(chunk, "content", None)
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: List[str] = []
        for block in c:
            if isinstance(block, dict):
                t = block.get("text")
                if isinstance(t, str):
                    parts.append(t)
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


def generate_answer(state: GraphState) -> GraphState:
    """单一生成模板：本地摘录 + 联网摘录 + 任务 hint；SQL 命中在 plan 阶段已写入 plan_sql_hit。"""
    question = state.get("question") or ""
    documents = state.get("documents") or []
    web_results = state.get("web_results") or []
    task = (state.get("task") or "general").strip()
    plan_sql_hit = (state.get("plan_sql_hit") or "").strip()

    dq_any = state.get("stream_delta_queue")
    dq: Optional[queue.Queue] = dq_any if isinstance(dq_any, queue.Queue) else None

    if plan_sql_hit:
        rewritten = _rewrite_sql_result_to_question_lang(question, plan_sql_hit)
        state["generation"] = rewritten
        state["generation_source"] = "sql_fallback"
        state["node"] = "generate"
        if dq is not None:
            try:
                dq.put_nowait(rewritten)
            except Exception:
                pass
        return state

    web_nonempty = [x.strip() for x in web_results if isinstance(x, str) and x.strip()]
    local_block = "\n\n".join(documents[:8]) if documents else "(none)"
    web_block = "\n\n".join(web_nonempty[:8]) if web_nonempty else "(none)"

    image_bytes = state.get("image_bytes") or b""
    has_image = len(image_bytes) > 0

    cache_payload = question + local_block + web_block + task
    if has_image:
        cache_payload += "|img:" + hashlib.md5(image_bytes).hexdigest()
    cache_key = get_semantic_hash(cache_payload)

    if cache_key in _cache_manager.generation_cache:
        _cache_manager.cache_stats["hits"] += 1
        cached_gen = _cache_manager.generation_cache[cache_key]
        state["generation"] = cached_gen
        state["generation_source"] = "vision_rag_cache" if has_image else "unified_cache"
        state["node"] = "generate"
        if dq is not None:
            try:
                dq.put_nowait(cached_gen)
            except Exception:
                pass
        return state

    _cache_manager.cache_stats["misses"] += 1
    try:
        if has_image:
            vision_text = _UNIFIED_VISION_ANSWER_TEXT.format(
                question=question,
                task=task,
                local_block=local_block,
                web_block=web_block,
            )
            data_url = _image_bytes_to_data_url(image_bytes)
            mm_message = HumanMessage(
                content=[
                    {"type": "text", "text": vision_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]
            )
            llm_v = get_vision_llm()
            if dq is not None:
                acc: List[str] = []
                for chunk in llm_v.stream([mm_message]):
                    piece = _lc_stream_chunk_text(chunk)
                    if piece:
                        acc.append(piece)
                        dq.put(piece)
                generation = "".join(acc)
            else:
                result = invoke_with_retry(lambda: llm_v.invoke([mm_message]))
                generation = result.content if hasattr(result, "content") else str(result)
            state["generation_source"] = "vision_rag"
        else:
            llm = get_fast_llm()
            prompt = ChatPromptTemplate.from_template(_UNIFIED_ANSWER_PROMPT)
            chain = prompt | llm
            invoke_payload = {
                "question": question,
                "task": task,
                "local_block": local_block,
                "web_block": web_block,
            }
            if dq is not None:
                acc: List[str] = []
                for chunk in chain.stream(invoke_payload):
                    piece = _lc_stream_chunk_text(chunk)
                    if piece:
                        acc.append(piece)
                        dq.put(piece)
                generation = "".join(acc)
            else:
                result = invoke_with_retry(lambda: chain.invoke(invoke_payload))
                generation = result.content if hasattr(result, "content") else str(result)
            if web_nonempty and documents:
                state["generation_source"] = "web_search"
            elif web_nonempty:
                state["generation_source"] = "web_fallback"
            elif documents:
                state["generation_source"] = "rag"
            else:
                state["generation_source"] = "unified"

        _cache_manager.generation_cache[cache_key] = generation
        _cache_manager.generation_cache_ts[cache_key] = time.time()
        state["generation"] = generation
    except Exception as e:
        _debug_log("H15", "generate failed, raise for retry", {"error": str(e)})
        raise

    state["node"] = "generate"
    return state


def web_search(state: GraphState) -> GraphState:
    """进行网络搜索"""
    question = state["question"]
    
    try:
        _debug_log(
            "H17",
            "tavily key fingerprint before websearch node",
            {"pid": os.getpid(), **_tavily_key_fingerprint()},
        )
        tavily_search = TavilySearchResults(max_results=3)
        results = invoke_with_retry(lambda: tavily_search.invoke({"query": question}))
        web_results = _extract_web_contents(results)
        state["web_results"] = web_results
    except Exception as e:
        _debug_log("H15", "websearch failed, raise for retry", {"error": str(e)})
        raise
    
    state["node"] = "web_search"
    return state


def _print_answer_provenance(state: GraphState) -> None:
    """终端打印本次答案的数据来源，便于区分向量 / 本地库 / 联网 / 缓存。"""
    if not _env_bool("LOG_ANSWER_PROVENANCE", True):
        return
    q = ((state.get("question") or "").strip().replace("\n", " "))[:160]
    docs = state.get("documents") or []
    webs = [w for w in (state.get("web_results") or []) if isinstance(w, str) and w.strip()]
    gs = (state.get("generation_source") or "").strip()
    task = (state.get("task") or "").strip()
    sql_hit = (state.get("plan_sql_hit") or "").strip()
    top = state.get("retrieval_top_score")
    need_web = state.get("need_web")

    explain_zh = {
        "sql_fallback": "生成走 PostgreSQL「products」直出（plan_route 命中配料）；向量检索仍可能跑过，但本答案不依赖向量块",
        "rag": "生成主要依据向量库(Chroma) 召回的本地摘录 + LLM",
        "web_search": "生成同时参考了向量召回与联网(Tavily)摘录 + LLM",
        "web_fallback": "生成主要依据联网摘录（向量无/弱或路由要求联网）+ LLM",
        "unified": "生成时本地向量块与联网摘录均偏空，LLM 主要凭提示与常识作答（请谨慎采纳）",
        "unified_cache": "命中生成结果内存缓存（同题同证据的早前答案）",
        "vision_rag": "用户附图：多模态模型直接看图定品 + 用户文字定意图；本地/联网摘录仅作辅助（与图不一致时可忽略）",
        "vision_rag_cache": "附图答案命中内存缓存（同图同证据的早前答案）",
    }.get(gs, f"未知 generation_source={gs!r}")

    payload = {
        "question_preview": q,
        "generation_source": gs,
        "summary_zh": explain_zh,
        "plan_task": task,
        "plan_need_web": bool(need_web) if need_web is not None else None,
        "vector_doc_count": len(docs),
        "retrieval_top_score": float(top) if isinstance(top, (int, float)) else top,
        "sql_products_hit": bool(sql_hit),
        "web_snippet_count": len(webs),
    }
    try:
        print("[answer_provenance] " + json.dumps(payload, ensure_ascii=False), flush=True)
    except Exception:
        print(f"[answer_provenance] {payload}", flush=True)


def finish(state: GraphState) -> GraphState:
    """结束节点"""
    _print_answer_provenance(state)
    state["node"] = "end"
    return state


# --- 5. 构建工作流图 ---

def build_graph():
    """检索 → 规则/结构化路由 → 可选联网 → 统一生成（无 grade 节点）。"""
    workflow = StateGraph(GraphState)
    retry_policy = _node_retry_policy() if _settings.enable_node_retry else None
    _debug_log(
        "H15",
        "build_graph retry policy",
        {
            "max_attempts": NODE_RETRY_MAX_ATTEMPTS,
            "initial_interval": NODE_RETRY_INITIAL_INTERVAL,
            "backoff_factor": NODE_RETRY_BACKOFF_FACTOR,
            "max_interval": NODE_RETRY_MAX_INTERVAL,
        },
    )

    if retry_policy is not None:
        workflow.add_node("retrieve", retrieve_documents, retry_policy=retry_policy)
        workflow.add_node("plan_route", plan_route, retry_policy=retry_policy)
        workflow.add_node("websearch", web_search, retry_policy=retry_policy)
        workflow.add_node("generate", generate_answer, retry_policy=retry_policy)
    else:
        workflow.add_node("retrieve", retrieve_documents)
        workflow.add_node("plan_route", plan_route)
        workflow.add_node("websearch", web_search)
        workflow.add_node("generate", generate_answer)
    workflow.add_node("end", finish)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "plan_route")
    workflow.add_conditional_edges(
        "plan_route",
        _plan_to_web_edge,
        {"websearch": "websearch", "generate": "generate"},
    )
    workflow.add_edge("websearch", "generate")
    workflow.add_edge("generate", "end")
    workflow.add_edge("end", END)

    return workflow.compile()


# 全局图实例（GRAPH_IMPL_VERSION 变更时强制重建，避免 worker 仍跑旧拓扑）
GRAPH_IMPL_VERSION = 5
_graph = None
_graph_impl_version_loaded: Optional[int] = None
_redis_client = None


def _debug_log(hypothesis_id: str, message: str, data: dict):
    if not _settings.enable_debug_log:
        return
    sample_rate = max(0.0, min(1.0, float(_settings.debug_log_sample_rate)))
    if sample_rate < 1.0 and random.random() > sample_rate:
        return
    try:
        with open("/Users/anthony/Desktop/llm/foodAIAgent/.cursor/debug-eb0752.log", "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "sessionId": "eb0752",
                        "runId": "sidebar-cache-collapse",
                        "hypothesisId": hypothesis_id,
                        "location": "graph_logic.py",
                        "message": message,
                        "data": data,
                        "timestamp": int(time.time() * 1000),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    except Exception:
        pass


def _get_redis_client():
    global _redis_client
    if not (_settings.enable_response_cache and _settings.enable_redis_response_cache):
        return None
    if _redis_client is not None:
        return _redis_client
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0").strip()
    if not redis_url:
        return None
    try:
        client = redis.Redis.from_url(redis_url, decode_responses=True)
        client.ping()
        _redis_client = client
        return _redis_client
    except Exception as e:
        _debug_log("H14", "redis unavailable", {"error": str(e)})
        return None


def _response_cache_redis_key(cache_key: str) -> str:
    return f"response_cache:{cache_key}"

def get_graph():
    """获取已编译的工作流图"""
    global _graph, _graph_impl_version_loaded
    if _graph is None or _graph_impl_version_loaded != GRAPH_IMPL_VERSION:
        _graph = build_graph()
        _graph_impl_version_loaded = GRAPH_IMPL_VERSION
    return _graph


# --- 6. 查询接口 ---

def _inject_step_seconds(value: Dict[str, Any], segment_start: List[float]) -> Dict[str, Any]:
    """为每个带 node 的 yield 附上本段耗时（秒），便于前端思考过程展示。"""
    out = dict(value)
    node = out.get("node")
    if node:
        now = time.perf_counter()
        out["step_seconds"] = now - segment_start[0]
        segment_start[0] = now
    return out


def _yield_graph_steps(initial_state: GraphState, seg: List[float]) -> Generator[Dict[str, Any], None, None]:
    graph = get_graph()
    for output in graph.stream(initial_state):
        for _, value in output.items():
            if isinstance(value, dict):
                yield _inject_step_seconds(dict(value), seg)
            else:
                yield value


def _graph_stream_worker(initial_state: GraphState, step_q: queue.Queue, seg: List[float]) -> None:
    try:
        for step in _yield_graph_steps(initial_state, seg):
            step_q.put(("step", step))
    except Exception as e:
        step_q.put(("error", e))
    finally:
        step_q.put(("stop", None))


def query_with_graph(
    question: str,
    image_bytes: Optional[bytes] = None,
    *,
    enable_answer_stream: bool = False,
) -> Generator[Dict[str, Any], None, None]:
    """
    执行查询并逐步返回结果
    
    Args:
        question: 用户问题
        image_bytes: 可选的图片字节数据
    
    Yields:
        包含node和generation信息的字典
    """
    purge_expired_caches()

    ib_opt: Optional[bytes] = image_bytes if image_bytes else None
    cache_key = _response_cache_key(question, ib_opt)
    _debug_log(
        "H8",
        "query_with_graph start",
        {
            "question_hash": cache_key,
            "question_len": len(question),
            "has_image": bool(ib_opt),
            "cache_size": len(_cache_manager.response_cache),
            "process_id": os.getpid(),
        },
    )
    if _settings.enable_response_cache and cache_key in _cache_manager.response_cache:
        _cache_manager.cache_stats["hits"] += 1
        _debug_log("H8", "response cache hit", {"question_hash": cache_key, "source": "memory"})
        if _env_bool("LOG_ANSWER_PROVENANCE", True):
            print(
                "[answer_provenance] "
                + json.dumps(
                    {
                        "question_preview": (question or "")[:160],
                        "generation_source": "response_cache_memory",
                        "summary_zh": "整回答命中进程内缓存，未重新执行向量/本地 SQL/联网",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        seg = [time.perf_counter()]
        yield _inject_step_seconds(
            {
                "node": "cache_hit",
                "generation": _cache_manager.response_cache[cache_key],
                "cache_source": "memory",
            },
            seg,
        )
        yield _inject_step_seconds(
            {
                "node": "end",
                "generation": _cache_manager.response_cache[cache_key],
            },
            seg,
        )
        return

    redis_client = _get_redis_client() if _settings.enable_response_cache else None
    if redis_client is not None:
        try:
            cached_response = redis_client.get(_response_cache_redis_key(cache_key))
            if cached_response:
                _cache_manager.cache_stats["hits"] += 1
                _cache_manager.response_cache[cache_key] = cached_response
                _cache_manager.response_cache_ts[cache_key] = time.time()
                _debug_log("H8", "response cache hit", {"question_hash": cache_key, "source": "redis"})
                if _env_bool("LOG_ANSWER_PROVENANCE", True):
                    print(
                        "[answer_provenance] "
                        + json.dumps(
                            {
                                "question_preview": (question or "")[:160],
                                "generation_source": "response_cache_redis",
                                "summary_zh": "整回答命中 Redis 缓存，未重新执行向量/本地 SQL/联网",
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                seg = [time.perf_counter()]
                yield _inject_step_seconds(
                    {
                        "node": "cache_hit",
                        "generation": cached_response,
                        "cache_source": "redis",
                    },
                    seg,
                )
                yield _inject_step_seconds(
                    {
                        "node": "end",
                        "generation": cached_response,
                    },
                    seg,
                )
                return
        except Exception as e:
            _debug_log("H14", "redis read error", {"error": str(e)})
    
    _cache_manager.cache_stats["misses"] += 1
    
    try:
        last_value: Dict[str, Any] = {}
        final_generation_by_event = ""
        
        initial_state: GraphState = {
            "question": question,
            "generation": "",
            "documents": [],
            "web_results": [],
            "document_relevance_score": "",
            "hallucination_score": "",
            "answer_is_helpful": "",
            "grade_documents": "",
            "answer_grade": "",
            "score": 0.0,
            "node": "start",
            "image_bytes": image_bytes or b"",
            "generation_source": "",
            "retrieval_top_score": 0.0,
            "need_web": False,
            "task": "",
            "plan_sql_hit": "",
        }
        
        # 执行工作流（每步 yield 附 step_seconds：距上一步 yield 的 wall time）
        seg = [time.perf_counter()]
        if enable_answer_stream:
            delta_q: queue.Queue = queue.Queue()
            step_q: queue.Queue = queue.Queue()
            streamed_state = {**initial_state, "stream_delta_queue": delta_q}
            th = threading.Thread(
                target=_graph_stream_worker,
                args=(streamed_state, step_q, seg),
                daemon=True,
            )
            th.start()
            merge_deadline = time.perf_counter() + 900.0
            while time.perf_counter() < merge_deadline:
                while True:
                    try:
                        piece = delta_q.get_nowait()
                        yield {"node": "__delta__", "delta": piece}
                    except queue.Empty:
                        break
                try:
                    kind, payload = step_q.get(timeout=0.05)
                except queue.Empty:
                    if not th.is_alive() and step_q.empty():
                        break
                    continue
                if kind == "step" and isinstance(payload, dict):
                    v = payload
                    last_value = v
                    if (v.get("node") or "") == "end":
                        final_generation_by_event = v.get("generation", "") or ""
                    yield v
                elif kind == "error":
                    th.join(timeout=120.0)
                    raise payload
                elif kind == "stop":
                    break
            else:
                raise TimeoutError(
                    "graph stream merge exceeded 900s (worker may be stuck)"
                )
            th.join(timeout=120.0)
            while True:
                try:
                    piece = delta_q.get_nowait()
                    yield {"node": "__delta__", "delta": piece}
                except queue.Empty:
                    break
            while True:
                try:
                    kind, payload = step_q.get_nowait()
                    if kind == "step" and isinstance(payload, dict):
                        v = payload
                        last_value = v
                        if (v.get("node") or "") == "end":
                            final_generation_by_event = v.get("generation", "") or ""
                        yield v
                    elif kind == "error":
                        raise payload
                except queue.Empty:
                    break
        else:
            for v in _yield_graph_steps(initial_state, seg):
                if isinstance(v, dict):
                    last_value = v
                    if (v.get("node") or "") == "end":
                        final_generation_by_event = v.get("generation", "") or ""
                yield v
        
        # 缓存最终答案
        final_generation = final_generation_by_event or (last_value.get("generation", "") if last_value else "")
        if not final_generation:
            final_generation = initial_state.get("generation", "") or ""
        if final_generation and _settings.enable_response_cache:
            _debug_log("H8", "storing final generation", {"store_key": cache_key, "final_len": len(final_generation)})
            _cache_manager.response_cache[cache_key] = final_generation
            _cache_manager.response_cache_ts[cache_key] = time.time()
            if redis_client is not None:
                try:
                    redis_client.setex(
                        _response_cache_redis_key(cache_key),
                        RESPONSE_CACHE_TTL_SECONDS,
                        final_generation,
                    )
                    _debug_log(
                        "H14",
                        "stored response cache redis",
                        {"store_key": cache_key, "ttl": RESPONSE_CACHE_TTL_SECONDS, "len": len(final_generation)},
                    )
                except Exception as e:
                    _debug_log("H14", "redis write error", {"error": str(e)})
        else:
            _debug_log("H8", "skip storing final generation", {"store_key": cache_key, "reason": "empty_generation"})
    
    except Exception as e:
        _debug_log("H21", "query_with_graph exception", {"error": str(e), "question_hash": cache_key})
        yield _inject_step_seconds(
            {
                "node": "generate_error",
                "detail": str(e),
                "generation_source": "web_error",
            },
            [time.perf_counter()],
        )
        raise


def get_cache_stats():
    """获取缓存统计信息"""
    return _cache_manager.get_cache_stats()


def clear_all_caches():
    """清空所有缓存"""
    _cache_manager.clear_all()
    print("All caches cleared")


if __name__ == "__main__":
    # 测试
    for result in query_with_graph("What are the main allergens in peanuts?"):
        print(result)
