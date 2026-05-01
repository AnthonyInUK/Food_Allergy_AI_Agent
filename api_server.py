"""
FastAPI 后端服务 - 将所有 Streamlit 逻辑转为 REST/WebSocket API
"""
import os
import re
import json
import time
import base64
import traceback
from typing import Optional, List, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, RedirectResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from sqlalchemy import text

from app_config import get_runtime_settings
from db import get_engine

# 导入现有的 Python 模块
from agent_logic import query_text
from important_data_store import record_important_qa, move_to_permanent_storage
from convo_store import (
    load_conversations,
    find_conversation_by_id,
    create_conversation,
    save_conversation,
    delete_conversation as delete_conversation_store,
)

load_dotenv()

from langsmith_config import configure_langsmith_tracing_compat, log_langsmith_startup_status

configure_langsmith_tracing_compat()

# 导入查询函数（不依赖Streamlit）
try:
    from graph_logic import query_with_graph
except:
    # 如果graph_logic导入失败，使用简单的查询函数
    def query_with_graph(prompt, image_bytes=None):
        """Fallback query function"""
        try:
            result = query_text(prompt)
            yield {"node": "end", "generation": result}
        except Exception as e:
            yield {"node": "end", "generation": f"Error: {str(e)}"}


def _is_placeholder_search_term(val: str) -> bool:
    """Avoid ILIKE on tokens like Unknown/N/A that match huge unrelated sets."""
    if not val or not str(val).strip():
        return True
    t = str(val).strip().lower()
    return t in {
        "unknown",
        "n/a",
        "na",
        "none",
        "n.a.",
        "-",
        "—",
        "?",
        "tbd",
        "不详",
        "未知",
        "不明",
        "无",
        "无品牌",
        "不知道",
        "unclear",
        "not visible",
        "n\\a",
    }


def _pg_lookup_products_for_image_analysis(
    product_name_en: str,
    brand_en: str,
    product_name: str,
    brand: str,
) -> List[tuple]:
    """
    按品牌/产品名模糊查 products（PostgreSQL，与 graph_logic / agent_logic 同源）。
    返回 (name, brand, allergens, ingredients) 行列表。
    """
    clauses: List[str] = []
    bind: dict = {}
    i = 0

    def add_like(col: str, val: str) -> None:
        nonlocal i
        if _is_placeholder_search_term(val):
            return
        key = f"p{i}"
        clauses.append(f"{col} ILIKE :{key}")
        bind[key] = f"%{val}%"
        i += 1

    if brand_en and brand_en != brand:
        add_like("brand", brand_en)
    if product_name_en and product_name_en != product_name:
        add_like("name", product_name_en)
    if brand and brand != brand_en:
        add_like("brand", brand)
    if product_name and product_name != product_name_en:
        add_like("name", product_name)

    if not clauses:
        return []

    where_sql = " OR ".join(clauses)
    stmt = text(
        f"""
        SELECT name, brand, allergens, ingredients
        FROM products
        WHERE {where_sql}
        ORDER BY length(coalesce(ingredients, '')) DESC NULLS LAST
        LIMIT 5
        """
    )
    with get_engine().connect() as conn:
        rows = conn.execute(stmt, bind).fetchall()
    return list(rows)


# 图片分析函数
def analyze_food_image(image_path: str) -> dict:
    """
    分析食物图片：
    1. GPT-4o 识别中/英检索字段（同轮输出，不再单独调用 mini 翻译）
    2. PostgreSQL products 模糊查询过敏原、配料等

    返回：包含分析结果、查询路径和耗时的字典
    """
    import time
    start_time = time.time()
    query_path = []
    
    try:
        import base64
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage

        # 读取和编码图片
        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        # 第一步：用GPT-4o识别图片中的产品
        step1_start = time.time()
        llm = ChatOpenAI(model="gpt-4o", temperature=0)

        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
                {
                    "type": "text",
                    "text": """请分析这张食物图片，输出格式如下（只输出这些信息，不要其他内容）：
产品名称: [名称]
品牌: [品牌；看不清时写「不详」]
英文检索名: [2～10 个词的英文，供数据库模糊检索，直译即可；无法判断写 N/A]
英文品牌: [品牌英文名；无则写 N/A]""",
                },
            ]
        )

        response = llm.invoke([message])
        recognition_result = response.content if hasattr(response, 'content') else str(response)
        
        step1_time = time.time() - step1_start
        query_path.append(f"🖼️ GPT-4o 图片识别 ({step1_time:.2f}s)")
        product_name = ""
        brand = ""
        product_name_en = ""
        brand_en = ""
        for line in recognition_result.split("\n"):
            line = line.strip()
            if "英文检索名:" in line or "英文检索名：" in line:
                sep = "英文检索名:" if "英文检索名:" in line else "英文检索名："
                product_name_en = line.split(sep, 1)[1].strip()
            elif "英文品牌:" in line or "英文品牌：" in line:
                sep = "英文品牌:" if "英文品牌:" in line else "英文品牌："
                brand_en = line.split(sep, 1)[1].strip()
            elif "产品名称:" in line or "产品名称：" in line:
                sep = "产品名称:" if "产品名称:" in line else "产品名称："
                product_name = line.split(sep, 1)[1].strip()
            elif line.startswith("品牌:") or line.startswith("品牌："):
                sep = "品牌:" if line.startswith("品牌:") else "品牌："
                brand = line.split(sep, 1)[1].strip()
        if not product_name_en or _is_placeholder_search_term(product_name_en):
            product_name_en = product_name
        if not brand_en or _is_placeholder_search_term(brand_en):
            brand_en = brand

        # 不再单独调用 mini 做翻译，省一次往返（约 2s+）；检索仍用中/英多路 ILIKE
        db_result = None
        
        if product_name_en or brand_en:
            try:
                step3_start = time.time()
                results = _pg_lookup_products_for_image_analysis(
                    product_name_en, brand_en, product_name, brand
                )

                if results:
                    db_result = "**找到的产品信息：**\n"
                    for idx, row in enumerate(results, 1):
                        name, brand_name, allergens, ingredients = row
                        db_result += f"\n【产品 {idx}】\n"
                        db_result += f"名称: {name}\n"
                        if brand_name:
                            db_result += f"品牌: {brand_name}\n"
                        if allergens:
                            db_result += f"过敏原: {allergens}\n"
                        if ingredients:
                            ing = str(ingredients)
                            db_result += f"成分: {ing[:200]}{'...' if len(ing) > 200 else ''}\n"

                step3_time = time.time() - step3_start
                if db_result:
                    query_path.append(f"💾 数据库查询命中 ({step3_time:.2f}s)")
                else:
                    query_path.append(f"💾 数据库查询未命中 ({step3_time:.2f}s)")
            
            except Exception as e:
                print(f"Database query error: {e}")
                query_path.append(f"❌ 数据库查询出错")
        
        # 组织返回结果 - 按照 DeepSeek 风格显示查询过程 + 最终答案
        result_text = "**🔍 查询过程：**\n\n"
        
        # 第一步：图片识别
        result_text += f"**第 1 步：图片识别**\n"
        result_text += f"→ {query_path[0]}\n"
        result_text += f"识别结果：产品名称 **{product_name}**，品牌 **{brand}**\n"
        if (product_name_en and product_name_en != product_name) or (
            brand_en and brand_en != brand and not _is_placeholder_search_term(brand_en)
        ):
            result_text += f"同轮给出的检索用英文：**{product_name_en}** / **{brand_en}**（无二次模型调用）\n"
        result_text += "\n"

        step_num = 2
        if len(query_path) > 1:
            for path_item in query_path[1:]:
                if "数据库" in path_item:
                    result_text += f"**第 {step_num} 步：数据库查询**\n"
                    result_text += f"→ {path_item}\n"
                    if db_result:
                        result_text += f"查询结果：✅ 命中\n\n"
                    else:
                        result_text += f"查询结果：❌ 未命中\n\n"
                    break
        
        # 分隔线 - 最终答案
        result_text += "═" * 50 + "\n\n"
        result_text += "**📋 最终结果：**\n\n"
        
        if db_result:
            result_text += db_result
        else:
            result_text += f"⚠️ **该产品暂未在数据库中找到详细信息**\n\n"
            result_text += f"该数据库主要包含来自开源食品数据库的英文产品信息。您可以：\n"
            result_text += f"1. 使用文字搜索来查询类似产品或品牌\n"
            result_text += f"2. 查询同品牌的其他产品\n"
            result_text += f"3. 搜索该产品类别的相关产品获取过敏原信息"
        
        total_time = time.time() - start_time
        result_text += f"\n\n⏱️ **总耗时:** {total_time:.2f}s"
        
        db_found = db_result is not None
        
        return {
            "response": result_text,
            "query_path": query_path,
            "elapsed_time": total_time,
            "product_name": product_name,
            "brand": brand,
            "found_in_db": db_found
        }
    
    except Exception as e:
        print(f"Image analysis error: {e}")
        import traceback
        traceback.print_exc()
        total_time = time.time() - start_time
        return {
            "response": f"图片分析失败: {str(e)}",
            "query_path": [f"❌ 分析失败"],
            "elapsed_time": total_time,
            "product_name": "",
            "brand": "",
            "found_in_db": False
        }

class Message(BaseModel):
    text: str
    role: str  # "user" or "assistant"


class ConversationInfo(BaseModel):
    id: str
    title: str
    messages: List[dict]


class QueryRequest(BaseModel):
    text: str
    conversation_id: Optional[str] = None
    image_base64: Optional[str] = None


def _decode_optional_image_b64(raw: Optional[str]) -> Optional[bytes]:
    if not raw or not isinstance(raw, str):
        return None
    try:
        s = raw.strip()
        if "," in s:
            s = s.split(",", 1)[1]
        return base64.b64decode(s, validate=False)
    except Exception:
        return None


class BulkDeleteRequest(BaseModel):
    conversation_ids: List[str]


def _ndjson_line(obj: dict) -> bytes:
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


def _log_answer_provenance_api(payload: dict) -> None:
    """与 graph_logic 中 LOG_ANSWER_PROVENANCE 对齐，在 API 层打印降级路径来源。"""
    v = os.getenv("LOG_ANSWER_PROVENANCE", "true").strip().lower()
    if v in ("0", "false", "no", "off"):
        return
    print("[answer_provenance] " + json.dumps(payload, ensure_ascii=False), flush=True)


def _normalize_response_text(value) -> str:
    """Normalize response value to plain text for API schema."""
    if isinstance(value, tuple):
        return str(value[0]) if value else ""
    return value if isinstance(value, str) else str(value or "")


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    cached: bool = False
    reasoning_trace: List[dict] = Field(default_factory=list)
    total_seconds: Optional[float] = None


def _build_reasoning_trace(steps: List[Any]) -> List[dict]:
    """Convert graph steps to frontend reasoning_trace format (optional per-step seconds)."""
    labels = {
        "cache_hit": "缓存命中",
        "translate": "问题翻译",
        "retrieve": "向量检索",
        "plan_route": "路由规划",
        "grade": "文档相关性评估",
        "generate": "答案生成",
        "web_search": "联网补充检索",
        "end": "结果返回",
        "generate_error": "答案生成异常（已自动降级）",
        "retrieve_error": "检索异常（已自动降级）",
        "web_search_error": "联网检索异常（已自动降级）",
    }
    trace: List[dict] = []
    for i, item in enumerate(steps):
        if isinstance(item, dict):
            node = (item.get("node") or "").strip()
            sec = item.get("step_seconds")
        else:
            node = str(item).strip()
            sec = None
        if not node or node == "end":
            continue
        if node.endswith("_error") and node not in labels:
            base = node.replace("_error", "")
            base_label = labels.get(base, base)
            label = f"{base_label}异常（已自动降级）"
        else:
            label = labels.get(node, node.replace("_", " "))
        entry: dict = {
            "key": f"{node}-{i}",
            "label": label,
        }
        if isinstance(sec, (int, float)) and sec == sec:  # not NaN
            entry["seconds"] = float(sec)
        trace.append(entry)
    return trace


# ============= 应用初始化 =============


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Food Allergy AI Agent API Server started")
    log_langsmith_startup_status()
    try:
        from graph_logic import warm_retrieval_stack

        warm_retrieval_stack()
    except Exception as e:
        print(f"⚠️ Retrieval warm-up skipped: {e}", flush=True)
    yield
    print("🛑 Server shutting down")


app = FastAPI(
    title="Food Allergy AI Agent API",
    description="Backend API for the Food Allergy AI Agent",
    version="1.0.0",
    lifespan=lifespan,
)


def _cors_allow_origins() -> tuple[list[str], bool]:
    """
    CORS_ALLOW_ORIGINS：逗号分隔，如 https://app.example.com
    未设置时用 *（开发）；生产务必设置且与浏览器访问前端的 Origin 一致。
    浏览器不允许 * 与 credentials 同时生效，故通配时不开启 credentials。
    """
    raw = (os.getenv("CORS_ALLOW_ORIGINS") or "").strip()
    if not raw:
        return ["*"], False
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    if not origins:
        return ["*"], False
    if origins == ["*"]:
        return ["*"], False
    return origins, True


_cors_origins, _cors_credentials = _cors_allow_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """HF Space 根路径：聊天 UI 在独立部署的 Next.js（frontend/）；此处进入 API 文档。"""
    return RedirectResponse(url="/docs", status_code=302)


# ============= 对话管理端点 =============


@app.get("/api/conversations")
async def list_conversations():
    """获取所有对话列表"""
    try:
        conversations = load_conversations()
        return {"conversations": conversations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """获取单个对话的详细信息"""
    try:
        conv = find_conversation_by_id(conversation_id)
        if not conv:
            raise HTTPException(
                status_code=404, detail="Conversation not found")
        return {"conversation": conv}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/conversations")
async def create_new_conversation(title: Optional[str] = None):
    """创建新对话"""
    try:
        conv = create_conversation(title=title or "New conversation")
        save_conversation(conv)
        return {"conversation": conv}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """删除对话"""
    try:
        delete_conversation_store(conversation_id)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/conversations/bulk-delete")
async def bulk_delete_conversations(request: BulkDeleteRequest):
    """批量删除对话（PG + JSON 双删）"""
    try:
        ids = [cid.strip() for cid in request.conversation_ids if cid and cid.strip()]
        if not ids:
            return {"success": True, "deleted_count": 0}
        for cid in ids:
            delete_conversation_store(cid)
        return {"success": True, "deleted_count": len(ids)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============= 用户设置端点 =============

class UserSettings(BaseModel):
    language: Optional[str] = "auto"  # auto, chinese, english
    theme: Optional[str] = "light"
    cache_enabled: Optional[bool] = True

user_settings_file = "data/user_settings.json"

def load_user_settings():
    """加载用户设置"""
    try:
        if os.path.exists(user_settings_file):
            with open(user_settings_file, "r") as f:
                return json.load(f)
    except:
        pass
    return {"language": "auto", "theme": "light", "cache_enabled": True}

def save_user_settings(settings):
    """保存用户设置"""
    os.makedirs("data", exist_ok=True)
    with open(user_settings_file, "w") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)

@app.get("/api/settings")
async def get_settings():
    """获取用户设置"""
    try:
        settings = load_user_settings()
        return settings
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/settings")
async def update_settings(settings: UserSettings):
    """更新用户设置"""
    try:
        settings_dict = settings.dict(exclude_none=True)
        current = load_user_settings()
        current.update(settings_dict)
        save_user_settings(current)
        return current
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============= 聊天和查询端点 =============


@app.post("/api/chat")
async def chat(request: QueryRequest):
    """处理用户查询并返回 AI 响应"""
    try:
        prompt = request.text
        conversation_id = request.conversation_id or None
        image_bytes = _decode_optional_image_b64(request.image_base64)

        # 如果没有对话 ID，创建一个新的
        if not conversation_id:
            conv = create_conversation(
                title=prompt[:30] + ("..." if len(prompt) > 30 else ""))
            conversation_id = conv["id"]
            save_conversation(conv)

        # 获取对话记录
        conv = find_conversation_by_id(conversation_id)
        if not conv:
            raise HTTPException(
                status_code=404, detail="Conversation not found")

        # 调用 AI Agent（使用 graph_logic 或 agent_logic）
        response = ""
        cached = False
        query_path_info = ""
        visited_trace: List[dict] = []
        
        try:
            import time
            query_start = time.time()
            
            for step in query_with_graph(prompt, image_bytes):
                node = step.get("node", "")
                if node:
                    visited_trace.append(
                        {
                            "node": node,
                            "step_seconds": step.get("step_seconds"),
                        }
                    )
                if node == "cache_hit":
                    cached = True
                    query_path_info = "⚡ 缓存命中"
                elif node == "end":
                    response = _normalize_response_text(step.get("generation"))
                    if not cached:
                        query_path_info = "📊 实时查询"
                    break
            
            query_time = time.time() - query_start
            
            # 如果有查询路径信息，添加到响应中
            if query_path_info:
                response += f"\n\n**查询状态:** {query_path_info} ({query_time:.2f}s)"
        except Exception as e:
            # Graph 主链路失败时仍给用户答案：走 agent_logic 的简单查询。
            # 终端打印真实异常，便于排查「思考过程里一直显示生成异常」的根因。
            print(f"[api/chat] graph path failed, using query_text fallback: {e!r}", flush=True)
            traceback.print_exc()
            response = _normalize_response_text(query_text(prompt))
            query_path_info = "⚠️ 简单查询模式"
            _log_answer_provenance_api(
                {
                    "question_preview": (prompt or "")[:160],
                    "generation_source": "agent_logic_query_text",
                    "summary_zh": "LangGraph 失败降级：agent_logic.query_text，非完整向量/路由/联网图",
                }
            )

        response = _normalize_response_text(response)

        # 保存消息到对话
        conv["messages"].append(
            {"role": "user", "text": prompt, "ts": time.time()})
        conv["messages"].append(
            {"role": "assistant", "text": response, "ts": time.time()})
        save_conversation(conv)

        # 记录到 Redis（后台自动持久化）
        try:
            record_important_qa(
                prompt, response, source="api", conversation_id=conversation_id
            )
        except Exception as e:
            print(f"Redis persistence failed: {e}")

        return ChatResponse(
            response=response,
            conversation_id=conversation_id,
            cached=cached,
            reasoning_trace=_build_reasoning_trace(visited_trace),
            total_seconds=query_time if "query_time" in locals() else None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(request: QueryRequest):
    """
    NDJSON 流式响应：type=start | step | delta | done | error。
    step 更新思考过程；delta 为答案正文增量（LLM stream）；done 含完整 response 与 reasoning_trace。
    """
    prompt = request.text
    conversation_id = request.conversation_id or None
    image_bytes = _decode_optional_image_b64(request.image_base64)

    def body_iter():
        cid = conversation_id
        try:
            if not cid:
                conv = create_conversation(
                    title=prompt[:30] + ("..." if len(prompt) > 30 else "")
                )
                cid = conv["id"]
                save_conversation(conv)

            conv = find_conversation_by_id(cid)
            if not conv:
                yield _ndjson_line({"type": "error", "message": "Conversation not found"})
                return

            yield _ndjson_line({"type": "start", "conversation_id": cid})
            # 首包：让前端立刻展开「思考过程」，避免首个 graph 步很慢时像卡住
            yield _ndjson_line(
                {
                    "type": "step",
                    "reasoning_trace": [],
                    "partial_total_seconds": 0.0,
                }
            )

            response = ""
            cached = False
            query_path_info = ""
            visited_trace: List[dict] = []
            query_start = time.time()

            try:
                use_llm_deltas = get_runtime_settings().enable_stream_llm_deltas
                for step in query_with_graph(
                    prompt,
                    image_bytes,
                    enable_answer_stream=use_llm_deltas,
                ):
                    if not isinstance(step, dict):
                        continue
                    node = (step.get("node") or "").strip()
                    if node == "__delta__":
                        chunk = step.get("delta") or ""
                        if chunk:
                            yield _ndjson_line({"type": "delta", "text": chunk})
                        continue
                    if node:
                        visited_trace.append(
                            {
                                "node": node,
                                "step_seconds": step.get("step_seconds"),
                            }
                        )
                    if node == "cache_hit":
                        cached = True
                        query_path_info = "⚡ 缓存命中"
                    elif node == "end":
                        response = _normalize_response_text(step.get("generation"))
                        if not cached:
                            query_path_info = "📊 实时查询"

                    yield _ndjson_line(
                        {
                            "type": "step",
                            "reasoning_trace": _build_reasoning_trace(visited_trace),
                            "partial_total_seconds": time.time() - query_start,
                        }
                    )
            except Exception as e:
                print(f"[api/chat/stream] graph path failed, fallback: {e!r}", flush=True)
                traceback.print_exc()
                response = _normalize_response_text(query_text(prompt))
                query_path_info = "⚠️ 简单查询模式"
                _log_answer_provenance_api(
                    {
                        "question_preview": (prompt or "")[:160],
                        "generation_source": "agent_logic_query_text",
                        "summary_zh": "LangGraph 失败降级：agent_logic.query_text，非完整向量/路由/联网图",
                    }
                )
                yield _ndjson_line(
                    {
                        "type": "step",
                        "reasoning_trace": _build_reasoning_trace(visited_trace),
                        "partial_total_seconds": time.time() - query_start,
                    }
                )

            query_time = time.time() - query_start
            response = _normalize_response_text(response)
            if query_path_info:
                response = (
                    response
                    + f"\n\n**查询状态:** {query_path_info} ({query_time:.2f}s)"
                )

            conv["messages"].append({"role": "user", "text": prompt, "ts": time.time()})
            conv["messages"].append(
                {"role": "assistant", "text": response, "ts": time.time()}
            )
            save_conversation(conv)
            try:
                record_important_qa(
                    prompt, response, source="api", conversation_id=cid
                )
            except Exception as e:
                print(f"Redis persistence failed: {e}")

            yield _ndjson_line(
                {
                    "type": "done",
                    "response": response,
                    "reasoning_trace": _build_reasoning_trace(visited_trace),
                    "total_seconds": query_time,
                    "cached": cached,
                    "conversation_id": cid,
                }
            )
        except Exception as e:
            yield _ndjson_line({"type": "error", "message": str(e)})

    return StreamingResponse(body_iter(), media_type="application/x-ndjson")


@app.post("/api/chat/save")
async def save_chat(conversation_id: str, user_prompt: str, ai_response: str):
    """将对话保存到永久存储"""
    try:
        success = move_to_permanent_storage(
            user_prompt, ai_response, source="api_save")
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============= 图片上传和分析 =============


@app.post("/api/upload-image")
async def upload_image(file: UploadFile = File(...), conversation_id: Optional[str] = None):
    """上传食品图片并进行分析"""
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, detail="File must be an image")

        # 读取图片数据
        image_bytes = await file.read()

        # 保存图片到临时目录用于分析
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        # 创建新对话或使用现有的
        if not conversation_id:
            conv = create_conversation(title=f"📸 {file.filename}")
            conversation_id = conv["id"]
            save_conversation(conv)
        else:
            conv = find_conversation_by_id(conversation_id)
            if not conv:
                raise HTTPException(
                    status_code=404, detail="Conversation not found")

        # 调用图片分析
        analysis_result = {}
        try:
            # 使用GPT-4o分析图片
            analysis_result = analyze_food_image(tmp_path)
            response = analysis_result.get("response", "")
        except Exception as e:
            # 失败则返回错误信息
            response = f"图片分析失败: {str(e)}"
            analysis_result = {
                "response": response,
                "query_path": ["❌ 分析失败"],
                "elapsed_time": 0,
                "product_name": "",
                "brand": "",
                "found_in_db": False
            }
        finally:
            # 删除临时文件
            try:
                os.unlink(tmp_path)
            except:
                pass

        # 构建完整响应（包含查询路径）
        full_response = response
        
        # 添加查询路径信息
        if analysis_result.get("query_path"):
            query_path_text = "\n\n**查询路径：**\n"
            for path_item in analysis_result["query_path"]:
                query_path_text += f"→ {path_item}\n"
            full_response += query_path_text
        
        # 添加耗时信息
        elapsed = analysis_result.get("elapsed_time", 0)
        full_response += f"\n**耗时:** {elapsed:.2f}s"
        
        # 保存到对话
        conv["messages"].append(
            {"role": "user", "text": f"📸 [上传图片: {file.filename}]", "ts": time.time()})
        conv["messages"].append(
            {"role": "assistant", "text": full_response, "ts": time.time()})
        save_conversation(conv)

        # 返回格式要与前端期望的ChatResponse一致
        return {
            "response": response,
            "conversation_id": conversation_id,
            "cached": False,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============= 健康检查 =============


@app.get("/api/health")
async def health_check():
    """服务健康检查"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
    }


# ============= WebSocket（实时流式响应） =============


@app.websocket("/ws/chat/{conversation_id}")
async def websocket_chat(websocket: WebSocket, conversation_id: str):
    """WebSocket 端点用于实时聊天流"""
    await websocket.accept()
    try:
        while True:
            # 接收来自客户端的消息
            data = await websocket.receive_json()
            prompt = data.get("text", "")

            if not prompt:
                continue

            # 获取或创建对话
            conv = find_conversation_by_id(conversation_id)
            if not conv:
                conv = create_conversation()
                conversation_id = conv["id"]
                save_conversation(conv)

            # 添加用户消息
            conv["messages"].append(
                {"role": "user", "text": prompt, "ts": time.time()})
            save_conversation(conv)

            # 流式发送 AI 响应
            full_response = ""
            for step in query_with_graph(prompt):
                if step["node"] == "end":
                    full_response = _normalize_response_text(step.get("generation"))
                    # 发送最终响应
                    await websocket.send_json({
                        "type": "response",
                        "content": full_response,
                    })
                else:
                    # 发送进度更新
                    await websocket.send_json({
                        "type": "progress",
                        "node": step.get("node"),
                        "content": step.get("content", ""),
                        "step_seconds": step.get("step_seconds"),
                    })

            # 保存 AI 响应
            conv["messages"].append(
                {"role": "assistant", "text": full_response, "ts": time.time()})
            save_conversation(conv)

            # 记录到 Redis
            try:
                record_important_qa(
                    prompt,
                    full_response,
                    source="websocket",
                    conversation_id=conversation_id,
                )
            except Exception as e:
                print(f"Redis persistence failed: {e}")

    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close(code=1000)


# ============= 启动脚本 =============


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
