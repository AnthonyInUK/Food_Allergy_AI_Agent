import os
import time
import base64
import hashlib
import asyncio
from typing import List, TypedDict, Annotated, Union
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st
from langgraph.graph import END, StateGraph, START

from agent_logic import get_sql_agent, get_db, get_llm, query_text as sql_query_text

load_dotenv()

# --- 1. 资源缓存池 ---


@st.cache_resource
def get_fast_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


@st.cache_resource
def get_vectorstore():
    import os
    try:
        embeddings = OpenAIEmbeddings()
        persist_dir = "data/chroma_db"
        # Ensure directory exists
        os.makedirs(persist_dir, exist_ok=True)
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    except Exception as e:
        print(f"Warning: Failed to initialize vectorstore: {e}")
        # Return None and handle gracefully in query logic
        return None


# --- 缓存辅助函数 ---
def get_semantic_hash(text: str) -> str:
    """将语义键转换为 MD5 哈希，提高缓存键效率"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def init_cache_system():
    """初始化多层缓存系统"""
    if "response_cache" not in st.session_state:
        st.session_state.response_cache = {}  # L1: 最终答案缓存（语义级，跨语言）
    if "retrieval_cache" not in st.session_state:
        st.session_state.retrieval_cache = {}  # L2: 向量检索结果缓存
    if "generation_cache" not in st.session_state:
        st.session_state.generation_cache = {}  # L3: LLM 生成结果缓存
    if "grade_cache" not in st.session_state:
        st.session_state.grade_cache = {}  # L4: 文档评估缓存
    if "hallucination_cache" not in st.session_state:
        st.session_state.hallucination_cache = {}  # L5: 幻觉检测缓存
    if "answer_grade_cache" not in st.session_state:
        st.session_state.answer_grade_cache = {}  # L6: 答案质量评估缓存
    if "cache_stats" not in st.session_state:
        st.session_state.cache_stats = {"hits": 0, "misses": 0}  # 缓存统计


def get_cache_stats():
    """获取缓存命中率统计"""
    stats = st.session_state.get("cache_stats", {"hits": 0, "misses": 0})
    total = stats["hits"] + stats["misses"]
    hit_rate = (stats["hits"] / total * 100) if total > 0 else 0
    return {"hit_rate": hit_rate, "total_queries": total, **stats}


def clear_all_caches():
    """清空所有缓存（用于调试或释放内存）"""
    cache_types = [
        "response_cache",
        "retrieval_cache",
        "generation_cache",
        "grade_cache",
        "hallucination_cache",
        "answer_grade_cache"
    ]

    cleared_count = 0
    for cache_name in cache_types:
        if cache_name in st.session_state:
            st.session_state[cache_name].clear()
            cleared_count += 1

    if "cache_stats" in st.session_state:
        st.session_state.cache_stats = {"hits": 0, "misses": 0}

    print(f"✓ 已清空 {cleared_count} 个缓存层")


# --- 2. 结构化输出 Schema ---
route_schema = {
    "name": "route_query",
    "description": "判定用户查询的意图和领域范围",
    "parameters": {
        "type": "object",
        "properties": {
            "datasource": {"type": "string", "enum": ["sql_db", "vector_db", "off_topic"]}
        },
        "required": ["datasource"]
    }
}

grade_schema = {
    "name": "grade",
    "parameters": {
        "type": "object",
        "properties": {"score": {"type": "string", "enum": ["yes", "no"]}},
        "required": ["score"]
    }
}

# --- 3. 状态定义 ---


class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[str]
    router_decision: str
    hallucination_score: str
    answer_score: str
    retry_count: int
    target_language: str

# --- 4. 节点逻辑 ---


def contextualize_question(state):
    print("--- 🚦 正在进行上下文补全 ---")
    question = state["question"]
    target_lang = state.get("target_language", "自动识别 (Auto)")

    msgs = StreamlitChatMessageHistory(key="messages")
    history = msgs.messages[:-1][-5:] if len(msgs.messages) > 1 else []

    # 🔧 修复：Normalized_Key 模式下即使没有历史也要转换
    if not history and target_lang != "Normalized_Key":
        return {"question": question}

    llm = get_fast_llm()
    system_instruction = """你是一个专业的问题重写专家。
        你的任务是：根据对话历史，将用户最新的提问改写为一个【完全独立、无歧义】的问题。
        
        【核心要求】
        1. 消除代词：必须将"这个"、"它"、"this"、"it"等词，替换为历史对话中提到的具体食品名称或品牌。
        2. 跨语言对齐：即使历史是中文而当前提问是英文（或反之），你也必须准确提取品牌名（如李锦记/Lee Kum Kee）并嵌入新问题中。
        3. 严禁偷懒：严禁输出类似 "this product" 或 "the sauce" 这种依然模糊的词，必须说出全名。
        4. 保持原意：不要回答问题，只需重写提问。直接输出重写后的结果。
        """

    # 核心优化点：如果我们要生成 Key，使用一种极其死板的格式
    if target_lang == "Normalized_Key":
        system_instruction = """你是一个多语言实体对齐专家。
        你的任务：提取问题中的核心意图，并【强制统一翻译为英文标准格式】。
        
        必须输出此格式：[意图]|[英文品牌]|[英文产品名]
        意图分类：AllergyCheck, InfoSearch, Appearance, Compare, List
        翻译示例：
        - "李锦记" -> "Lee Kum Kee"
        - "老抽" -> "Dark Soy Sauce"
        - "能吃吗" / "can i eat" -> "AllergyCheck"
        - "长什么样" / "look like" -> "Appearance"
        - "对比" / "compare" -> "Compare"
        
        输出示例：
        - "我对大豆过敏，能喝李锦记老抽吗" -> AllergyCheck|Lee Kum Kee|Dark Soy Sauce
        - "I'm allergic to soy. Can I have Lee Kum Kee dark soy sauce?" -> AllergyCheck|Lee Kum Kee|Dark Soy Sauce
        
        ⚠️ 严禁输出任何中文或多余单词！必须完全按照格式输出！"""
    else:
        # 用于 UI 展示的提示词保持原有的灵活性
        system_instruction = "你是一个问题重写专家。根据对话历史，将提问改写为独立的完整提问。处理代词指代。"

    if target_lang == "English":
        system_instruction += " 请务必使用【英文】重写问题。"
    elif target_lang == "Français":
        system_instruction += " 请务必使用【法语】重写问题。"
    elif target_lang == "自动识别 (Auto)":
        system_instruction += " 请保持与用户提问相同的语言。"
    else:
        system_instruction += " 请使用【中文】重写问题。"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])
    res = (prompt | llm).invoke({"history": history, "question": question})
    print(f"【问题补全结果】: {res.content}")
    return {"question": res.content}


def route_question(state):
    """【100% 还原你最满意的高精度路由提示词】"""
    print("--- 智能路由与安全网关 ---")
    llm = get_fast_llm()
    structured_llm = llm.with_structured_output(route_schema)

    system = """你是一个高精度的食品安全路由专家。
    【核心分发规则 - 必须死守】
    1. 必须选 'sql_db' (精准查询)：
       - 问题中包含【具体品牌】（如：李锦记、Lee Kum Kee、康师傅、海天等），必须选此项。
       - 即便是问“能吃吗”、“含大豆吗”，只要有品牌，也必须路由到 'sql_db'。
       - 问题涉及【特定产品名】（如：老抽、酱油、番茄酱）。
       - 问题要求看【图片/外观/长什么样/包装】。
       - 涉及统计或列表（如：有哪些不含大豆的酱？）。
    
    2. 必须选 'vector_db' (常识/建议)：
    
     - 仅当问题是【一般性知识】（如：什么是面筋？）或【完全没有品牌名】的模糊咨询。
       - 关于【成分知识】（如：什么是面筋？防腐剂有害吗？）。
       - 只要涉及食品、过敏、身体安全，必须在这 1 和 2 之间选择！
    
    3. 必须选 'off_topic' (严格拦截)：
       - 仅当问题与【食品、过敏、品牌、营养、安全】完全无关时。
       - 例如：问代码、政治、闲聊、问天气、问数学题等。
    """
    # 我们把这个作为“硬提示”塞给路由器
    question_to_route = state["question"]
    # 提取品牌名单（可以动态获取，这里示例几个核心品牌）
    known_brands = ["lee kum kee", "李锦记", "haday", "海天", "master kong", "康师傅"]

    # 如果问题里有这些词，我们在问题末尾强行加个提示
    if any(brand in question_to_route.lower() for brand in known_brands):
        question_to_route += " (Note: This question contains a specific brand, prioritize structured data source.)"

    res = (ChatPromptTemplate.from_messages(
        [("system", system), ("human", "{question}")]) | structured_llm).invoke({"question": state["question"]})
    decision = res["datasource"] if isinstance(res, dict) else res.datasource
    return {"router_decision": decision}


def handle_off_topic(state):
    print("--- 🚫 拦截非法请求 ---")
    target_lang = state.get("target_language", "自动识别 (Auto)")
    msg = "抱歉，我是一名专业的食品过敏助手。我只能回答与食品成分、过敏原、食品品牌以及食品安全相关的问题。"
    if target_lang == "English":
        msg = "Sorry, I am a professional Food Allergy Assistant. I can only answer questions related to food ingredients, allergens, brands, and safety."
    elif target_lang == "Français":
        msg = "Désolé, je suis un assistant professionnel pour les allergies alimentaires. Je ne peux répondre qu'aux questions relatives aux ingrédients, aux allergènes, aux marques et à la sécurité alimentaire."
    return {"generation": msg}


def call_sql_agent(state):
    print("--- 启动 SQL Agent ---")
    response, _ = sql_query_text(state["question"])
    return {"generation": response}


def retrieve(state):
    print("--- 检索本地知识库 ---")
    question = state["question"]

    # 检索缓存：对相同问题的向量检索结果进行缓存
    if "retrieval_cache" in st.session_state:
        cache_key = get_semantic_hash(question.lower().strip())
        if cache_key in st.session_state.retrieval_cache:
            print("  ✓ 命中检索缓存")
            return {"documents": st.session_state.retrieval_cache[cache_key]}

    docs = get_vectorstore().similarity_search(question, k=3)
    doc_texts = [
        f"内容: {d.page_content}\n来源: {d.metadata.get('source', '本地知识库')}" for d in docs]

    # 存入检索缓存
    if "retrieval_cache" in st.session_state:
        st.session_state.retrieval_cache[cache_key] = doc_texts

    return {"documents": doc_texts}


def grade_documents(state):
    """评估文档质量（优化版：对评估结果进行缓存）"""
    if not state.get("documents"):
        return {"web_search": "Yes"}

    # 初始化评估缓存
    if "grade_cache" not in st.session_state:
        st.session_state.grade_cache = {}

    # 生成缓存键：问题 + 文档内容
    question = state["question"]
    docs_text = ' '.join(state["documents"])
    cache_key = get_semantic_hash(f"grade|{question}|{docs_text}")

    # 检查缓存
    if cache_key in st.session_state.grade_cache:
        print("  ✓ 命中文档评估缓存")
        return {"web_search": st.session_state.grade_cache[cache_key]}

    # 缓存未命中：执行 LLM 评估
    llm = get_fast_llm()
    res = (ChatPromptTemplate.from_messages([("system", "判断资料是否足以回答问题。"), ("human", "问题: {question} \n资料: {documents}")]) | llm.with_structured_output(
        grade_schema)).invoke({"question": question, "documents": docs_text})
    score = res["score"] if isinstance(res, dict) else res.score
    result = "No" if score == "yes" else "Yes"

    # 存入缓存
    st.session_state.grade_cache[cache_key] = result

    return {"web_search": result}


def generate(state):
    print("--- 生成回答 ---")
    retry_count = state.get("retry_count", 0) + 1
    target_lang = state.get("target_language", "自动识别 (Auto)")

    # 生成缓存键：基于问题 + 文档内容 + 语言
    if "generation_cache" not in st.session_state:
        st.session_state.generation_cache = {}

    docs_text = ' '.join(state.get("documents", []))
    cache_key = get_semantic_hash(
        f"{state['question']}|{docs_text}|{target_lang}")

    # 检查生成缓存
    if cache_key in st.session_state.generation_cache:
        print("  ✓ 命中生成缓存（跳过 LLM 调用）")
        return {"generation": st.session_state.generation_cache[cache_key], "retry_count": retry_count}

    lang_instruction = ""
    if target_lang == "简体中文":
        lang_instruction = "请使用【简体中文】回答。"
    elif target_lang == "English":
        lang_instruction = "Please reply in 【English】."
    elif target_lang == "Français":
        lang_instruction = "Répondez en 【Français】."
    else:
        lang_instruction = "请使用用户提问的语言回答。"

    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        f"你是一个食品过敏专家。\n{lang_instruction}\n【重要】必须在末尾列出参考来源。\n资料: {{documents}}\n问题: {{question}}")
    response = (prompt | llm).invoke(
        {"documents": state["documents"], "question": state["question"]})

    # 存入生成缓存
    st.session_state.generation_cache[cache_key] = response.content

    return {"generation": response.content, "retry_count": retry_count}


def web_search(state):
    print("--- 触发联网搜索 ---")
    from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
    search = TavilySearchResults(api_wrapper=TavilySearchAPIWrapper(
        tavily_api_key=os.getenv("TAVILY_API_KEY")), k=3)
    results = search.invoke({"query": state["question"]})
    web_docs = [f"内容: {r['content']}\n来源: {r['url']}" for r in results]
    return {"documents": web_docs}


def hallucination_grader(state):
    """幻觉检测（优化版：对检测结果进行缓存）"""
    # 初始化幻觉检测缓存
    if "hallucination_cache" not in st.session_state:
        st.session_state.hallucination_cache = {}

    # 生成缓存键：文档 + 回答
    docs_text = ' '.join(state["documents"])
    generation = state["generation"]
    cache_key = get_semantic_hash(f"hallucination|{docs_text}|{generation}")

    # 检查缓存
    if cache_key in st.session_state.hallucination_cache:
        print("  ✓ 命中幻觉检测缓存")
        return {"hallucination_score": st.session_state.hallucination_cache[cache_key]}

    # 缓存未命中：执行 LLM 判断
    llm = get_fast_llm()
    res = (ChatPromptTemplate.from_messages([("system", "判断回答是否基于参考资料。"), ("human", "资料: {documents} \n回答: {generation}")]) | llm.with_structured_output(
        grade_schema)).invoke({"documents": docs_text, "generation": generation})
    score = res["score"] if isinstance(res, dict) else res.score

    # 存入缓存
    st.session_state.hallucination_cache[cache_key] = score

    return {"hallucination_score": score}


def answer_grader(state):
    """答案质量评估（优化版：对评估结果进行缓存）"""
    # 初始化答案评估缓存
    if "answer_grade_cache" not in st.session_state:
        st.session_state.answer_grade_cache = {}

    # 生成缓存键：问题 + 回答
    question = state["question"]
    generation = state["generation"]
    cache_key = get_semantic_hash(f"answer|{question}|{generation}")

    # 检查缓存
    if cache_key in st.session_state.answer_grade_cache:
        print("  ✓ 命中答案评估缓存")
        return {"answer_score": st.session_state.answer_grade_cache[cache_key]}

    # 缓存未命中：执行 LLM 判断
    llm = get_fast_llm()
    res = (ChatPromptTemplate.from_messages([("system", "判断回答是否解决了用户问题。"), ("human", "问题: {question} \n回答: {generation}")]) | llm.with_structured_output(
        grade_schema)).invoke({"question": question, "generation": generation})
    score = res["score"] if isinstance(res, dict) else res.score

    # 存入缓存
    st.session_state.answer_grade_cache[cache_key] = score

    return {"answer_score": score}


def parallel_graders(state):
    """🚀 并行执行幻觉检测和答案评估（节省 40-50% 时间）"""
    print("--- 🚀 并行执行质量评估 ---")

    # 使用线程池并行执行两个独立的评估
    with ThreadPoolExecutor(max_workers=2) as executor:
        # 同时提交两个任务
        future_hallucination = executor.submit(hallucination_grader, state)
        future_answer = executor.submit(answer_grader, state)

        # 等待两个任务完成
        hallucination_result = future_hallucination.result()
        answer_result = future_answer.result()

    print("  ✅ 并行评估完成")

    # 合并结果
    return {
        "hallucination_score": hallucination_result["hallucination_score"],
        "answer_score": answer_result["answer_score"]
    }

# --- 5. 构建工作流 ---


def dec_gen(
    state): return "web_search" if state["web_search"] == "Yes" else "generate"


def dec_final(state):
    if state.get("retry_count", 0) >= 2:
        return "useful"
    if state["hallucination_score"] == "no":
        return "not supported"
    return "useful" if state["answer_score"] == "yes" else "not useful"


workflow = StateGraph(GraphState)
workflow.add_node("contextualize_question", contextualize_question)
workflow.add_node("route_question", route_question)
workflow.add_node("handle_off_topic", handle_off_topic)
workflow.add_node("sql_agent", call_sql_agent)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("web_search", web_search)
workflow.add_node("parallel_graders", parallel_graders)  # 🚀 新增并行评估节点

workflow.add_edge(START, "contextualize_question")
workflow.add_edge("contextualize_question", "route_question")
workflow.add_conditional_edges("route_question", lambda x: x["router_decision"], {
                               "sql_db": "sql_agent", "vector_db": "retrieve", "off_topic": "handle_off_topic"})
workflow.add_edge("handle_off_topic", END)
workflow.add_edge("sql_agent", END)
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", dec_gen, {
                               "web_search": "web_search", "generate": "generate"})
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", "parallel_graders")  # 🚀 改为并行评估
workflow.add_conditional_edges("parallel_graders", dec_final, {  # 🚀 从并行节点决策
                               "useful": END, "not useful": "web_search", "not supported": "generate"})

app = workflow.compile()

# --- 6. 统一查询入口 ---


def query_with_graph(question: str, image_bytes: bytes = None):
    start_time = time.time()
    target_lang = st.session_state.get("target_language", "自动识别 (Auto)")

    # 1. 初始化多层缓存系统
    init_cache_system()

    # 2. 图片识别逻辑 (独立运行，不使用缓存)
    if image_bytes:
        from agent_logic import query_text as vision_query
        res, dur = vision_query(question, image_bytes=image_bytes)
        yield {"node": "end", "generation": res, "duration": dur}
        return

    # 🚀 3. 超快速路径检测：在工作流之前就拦截简单查询
    from agent_logic import query_text as direct_sql_query_func

    # 检测常见品牌 + 图片/过敏原关键词
    q_lower = question.lower()
    quick_brands = ["李锦记", "lee kum kee", "海天", "haday", "康师傅", "master kong"]
    quick_keywords = ["长什么样", "看图", "图片", "外观", "包装", "能吃", "过敏",
                      "look like", "picture", "image", "allerg", "safe"]

    # 排除关键词：这些问题需要 Agent 多跳推理
    # 使用单词边界避免误匹配（如 "all" 不应匹配 "allergic"）
    exclude_keywords = ["对比", "区别", "哪些", "列表", "比较", "所有", "全部", "有什么",
                        "compare", "difference", "list", " all ", "what are", "which"]

    has_brand = any(brand in q_lower for brand in quick_brands)
    has_keyword = any(kw in q_lower for kw in quick_keywords)
    has_exclude = any(kw in q_lower for kw in exclude_keywords)

    # 只有在满足条件且不包含排除关键词时才触发快速路径
    if has_brand and has_keyword and not has_exclude:
        print("  🚀🚀🚀 触发超快速路径：绕过工作流，直接查询")
        yield {"node": "fast_path_detected", "status": "activated"}

        # 生成语义指纹用于缓存（即使走快速路径也要缓存）
        fingerprint_res = contextualize_question(
            {"question": question, "target_language": "Normalized_Key"})
        semantic_text = fingerprint_res.get("question", "").lower().strip()
        semantic_key = get_semantic_hash(semantic_text)

        print(f"  📌 快速路径语义指纹: {semantic_text}")
        print(f"  📌 缓存键: {semantic_key}")

        # 调试：打印当前缓存状态
        if "response_cache" in st.session_state:
            cache_keys = list(st.session_state.response_cache.keys())
            print(f"  🔍 当前缓存中有 {len(cache_keys)} 条记录")
            if len(cache_keys) > 0:
                print(f"  🔍 缓存键列表: {cache_keys[:3]}")  # 只打印前3个

        # 检查缓存
        if "response_cache" in st.session_state and semantic_key in st.session_state.response_cache:
            print("  ✓✓✓ 快速路径也命中了语义缓存！")
            if "cache_stats" in st.session_state:
                st.session_state.cache_stats["hits"] += 1
            result = st.session_state.response_cache[semantic_key]
            yield {"node": "end", "generation": result, "duration": time.time() - start_time}
            return

        # 直接调用 SQL 查询，绕过整个工作流
        try:
            result, duration = direct_sql_query_func(question)
            # 存入缓存
            if "response_cache" not in st.session_state:
                st.session_state.response_cache = {}
            st.session_state.response_cache[semantic_key] = result
            print(f"  💾 已存入语义缓存（键: {semantic_key}）")
            if "cache_stats" in st.session_state:
                st.session_state.cache_stats["misses"] += 1

            yield {"node": "end", "generation": result, "duration": time.time() - start_time}
            return
        except Exception as e:
            print(f"  ⚠️ 快速路径失败: {e}，回退到正常流程")
            # 如果失败，继续正常流程

    # 🚀 多跳查询的快速通道：直接路由到 SQL Agent，跳过 route_question
    if has_exclude and has_brand:
        detected_keyword = [k for k in exclude_keywords if k in q_lower][0]
        print(f"  🤖 检测到复杂查询（包含'{detected_keyword}'），使用 Agent 多跳推理")
        yield {"node": "complex_query_detected", "keyword": detected_keyword}

        # 🚀 优化：对于多跳查询，也检查缓存（跳过前面的节点）
        fingerprint_res = contextualize_question(
            {"question": question, "target_language": "Normalized_Key"})
        semantic_text = fingerprint_res.get("question", "").lower().strip()
        semantic_key = get_semantic_hash(semantic_text)

        print(f"  📌 多跳查询语义指纹: {semantic_text}")
        print(f"  📌 缓存键: {semantic_key}")

        # 检查缓存
        if "response_cache" in st.session_state and semantic_key in st.session_state.response_cache:
            print("  ✓✓✓ 多跳查询命中了语义缓存！跳过完整工作流")
            if "cache_stats" in st.session_state:
                st.session_state.cache_stats["hits"] += 1
            result = st.session_state.response_cache[semantic_key]
            yield {"node": "end", "generation": result, "duration": time.time() - start_time}
            return

        # 缓存未命中：生成展示用的补全意图，然后直接执行 SQL Agent
        print("  ✗ 多跳缓存未命中，执行 SQL Agent（跳过route_question）")
        if "cache_stats" in st.session_state:
            st.session_state.cache_stats["misses"] += 1

        display_q_res = contextualize_question(
            {"question": question, "target_language": target_lang})
        refined_q = display_q_res.get("question", question)
        yield {"node": "contextualize_question", "status": "complete", "refined_q": refined_q}

        # 直接调用 SQL Agent（跳过 route_question，节省0.5-1秒）
        yield {"node": "sql_agent", "status": "running"}
        response, _ = sql_query_text(refined_q)

        # 存入缓存
        if "response_cache" not in st.session_state:
            st.session_state.response_cache = {}
        st.session_state.response_cache[semantic_key] = response
        print(f"  💾 已存入多跳查询缓存（键: {semantic_key}）")

        yield {"node": "end", "generation": response, "duration": time.time() - start_time}
        return

    # 4. 生成语义指纹作为缓存 Key (统一归一化为英文)
    # 这一步是关键：让"能喝吗"和"Can I drink"在后台都生成相同的英文句子
    fingerprint_res = contextualize_question(
        {"question": question, "target_language": "Normalized_Key"})
    semantic_text = fingerprint_res.get("question", "").lower().strip()

    # 将语义文本转换为哈希，提升缓存键查找效率
    semantic_key = get_semantic_hash(semantic_text)

    # --- 打印观察结果 ---
    print(f"\n{'='*20} [SEMANTIC CACHE] {'='*20}")
    print(f"【语义指纹】: {semantic_text}")
    print(f"【缓存键】: {semantic_key}")
    print(f"{'='*55}\n")

    # 5. 生成展示用的补全意图 (仅调用一次，避免重复)
    display_q_res = contextualize_question(
        {"question": question, "target_language": target_lang})
    refined_q = display_q_res.get("question", question)
    yield {"node": "contextualize_question", "status": "complete", "refined_q": refined_q}

    # 6. 语义级缓存检查
    if semantic_key in st.session_state.response_cache:
        print("  ✓✓✓ 命中语义缓存！跳过工作流执行")
        st.session_state.cache_stats["hits"] += 1
        yield {"node": "cache_hit", "status": "complete"}
        final_res = st.session_state.response_cache[semantic_key]
    else:
        # 7. 缓存未命中：执行正式工作流
        print("  ✗ 缓存未命中，执行完整工作流")
        st.session_state.cache_stats["misses"] += 1
        final_res = "抱歉，由于逻辑异常。"

        # 传入补全后的问题，并跳过工作流内部的重复补全节点
        for event in app.stream({"question": refined_q, "target_language": target_lang, "retry_count": 0}, stream_mode="updates"):
            for node_name, output in event.items():
                if node_name == "contextualize_question":
                    continue
                yield {"node": node_name, "status": "running"}
                if "generation" in output:
                    final_res = output["generation"]

        # 将结果存入缓存 (使用哈希键)
        st.session_state.response_cache[semantic_key] = final_res

    # 8. 打印缓存统计
    stats = get_cache_stats()
    print(
        f"📊 缓存命中率: {stats['hit_rate']:.1f}% ({stats['hits']}/{stats['total_queries']})")

    yield {"node": "end", "generation": final_res, "duration": time.time() - start_time}
