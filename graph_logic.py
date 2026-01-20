import os
import time
import base64
from typing import List, TypedDict, Annotated, Union
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
    embeddings = OpenAIEmbeddings()
    return Chroma(persist_directory="data/chroma_db", embedding_function=embeddings)


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

    msgs = StreamlitChatMessageHistory(key="messages")
    history = msgs.messages[:-1][-5:] if len(msgs.messages) > 1 else []

    if not history:
        return {"question": question}

    llm = get_fast_llm()
    system_instruction = """你是一个专业的问题重写专家。
        你的任务是：根据对话历史，将用户最新的提问改写为一个【完全独立、无歧义】的问题。
        
        【核心要求】
        1. 消除代词：必须将“这个”、“它”、“this”、“it”等词，替换为历史对话中提到的具体食品名称或品牌。
        2. 跨语言对齐：即使历史是中文而当前提问是英文（或反之），你也必须准确提取品牌名（如李锦记/Lee Kum Kee）并嵌入新问题中。
        3. 严禁偷懒：严禁输出类似 "this product" 或 "the sauce" 这种依然模糊的词，必须说出全名。
        4. 保持原意：不要回答问题，只需重写提问。直接输出重写后的结果。
        """

    # 获取目标语言
    target_lang = state.get("target_language", "自动识别 (Auto)")

    # 核心优化点：如果我们要生成 Key，使用一种极其死板的格式
    if target_lang == "Normalized_Key":
        system_instruction = """你是一个多语言实体对齐专家。
        你的任务：从对话历史中提取核心意图，并将其【强制统一翻译为英文】。
        
        必须输出此格式：[意图]|[英文品牌]|[英文产品名]
        意图分类：AllergyCheck, InfoSearch, Appearance
        翻译示例：
        - “李锦记” -> "Lee Kum Kee"
        - “老抽” -> "Dark Soy Sauce"
        - “能吃吗” -> "AllergyCheck"
        
        输出示例：AllergyCheck|Lee Kum Kee|Dark Soy Sauce
        严禁输出任何中文或多余单词。"""
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
    docs = get_vectorstore().similarity_search(state["question"], k=3)
    doc_texts = [
        f"内容: {d.page_content}\n来源: {d.metadata.get('source', '本地知识库')}" for d in docs]
    return {"documents": doc_texts}


def grade_documents(state):
    if not state.get("documents"):
        return {"web_search": "Yes"}
    llm = get_fast_llm()
    res = (ChatPromptTemplate.from_messages([("system", "判断资料是否足以回答问题。"), ("human", "问题: {question} \n资料: {documents}")]) | llm.with_structured_output(
        grade_schema)).invoke({"question": state["question"], "documents": ' '.join(state["documents"])})
    score = res["score"] if isinstance(res, dict) else res.score
    return {"web_search": "No" if score == "yes" else "Yes"}


def generate(state):
    print("--- 生成回答 ---")
    retry_count = state.get("retry_count", 0) + 1
    target_lang = state.get("target_language", "自动识别 (Auto)")
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
    llm = get_fast_llm()
    res = (ChatPromptTemplate.from_messages([("system", "判断回答是否基于参考资料。"), ("human", "资料: {documents} \n回答: {generation}")]) | llm.with_structured_output(
        grade_schema)).invoke({"documents": ' '.join(state["documents"]), "generation": state["generation"]})
    return {"hallucination_score": res["score"] if isinstance(res, dict) else res.score}


def answer_grader(state):
    llm = get_fast_llm()
    res = (ChatPromptTemplate.from_messages([("system", "判断回答是否解决了用户问题。"), ("human", "问题: {question} \n回答: {generation}")]) | llm.with_structured_output(
        grade_schema)).invoke({"question": state["question"], "generation": state["generation"]})
    return {"answer_score": res["score"] if isinstance(res, dict) else res.score}

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
workflow.add_node("hallucination_grader", hallucination_grader)
workflow.add_node("answer_grader", answer_grader)

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
workflow.add_edge("generate", "hallucination_grader")
workflow.add_edge("hallucination_grader", "answer_grader")
workflow.add_conditional_edges("answer_grader", dec_final, {
                               "useful": END, "not useful": "web_search", "not supported": "generate"})

app = workflow.compile()

# --- 6. 统一查询入口 ---


def query_with_graph(question: str, image_bytes: bytes = None):
    start_time = time.time()
    target_lang = st.session_state.get("target_language", "自动识别 (Auto)")

    # 1. 确保缓存已初始化
    if "response_cache" not in st.session_state:
        st.session_state.response_cache = {}

    # 2. 图片识别逻辑 (独立运行)
    if image_bytes:
        from agent_logic import query_text as vision_query
        res, dur = vision_query(question, image_bytes=image_bytes)
        yield {"node": "end", "generation": res, "duration": dur}
        return

    # 3. 生成语义指纹作为缓存 Key (统一归一化为英文)
    # 这一步是关键：让“能喝吗”和“Can I drink”在后台都生成相同的英文句子
    fingerprint_res = contextualize_question(
        {"question": question, "target_language": "Normalized_Key"})
    semantic_key = fingerprint_res.get("question", "").lower().strip()

    # --- 打印观察结果 ---
    print(f"\n{'='*20} [SEMANTIC CACHE] {'='*20}")
    print(f"【捕获指纹】: {semantic_key}")
    # 预期输出：allergycheck|lee kum kee|dark soy sauce
    print(f"{'='*55}\n")

    display_q_res = contextualize_question(
        {"question": question, "target_language": target_lang})
    refined_q = display_q_res["question"]
    # --- 打印 Key 供你观察 ---
    print(f"【标准化语义 Key】: {semantic_key}")

    # 4. 生成展示用的补全意图 (用于 UI 显示)
    display_q_res = contextualize_question(
        {"question": question, "target_language": target_lang})
    refined_q = display_q_res.get("question", question)
    yield {"node": "contextualize_question", "status": "complete", "refined_q": refined_q}

    # 5. 语义级缓存检查
    if semantic_key in st.session_state.response_cache:
        yield {"node": "cache_hit", "status": "complete"}
        final_res = st.session_state.response_cache[semantic_key]
    else:
        # 6. 缓存未命中：执行正式工作流
        final_res = "抱歉，由于逻辑异常。"
        # 传入补全后的问题，并跳过工作流内部的重复补全节点
        for event in app.stream({"question": refined_q, "target_language": target_lang, "retry_count": 0}, stream_mode="updates"):
            for node_name, output in event.items():
                if node_name == "contextualize_question":
                    continue
                yield {"node": node_name, "status": "running"}
                if "generation" in output:
                    final_res = output["generation"]

        # 将结果存入缓存 (使用语义指纹)
        st.session_state.response_cache[semantic_key] = final_res

    yield {"node": "end", "generation": final_res, "duration": time.time() - start_time}
