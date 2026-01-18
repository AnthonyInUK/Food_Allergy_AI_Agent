import os
import time
import base64
from typing import List, TypedDict, Annotated, Union
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from langgraph.graph import END, StateGraph, START

from agent_logic import get_sql_agent, get_db, get_llm, query_text as sql_query_text

load_dotenv()

# --- 1. 资源缓存池 ---


@st.cache_resource
def get_fast_llm():
    """使用 gpt-4o-mini 处理评分任务，速度极快"""
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


@st.cache_resource
def get_vectorstore():
    """缓存向量库连接"""
    embeddings = OpenAIEmbeddings()
    return Chroma(persist_directory="data/chroma_db", embedding_function=embeddings)

# --- 2. 定义结构化输出 Schema (使用 JSON Schema 避免 Pydantic 冲突) ---


route_schema = {
    "name": "route_query",
    "description": "选择最合适的数据源",
    "parameters": {
        "type": "object",
        "properties": {
            "datasource": {"type": "string", "enum": ["sql_db", "vector_db"]}
        },
        "required": ["datasource"]
    }
}

grade_schema = {
    "name": "grade_member",
    "description": "评分资料相关性",
    "parameters": {
        "type": "object",
        "properties": {
            "score": {"type": "string", "enum": ["yes", "no"]}
        },
        "required": ["score"]
    }
}

hallucination_schema = {
    "name": "grade_hallucination",
    "description": "事实核查分数",
    "parameters": {
        "type": "object",
        "properties": {
            "binary_score": {"type": "string", "enum": ["yes", "no"]}
        },
        "required": ["binary_score"]
    }
}

answer_schema = {
    "name": "grade_answer",
    "description": "有用性分数",
    "parameters": {
        "type": "object",
        "properties": {
            "binary_score": {"type": "string", "enum": ["yes", "no"]}
        },
        "required": ["binary_score"]
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

# --- 4. 节点逻辑 (Node Logic) ---


def route_question(state):
    print("--- 智能路由 ---")
    llm = get_fast_llm()
    structured_llm = llm.with_structured_output(route_schema)
    system = """你是一个专业的路由分发器。
    - 如果问题涉及【具体品牌】（如李锦记、海天）或【特定产品名称】，路由到 'sql_db'。因为结构化数据库查询更精准。
    - 如果问题是【一般性知识】（如大豆过敏能吃什么）或【生僻/描述性产品】，路由到 'vector_db'。
    """
    res = (ChatPromptTemplate.from_messages(
        [("system", system), ("human", "{question}")]) | structured_llm).invoke({"question": state["question"]})
    decision = res["datasource"] if isinstance(res, dict) else res.datasource
    return {"router_decision": decision}


def call_sql_agent(state):
    print("--- 启动 SQL Agent ---")
    response, _ = sql_query_text(state["question"])
    return {"generation": response}


def retrieve(state):
    print("--- 检索本地向量库 ---")
    docs = get_vectorstore().similarity_search(state["question"], k=3)
    return {"documents": [d.page_content for d in docs], "question": state["question"]}


def grade_documents(state):
    print("--- 文档相关性评估 ---")
    if not state.get("documents"):
        return {"web_search": "Yes"}
    llm = get_fast_llm()
    structured_llm = llm.with_structured_output(grade_schema)
    system = "判断资料是否足以回答问题。"
    res = (ChatPromptTemplate.from_messages([("system", system), ("human", "问题: {question} \n资料: {documents}")]) | structured_llm).invoke(
        {"question": state["question"], "documents": ' '.join(state["documents"])})
    score = res["score"] if isinstance(res, dict) else res.score
    return {"web_search": "No" if score == "yes" else "Yes"}


def generate(state):
    print("--- 生成回答 [Smart Mode] ---")
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        "你是一个专业的食品过敏专家。基于资料回答：\n资料: {documents}\n问题: {question}")
    response = (prompt | llm).invoke(
        {"documents": state["documents"], "question": state["question"]})
    return {"generation": response.content}


def web_search(state):
    print("--- 联网搜索 ---")
    from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
    search = TavilySearchResults(api_wrapper=TavilySearchAPIWrapper(
        tavily_api_key=os.getenv("TAVILY_API_KEY")), k=3)
    results = search.invoke({"query": state["question"]})
    return {"documents": [str(r) for r in results], "question": state["question"]}


def hallucination_grader(state):
    print("--- 进行事实核查 ---")
    llm = get_fast_llm()
    structured_llm = llm.with_structured_output(hallucination_schema)
    system = "判断回答是否完全基于参考资料。"
    res = (ChatPromptTemplate.from_messages([("system", system), ("human", "资料: {documents} \n回答: {generation}")]) | structured_llm).invoke(
        {"documents": ' '.join(state["documents"]), "generation": state["generation"]})
    score = res["binary_score"] if isinstance(res, dict) else res.binary_score
    return {"hallucination_score": score}


def answer_grader(state):
    print("--- 评估回答有用性 ---")
    llm = get_fast_llm()
    structured_llm = llm.with_structured_output(answer_schema)
    system = "判断回答是否解决了用户问题。"
    res = (ChatPromptTemplate.from_messages([("system", system), ("human", "问题: {question} \n回答: {generation}")]) | structured_llm).invoke(
        {"question": state["question"], "generation": state["generation"]})
    score = res["binary_score"] if isinstance(res, dict) else res.binary_score
    return {"answer_score": score}

# --- 5. 工作流编排 ---


def decide_to_generate(
    state): return "web_search" if state["web_search"] == "Yes" else "generate"


def grade_gen(state):
    if state["hallucination_score"] == "no":
        return "not supported"
    return "useful" if state["answer_score"] == "yes" else "not useful"


workflow = StateGraph(GraphState)
workflow.add_node("route_question", route_question)
workflow.add_node("sql_agent", call_sql_agent)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("web_search", web_search)
workflow.add_node("hallucination_grader", hallucination_grader)
workflow.add_node("answer_grader", answer_grader)

workflow.add_edge(START, "route_question")
workflow.add_conditional_edges("route_question", lambda x: x["router_decision"], {
                               "sql_db": "sql_agent", "vector_db": "retrieve"})
workflow.add_edge("sql_agent", END)
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", decide_to_generate, {
                               "web_search": "web_search", "generate": "generate"})
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", "hallucination_grader")
workflow.add_edge("hallucination_grader", "answer_grader")
workflow.add_conditional_edges("answer_grader", grade_gen, {
                               "useful": END, "not useful": "web_search", "not supported": "generate"})

app = workflow.compile()

# --- 6. 查询入口 ---


def query_with_graph(question: str, image_bytes: bytes = None):
    start_time = time.time()
    if image_bytes:
        from agent_logic import query_text as vision_query
        res, dur = vision_query(question, image_bytes=image_bytes)
        yield {"node": "end", "generation": res, "duration": dur}
        return

    final_res = "抱歉，由于逻辑判定异常，未能生成有效回答。"
    for event in app.stream({"question": question}, stream_mode="updates"):
        for node_name, output in event.items():
            yield {"node": node_name, "status": "running"}
            if "generation" in output:
                final_res = output["generation"]

    yield {"node": "end", "generation": final_res, "duration": time.time() - start_time}
