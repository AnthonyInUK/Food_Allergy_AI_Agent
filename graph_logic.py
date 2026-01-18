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
# 修复 Pydantic 导入：使用原生 pydantic 或 langchain_core 兼容路径
try:
    from pydantic.v1 import BaseModel, Field
except ImportError:
    from pydantic import BaseModel, Field
from langgraph.graph import END, StateGraph, START

from agent_logic import get_sql_agent, get_db, get_llm, query_text as sql_query_text

load_dotenv()

# --- 1. 定义状态 ---


class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[str]
    router_decision: str  # 'sql_db', 'vector_db'
    hallucination_score: str  # 'yes', 'no'
    answer_score: str  # 'yes', 'no'

# --- 2. 节点逻辑：路由 (Router) ---


class RouteQuery(BaseModel):
    """根据用户问题路由到最合适的查询路径。"""
    datasource: str = Field(
        description="选择 'sql_db' (针对品牌统计、列表、SQL查询) 或 'vector_db' (针对具体过敏原分析、产品识别)。"
    )


def route_question(state):
    print("--- 正在智能路由 ---")
    question = state["question"]
    llm = get_llm()
    structured_llm_router = llm.with_structured_output(RouteQuery)

    system = """你是一个专业的路由分发器。
    - 用户问题涉及品牌列表、数量统计、模糊搜索（如：李锦记有哪些酱？不含大豆的酱有哪些？），路由到 'sql_db'。
    - 用户问题涉及具体产品成分、过敏建议、或者一般性食品安全问题，路由到 'vector_db'。
    """
    route_prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", "{question}")])
    router_agent = route_prompt | structured_llm_router
    decision = router_agent.invoke({"question": question})

    return {"router_decision": decision.datasource}

# --- 3. 节点逻辑：SQL Agent ---


def call_sql_agent(state):
    print("--- 启动 SQL Agent 查询 ---")
    question = state["question"]
    response, _ = sql_query_text(question)
    return {"generation": response}

# --- 4. 节点逻辑：RAG 检索 (Retrieve) ---


def retrieve(state):
    print("--- 检索本地向量库 ---")
    question = state["question"]
    embeddings = OpenAIEmbeddings()
    try:
        vectorstore = Chroma(
            persist_directory="data/chroma_db", embedding_function=embeddings)
        docs = vectorstore.similarity_search(question, k=3)
        doc_texts = [d.page_content for d in docs]
    except Exception as e:
        print(f"向量库加载失败: {e}")
        doc_texts = []

    return {"documents": doc_texts, "question": question}

# --- 5. 节点逻辑：评估文档 (Grade Documents) ---


class GradeMember(BaseModel):
    """评分：判断检索到的文档是否与问题相关。"""
    score: str = Field(description="相关性分值：'yes' 或 'no'")


def grade_documents(state):
    print("--- 评估文档质量 ---")
    question = state["question"]
    documents = state["documents"]
    if not documents:
        return {"web_search": "Yes", "documents": documents}

    llm = get_llm()
    structured_llm_grader = llm.with_structured_output(GradeMember)
    system = "你是一个质量评估员。判断提供的参考资料是否足以回答用户的问题。若是则回 'yes'，否则回 'no'。"
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "问题: {question} \n\n 资料: {documents}")
    ])
    grader_chain = grade_prompt | structured_llm_grader
    result = grader_chain.invoke(
        {"question": question, "documents": ' '.join(documents)})

    return {"web_search": "No" if result.score == "yes" else "Yes", "documents": documents}

# --- 6. 节点逻辑：生成回答 (Generate) ---


def generate(state):
    print("--- 生成初步答复 ---")
    question = state["question"]
    documents = state["documents"]
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template("""你是一个专业的食品过敏专家。请基于资料回答问题。
    如果资料中没有相关信息，请明确说明。
    资料: {documents}
    问题: {question}
    回答:""")
    chain = prompt | llm
    response = chain.invoke({"documents": documents, "question": question})
    return {"generation": response.content}

# --- 7. 节点逻辑：联网搜索 (Web Search) ---


def web_search(state):
    print("--- 触发联网搜索 ---")
    question = state["question"]
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {"documents": ["未配置联网搜索 Key"], "question": question}

    try:
        from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
        search_wrapper = TavilySearchAPIWrapper(tavily_api_key=api_key)
        search = TavilySearchResults(api_wrapper=search_wrapper, k=3)
        results = search.invoke({"query": question})
        web_docs = [str(r) for r in results]
    except Exception as e:
        web_docs = [f"搜索失败: {str(e)}"]
    return {"documents": web_docs, "question": question}

# --- 8. 新增节点：幻觉检查 (Hallucination Grader) ---


class GradeHallucination(BaseModel):
    binary_score: str = Field(description="回答是否基于文档：'yes' 或 'no'")


def hallucination_grader(state):
    print("--- 进行幻觉检查 ---")
    generation = state["generation"]
    documents = state["documents"]
    llm = get_llm()
    structured_llm_grader = llm.with_structured_output(GradeHallucination)

    system = "你是一个事实核查员。判断 AI 生成的回答是否完全基于提供的参考资料。若无幻觉回 'yes'，有幻觉回 'no'。"
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "资料: {documents} \n\n 回答: {generation}")
    ])
    grader_chain = grade_prompt | structured_llm_grader
    result = grader_chain.invoke(
        {"documents": ' '.join(documents), "generation": generation})
    return {"hallucination_score": result.binary_score}

# --- 9. 新增节点：有用性检查 (Answer Grader) ---


class GradeAnswer(BaseModel):
    binary_score: str = Field(description="回答是否有用：'yes' 或 'no'")


def answer_grader(state):
    print("--- 评估回答有用性 ---")
    generation = state["generation"]
    question = state["question"]
    llm = get_llm()
    structured_llm_grader = llm.with_structured_output(GradeAnswer)

    system = "你是一个问答评估员。判断 AI 生成的回答是否真正解决了用户的问题。有用回 'yes'，无用回 'no'。"
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "问题: {question} \n\n 回答: {generation}")
    ])
    grader_chain = grade_prompt | structured_llm_grader
    result = grader_chain.invoke(
        {"question": question, "generation": generation})
    return {"answer_score": result.binary_score}

# --- 10. 条件路由函数 ---


def decide_to_generate(state):
    if state["web_search"] == "Yes":
        return "web_search"
    else:
        return "generate"


def grade_generation_v_documents_and_question(state):
    """根据结果决定下一步"""
    if state["hallucination_score"] == "no":
        return "not supported"
    return "useful" if state["answer_score"] == "yes" else "not useful"


# --- 11. 构建工作流图 ---
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
workflow.add_conditional_edges("answer_grader", grade_generation_v_documents_and_question, {
                               "useful": END, "not useful": "web_search", "not supported": "generate"})

app = workflow.compile()

# --- 12. 支持流式思考的查询函数 ---


def query_with_graph(question: str, image_bytes: bytes = None):
    start_time = time.time()

    # 针对图片识别，走 Vision 逻辑 (这里暂不通过 Graph 流显示，保持原本逻辑)
    if image_bytes:
        from agent_logic import query_text as vision_query
        res, dur = vision_query(question, image_bytes=image_bytes)
        yield {"node": "end", "generation": res, "duration": dur}
        return

    # 针对纯文本，输出思考流
    final_generation = "抱歉，我无法生成回答。"
    for event in app.stream({"question": question}, stream_mode="updates"):
        for node_name, output in event.items():
            # 将当前运行的节点 yield 给前端
            yield {"node": node_name, "status": "running"}
            if "generation" in output:
                final_generation = output["generation"]

    end_time = time.time()
    yield {"node": "end", "generation": final_generation, "duration": end_time - start_time}
