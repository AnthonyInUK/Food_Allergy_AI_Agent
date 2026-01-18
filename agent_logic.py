import os
import time
import base64
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage

# 修复导入：在当前环境下 ConversationBufferMemory 位于 langchain_classic
try:
    from langchain.memory import ConversationBufferMemory
except ImportError:
    try:
        from langchain_classic.memory import ConversationBufferMemory
    except ImportError:
        from langchain_classic.memory.buffer import ConversationBufferMemory

from langchain_community.chat_message_histories import StreamlitChatMessageHistory

load_dotenv()

# --- 1. 全局配置与行为准则 ---
SQL_SYSTEM_PREFIX = """你是一个严谨的食品过敏专家。

【跨语言搜索与补位逻辑 - 必须执行】
1. 识别产品的中英文名：同一产品往往分为中文记录（常为空）和英文记录（常有成分表）。搜中文时必须自动生成对应英文名进行联合查询。
2. 强制取齐逻辑：使用 `OR` 连接中英文名，并强制增加 `AND ingredients != ''` 过滤。示例：`WHERE (name LIKE '%精选老抽%' OR name LIKE '%Premium Dark Soy Sauce%') AND ingredients != ''`
3. 质量优先排序：必须使用 `ORDER BY length(ingredients) DESC` 让内容最详实的记录排在最前面。

【核心交互逻辑 - 结果过滤】
1. 意图精准匹配（极重要）：
   - 如果用户只问“过敏成分”或“过敏原”，你只需给出过敏原判定结论，**绝对严禁**列出冗长的配料表（ingredients）或展示图片。
   - 只有当用户明确问“有什么成分”、“配料表是什么”时，才提供详细配料表。
   - 只有当用户明确说“想看图”、“长什么样”时，才展示图片。
2. 翻译呈现：无论提取到的是英文、德语还是法语成分，必须将其【翻译成中文】展示。
3. 结论先行：直接告诉用户建议（能吃/不能吃/含有什么过敏原）。
4. 记忆指代：必须查阅 'chat_history' 解析“它”、“这个”。严禁反问！

【查询技术限制- 性能与质量优化】
1. 动态数量：针对特定产品的提问使用 LIMIT 1；针对列表类或模糊查询使用 LIMIT 5。
2. 列剪枝：仅查询满足当前意图的最小必要列。例如：若用户未要求看图，严禁查询 `image_url`；若用户只问过敏原，优先查 `allergens`，仅在前者为空时才查 `ingredients` 兜底。
3. 索引优化：对于已知确切品牌（如“李锦记”），SQL 应优先使用 `brand = 'Lee Kum Kee'` 而非 `LIKE`，以利用数据库索引。
4. 排除噪声：在查询条件中增加 `AND length(ingredients) > 5` 来过滤掉库中那些只有名字没有实际内容的垃圾记录。
5. 语言约束：始终使用中文回答。
"""


@st.cache_resource
def get_db():
    """缓存数据库连接和表结构信息"""
    return SQLDatabase.from_uri("sqlite:///data/food_data.db")


@st.cache_resource
def get_llm():
    """缓存 GPT-4o 主模型"""
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )


@st.cache_resource
def get_sql_agent():
    """缓存 Agent 实例，避免重复构建推理链"""
    db = get_db()
    llm = get_llm()

    # 注意：这里不再绑定 memory，历史记录在 invoke 时动态传入，以提升并发稳定性
    return create_sql_agent(
        llm,
        db=db,
        agent_type="openai-tools",
        verbose=True,
        prefix=SQL_SYSTEM_PREFIX,
        extra_prompt_messages=[MessagesPlaceholder(
            variable_name="chat_history")],
        max_iterations=5,
        top_k=5
    )


def query_text(question: str, image_bytes: bytes = None):
    """
    核心查询接口：支持文本和图片。
    """
    start_time = time.time()

    # 获取当前会话的聊天历史
    msgs = StreamlitChatMessageHistory(key="messages")
    chat_history = msgs.messages[-10:] if len(msgs.messages) > 0 else []

    if image_bytes:
        # --- 视觉识别逻辑 ---
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        input_content = [
            {"type": "text", "text": "请识别这张图片中的食品名称、品牌以及过敏原信息。"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            },
        ]

        llm_vision = get_llm()
        vision_msg = HumanMessage(content=input_content)
        vision_response = llm_vision.invoke([vision_msg])

        # 将视觉识别结果转化为 SQL Agent 可理解的问题
        refined_question = f"基于视觉识别结果：'{vision_response.content}'，请在数据库中查询并分析该产品的详细过敏原风险。"
        agent = get_sql_agent()
        response = agent.invoke({
            "input": refined_question,
            "chat_history": chat_history
        })
    else:
        # --- 纯文本逻辑 ---
        agent = get_sql_agent()
        response = agent.invoke({
            "input": question,
            "chat_history": chat_history
        })

    end_time = time.time()
    duration = end_time - start_time

    return response["output"], duration
