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
        # 如果还是不行，直接从具体路径导入
        from langchain_classic.memory.buffer import ConversationBufferMemory

from langchain_community.chat_message_histories import StreamlitChatMessageHistory

load_dotenv()


@st.cache_resource
def get_db():
    """缓存数据库连接和表结构信息，大幅提升初始化速度"""
    return SQLDatabase.from_uri("sqlite:///data/food_data.db")


@st.cache_resource
def get_llm():
    """缓存 LLM 实例"""
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )


def get_sql_agent():
    db = get_db()
    llm = get_llm()

    # 1. 终极版系统提示词：整合翻译、图片策略、上下文记忆与分类定义优化
    system_prefix = """你是一个极其聪明的食品过敏专家。
    【数据处理原则（重要）】
    1. 质量优先：数据库中存在重复的产品记录。请优先选择【有图片链接 (image_url IS NOT NULL)】且【过敏原信息较全】的记录。
    2. 排序技巧：在 SQL 中可以使用 `ORDER BY (image_url IS NOT NULL) DESC` 来让带图片的靠前。
    3. 分类定义优化（重要）：
       - 当用户提到“零食”时，请优先寻找 `categories` 中包含 'Snacks', 'Chips', 'Confectionery', 'Biscuits', 'Candy' 的产品。
       - 除非用户明确提到，否则不要将“方便面 (Instant Noodles)”、“挂面 (Pasta/Noodles)”或“米饭 (Rice)”作为“零食”返回。
    
    【核心交互逻辑】
    1. 语言翻译（硬性要求）：
       - 数据库中的 ingredients (配料) 和 name 可能包含德语、法语等外语。
       - 无论数据库原文是什么，你必须将其翻译成用户提问所使用的语言（通常是中文）进行回复。
    
    2. 图片显示策略（智能判断）：
       - 默认情况下不要显示图片。
       - 只有当用户明确表达想看图片（如问“长什么样”、“给张图”、“看照片”、“展示产品”）时，才使用 Markdown 语法 ![产品名](url) 直接显示图片。
    
    3. 上下文记忆：
       - 你必须查阅 'chat_history'。如果用户提到“这个”、“它”，指的就是上文刚讨论过的产品。
       - 严禁反问！如果历史里有产品名，就直接基于该产品进行过敏原查询。
    
    4. 回答风格：
       - 结论先行：直接告诉用户建议。
       - 简洁明了：翻译后的配料表只列出关键成分。
    
    【查询约束】
    - 执行 SQL 必须带上 LIMIT 5。
    - 品牌匹配：'李锦记' = 'Lee Kum Kee'。
    - 隐藏技术细节：不要展示 Barcode、数据库 ID。
    """
    # 2. 创建 Agent，不再在这里传 memory，改为显式占位符
    agent_executor = create_sql_agent(
        llm,
        db=db,
        agent_type="openai-tools",
        verbose=True,
        prefix=system_prefix,
        extra_prompt_messages=[MessagesPlaceholder(
            variable_name="chat_history")],
    )

    return agent_executor


def query_text(question: str, image_bytes: bytes = None):
    start_time = time.time()

    # 3. 手动获取历史记录并传给 Agent
    msgs = StreamlitChatMessageHistory(key="messages")
    chat_history = msgs.messages[-10:] if len(msgs.messages) > 0 else []

    agent = get_sql_agent()

    if image_bytes:
        # 处理图片输入流
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        # 构建多模态消息
        input_content = [
            {"type": "text", "text": question},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            },
        ]

        # 对于图片识别，我们直接调用 LLM 的多模态能力，或者让 Agent 处理
        # 这里为了保持 Agent 的 SQL 能力，我们将识别出的信息作为 context 传给它
        llm_vision = get_llm()
        vision_msg = HumanMessage(content=input_content)
        vision_response = llm_vision.invoke([vision_msg])

        # 拿到视觉识别结果（例如产品名）后，再让 Agent 去查库
        refined_question = f"基于图片识别结果：'{vision_response.content}'，请在数据库中查询该产品的详细过敏原信息。"
        response = agent.invoke({
            "input": refined_question,
            "chat_history": chat_history
        })
    else:
        # 纯文本查询
        response = agent.invoke({
            "input": question,
            "chat_history": chat_history
        })

    end_time = time.time()
    duration = end_time - start_time

    return response["output"], duration


if __name__ == "__main__":
    # 测试代码
    # res = query_text("李锦记有哪些不含大豆的酱？")
    # print(res)
    pass
