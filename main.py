from graph_logic import query_with_graph
import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

load_dotenv()

st.set_page_config(page_title="Food Allergy AI Agent", layout="wide")

# 1. 初始化记忆和处理状态
msgs = StreamlitChatMessageHistory(key="messages")
if "last_processed_file" not in st.session_state:
    st.session_state.last_processed_file = None

st.title("🥗 Food Allergy AI Agent")
st.markdown("上传食品图片或直接提问，我会帮你检查过敏原。")

# 2. 侧边栏：上传功能
with st.sidebar:
    st.header("图片识别")
    uploaded_file = st.file_uploader(
        "上传食品包装或配料表图片",
        type=["jpg", "jpeg", "png"],
        key="sidebar_uploader"
    )
    if uploaded_file:
        st.image(uploaded_file, caption="待处理图片", use_container_width=True)

        if uploaded_file.name != st.session_state.last_processed_file:
            if st.button("开始识别过敏原"):
                with st.chat_message("assistant"):
                    with st.spinner("视觉识别中..."):
                        try:
                            image_bytes = uploaded_file.getvalue()
                            # 处理流式生成器
                            for step in query_with_graph("请识别这张图片中的食品名称，并根据数据库查询其过敏原信息。", image_bytes=image_bytes):
                                if step["node"] == "end":
                                    response = step["generation"]

                            msgs.add_user_message("📸 [用户上传了图片]")
                            msgs.add_ai_message(response)
                            st.session_state.last_processed_file = uploaded_file.name
                            st.rerun()
                        except Exception as e:
                            st.error(f"识别失败: {str(e)}")

# 3. 主界面渲染历史记录
for msg in msgs.messages:
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# 4. 底部文字问答入口 (带思考过程展示)
if prompt := st.chat_input("例如：李锦记有哪些不含大豆的酱？"):
    st.chat_message("user").markdown(prompt)
    msgs.add_user_message(prompt)

    with st.chat_message("assistant"):
        # 使用 st.status 显示 DeepSeek 风格的思考过程
        with st.status("🔍 正在思考...", expanded=True) as status:
            final_response = ""
            for step in query_with_graph(prompt):
                node = step.get("node")

                # 动态显示当前执行的节点
                if node == "route_question":
                    st.write("🚦 正在分析您的问题意图...")
                elif node == "retrieve":
                    st.write("📚 正在检索本地向量数据库...")
                elif node == "sql_agent":
                    st.write("📊 正在构造 SQL 并在食品数据库中搜索...")
                elif node == "grade_documents":
                    st.write("⚖️ 正在评估找到的资料是否相关...")
                elif node == "web_search":
                    st.write("🌐 本地资料不足，正在启动联网搜索...")
                elif node == "hallucination_grader":
                    st.write("🕵️ 正在进行事实核查，确保回答无误...")
                elif node == "answer_grader":
                    st.write("✅ 正在确认回答是否解决了您的问题...")

                if node == "end":
                    final_response = step["generation"]
                    duration = step["duration"]
                    status.update(
                        label=f"✅ 思考完成 (耗时 {duration:.2f}秒)", state="complete", expanded=False)

        # 思考完成后，显示最终回答
        st.markdown(final_response)
        msgs.add_ai_message(final_response)
