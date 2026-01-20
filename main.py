from graph_logic import query_with_graph
import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

load_dotenv()

st.set_page_config(page_title="Food Allergy AI Agent", layout="wide")

# 1. 初始化记忆、处理状态和语义缓存
msgs = StreamlitChatMessageHistory(key="messages")
if "last_processed_file" not in st.session_state:
    st.session_state.last_processed_file = None
if "response_cache" not in st.session_state:
    st.session_state.response_cache = {}

st.title("🥗 Food Allergy AI Agent")
st.markdown("上传食品图片或直接提问，我会帮你检查过敏原。")

with st.sidebar:
    st.header("⚙️ 设置")
    language = st.selectbox(
        "选择回复语言 / Language",
        ["自动识别 (Auto)", "简体中文", "English", "Français"],
        index=0
    )
    st.session_state.target_language = language

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
                            response = ""
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

# 4. 底部文字问答入口 (带思考过程展示与缓存)
if prompt := st.chat_input("例如：李锦记有哪些不含大豆的酱？"):
    st.chat_message("user").markdown(prompt)
    msgs.add_user_message(prompt)

    with st.chat_message("assistant"):
        with st.status("🔍 正在思考...", expanded=True) as status:
            final_response = ""
            query_gen = query_with_graph(prompt)

            # 第一步：显式告知用户正在查缓存（增加专业感）
            st.write("📂 正在进行语义缓存比对...")

            try:
                # 运行生成器
                for step in query_gen:
                    node = step.get("node")

                    if node == "contextualize_question":
                        refined_q = step.get("refined_q", prompt)
                        st.write(f"🚦 识别到您的意图为: **{refined_q}**")

                    # 【核心修复】：统一使用 response_cache 这个 Key
                    elif node == "cache_hit":
                        st.success("✨ **[语义级命中]** 发现历史提议意图，正在闪现答案...")

                    elif node == "route_question":
                        st.write("🚦 正在分析问题分发路径...")
                    elif node == "retrieve":
                        st.write("📚 正在检索本地向量数据库...")
                    elif node == "sql_agent":
                        st.write("📊 正在执行 SQL 精准数据库查询...")
                    elif node == "grade_documents":
                        st.write("⚖️ 正在评估资料相关性...")
                    elif node == "web_search":
                        st.write("🌐 本地资料不足，正在启动联网搜索...")
                    elif node == "hallucination_grader":
                        st.write("🕵️ 正在进行事实核查...")
                    elif node == "answer_grader":
                        st.write("✅ 正在确认回答是否解决了您的问题...")

                    if node == "end":
                        final_response = step["generation"]
                        duration = step["duration"]
                        status.update(
                            label=f"✅ 思考完成 (耗时 {duration:.2f}秒)", state="complete", expanded=False)
            except Exception as e:
                st.error(f"逻辑执行出错: {str(e)}")

        # 思考完成后，显示最终回答
        if final_response:
            st.markdown(final_response)
            msgs.add_ai_message(final_response)
