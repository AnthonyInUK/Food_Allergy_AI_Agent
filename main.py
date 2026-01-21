from graph_logic import query_with_graph, get_cache_stats, clear_all_caches
import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

load_dotenv()

st.set_page_config(page_title="Food Allergy AI Agent", layout="wide")

# 1. Initialize memory, processing state and semantic cache
msgs = StreamlitChatMessageHistory(key="messages")
if "last_processed_file" not in st.session_state:
    st.session_state.last_processed_file = None
if "response_cache" not in st.session_state:
    st.session_state.response_cache = {}

st.title("🥗 Food Allergy AI Agent")
st.markdown(
    "Upload food images or ask questions directly. I'll help you check for allergens.")

with st.sidebar:
    st.header("⚙️ Settings")
    language = st.selectbox(
        "Reply Language / 回复语言",
        ["自动识别 (Auto)", "简体中文", "English", "Français"],
        index=0
    )
    st.session_state.target_language = language

    st.divider()

    # Cache Statistics
    st.header("📊 Smart Cache System")
    st.caption(
        "Auto-cache all evaluation results to boost speed while ensuring quality")

    stats = get_cache_stats()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Hit Rate", f"{stats['hit_rate']:.1f}%")
    with col2:
        st.metric("Total Queries", stats['total_queries'])

    # Display cache layers
    cache_layers = 0
    cache_sizes = []
    for cache_name in ["response_cache", "retrieval_cache", "generation_cache",
                       "grade_cache", "hallucination_cache", "answer_grade_cache"]:
        if cache_name in st.session_state:
            cache_layers += 1
            cache_sizes.append(len(st.session_state[cache_name]))

    if cache_layers > 0:
        total_cached_items = sum(cache_sizes)
        st.info(
            f"🗄️ {cache_layers} Cache Layers | {total_cached_items} Records")

    if st.button("🗑️ Clear All Caches", use_container_width=True):
        clear_all_caches()
        st.success("✅ All caches cleared!")
        st.rerun()

# 2. Sidebar: Upload function
with st.sidebar:
    st.header("Image Recognition")
    uploaded_file = st.file_uploader(
        "Upload food packaging or ingredient label",
        type=["jpg", "jpeg", "png"],
        key="sidebar_uploader"
    )
    if uploaded_file:
        st.image(uploaded_file, caption="Image to Process",
                 use_container_width=True)

        if uploaded_file.name != st.session_state.last_processed_file:
            if st.button("Start Allergen Recognition"):
                with st.chat_message("assistant"):
                    with st.spinner("Vision recognition in progress..."):
                        try:
                            # Force English for image recognition
                            original_lang = st.session_state.get(
                                "target_language", "English")
                            st.session_state.target_language = "English"

                            image_bytes = uploaded_file.getvalue()
                            response = ""
                            # Process streaming generator
                            for step in query_with_graph("Please identify the food product name in this image and query the database for allergen information.", image_bytes=image_bytes):
                                if step["node"] == "end":
                                    response = step["generation"]

                            msgs.add_user_message("📸 [User uploaded an image]")
                            msgs.add_ai_message(response)
                            st.session_state.last_processed_file = uploaded_file.name

                            # Restore original language
                            st.session_state.target_language = original_lang
                            st.rerun()
                        except Exception as e:
                            st.error(f"Recognition failed: {str(e)}")
                            # Restore original language on error
                            st.session_state.target_language = original_lang

# 3. Main interface: Render chat history
for msg in msgs.messages:
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# 4. Bottom text input with thinking process display and cache
if prompt := st.chat_input("e.g.: Which Lee Kum Kee sauces are soy-free?"):
    st.chat_message("user").markdown(prompt)
    msgs.add_user_message(prompt)

    with st.chat_message("assistant"):
        with st.status("🔍 Thinking...", expanded=True) as status:
            # Force English for UI display
            original_lang = st.session_state.get("target_language", "English")
            st.session_state.target_language = "English"

            final_response = ""
            query_gen = query_with_graph(prompt)

            # Step 1: Notify user about semantic cache checking
            st.write("📂 Checking semantic cache...")

            try:
                # 运行生成器
                for step in query_gen:
                    node = step.get("node")

                    if node == "contextualize_question":
                        refined_q = step.get("refined_q", prompt)
                        st.write(f"🚦 Detected intent: **{refined_q}**")

                    # Semantic cache hit
                    elif node == "cache_hit":
                        st.success(
                            "✨ **[Semantic Cache Hit]** Found historical query, retrieving answer...")

                    elif node == "fast_path_detected":
                        st.success(
                            "🚀 **[Fast Path]** Simple query detected, direct SQL (1000x faster)...")

                    elif node == "complex_query_detected":
                        keyword = step.get("keyword", "complex keyword")
                        st.info(
                            f"🤖 **[Multi-hop Mode]** Detected '{keyword}', enabling Agent deep analysis...")

                    elif node == "route_question":
                        st.write("🚦 Analyzing query routing...")
                    elif node == "retrieve":
                        st.write("📚 Retrieving from local vector database...")
                    elif node == "sql_agent":
                        st.write("📊 Executing SQL precision database query...")
                    elif node == "grade_documents":
                        st.write("⚖️ Evaluating document relevance...")
                    elif node == "generate":
                        st.write("✍️ Generating response...")
                    elif node == "web_search":
                        st.write(
                            "🌐 Local data insufficient, launching web search...")
                    elif node == "parallel_graders":
                        st.write(
                            "🚀 **[Parallel Acceleration]** Fact-checking and quality assessment in parallel...")
                    elif node == "handle_off_topic":
                        st.write("🚫 Non-food related query detected...")

                    if node == "end":
                        final_response = step["generation"]
                        duration = step["duration"]
                        status.update(
                            label=f"✅ Thinking Complete (took {duration:.2f}s)", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Execution error: {str(e)}")

        # Display final response
        if final_response:
            st.markdown(final_response)
            msgs.add_ai_message(final_response)

        # Restore original language setting
        st.session_state.target_language = original_lang
