"""
Food Allergy AI Agent - 命令行版本
主要UI已迁移到Next.js前端 (frontend/)
此脚本用于CLI测试和后端API测试
"""

import sys
import argparse
from dotenv import load_dotenv

load_dotenv()

from langsmith_config import configure_langsmith_tracing_compat

configure_langsmith_tracing_compat()

from graph_logic import query_with_graph, get_cache_stats, clear_all_caches

import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from convo_store import (
    create_conversation,
    delete_conversation,
    find_conversation_by_id,
    load_conversations,
    save_conversation,
)

# Modern 3D/Glassmorphism UI Styling
st.markdown("""
<style>
    /* Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* 3D Glassmorphism Cards for messages */
    div.stChatMessage {
        background: rgba(255, 255, 255, 0.7) !important;
        backdrop-filter: blur(10px);
        border-radius: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        margin-bottom: 1rem !important;
        transition: transform 0.2s ease-in-out;
    }
    
    div.stChatMessage:hover {
        transform: translateY(-5px);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.8);
        border-right: 1px solid rgba(0,0,0,0.05);
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 20px;
        transition: all 0.3s;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }

    /* --- Chat Input Box Styling --- */
    div[data-testid="stChatInput"] {
        border-radius: 30px !important;
        background-color: white !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
        padding: 8px 15px 8px 50px !important;
        max-width: 800px !important;
        margin: 0 auto !important;
        border: 1px solid #e0e0e0 !important;
        position: relative !important; /* Make this the absolute positioning anchor */
    }
    
    div[data-testid="stChatInput"] > div {
        border: none !important;
        background-color: transparent !important;
    }

    /* Plus Button - Absolutely positioned INSIDE the input box */
    .plus-button-container {
        position: absolute;
        top: 50%;
        left: 12px;
        transform: translateY(-50%);
        z-index: 1000;
        background: #3d5afe;
        color: white;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        pointer-events: none;
        font-size: 22px;
        font-weight: bold;
    }

    /* Invisible File Uploader - Same position as the button */
    .stFileUploader {
        position: absolute;
        top: 50%;
        left: 12px;
        transform: translateY(-50%);
        width: 32px !important;
        height: 32px !important;
        z-index: 1001;
        opacity: 0 !important;
        cursor: pointer;
    }
    
    .stFileUploader > div {
        width: 32px !important;
        height: 32px !important;
    }
    
    .stFileUploader > label {
        width: 32px !important;
        height: 32px !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Mobile responsive override */
    @media (max-width: 800px) {
        .plus-button-container, .stFileUploader {
            left: 10px !important;
            transform: translateY(-50%) !important;
        }
        div[data-testid="stChatInput"] {
            padding-left: 50px !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# Note: we intentionally avoid calling Streamlit's rerun APIs to remain
# compatible across versions. Sidebar actions (select/new/delete) update
# `st.session_state` directly so the UI updates without forcing a rerun.


# 1. Initialize memory, processing state and semantic cache
msgs = StreamlitChatMessageHistory(key="messages")
if "last_processed_file" not in st.session_state:
    st.session_state.last_processed_file = None
if "response_cache" not in st.session_state:
    st.session_state.response_cache = {}

# --- Conversation persistence state ---
convos = load_conversations()
if "current_conversation_id" not in st.session_state:
    # open first existing convo or create a new one
    if convos:
        st.session_state.current_conversation_id = convos[0].get("id")
    else:
        conv = create_conversation()
        st.session_state.current_conversation_id = conv["id"]
        convos = load_conversations()


st.title("🥗 Food Allergy AI Agent")
st.markdown(
    "Upload food images or ask questions directly. I'll help you check for allergens.")

with st.sidebar:
    st.header("💬 Conversations")
    # Reload conversations each sidebar render
    convos = load_conversations()
    # Use title directly, append short ID only if titles collide
    conv_map = {}
    for c in convos:
        display_name = c.get("title", "Untitled")
        if display_name in conv_map:
            display_name = f"{display_name} ({c.get('id')[:4]})"
        conv_map[display_name] = c.get("id")

    # selection
    selected_title = None
    if convs := list(conv_map.keys()):
        selected_title = st.selectbox("Open conversation", convs, index=0)
        if selected_title:
            selected_id = conv_map[selected_title]
            if st.session_state.get("current_conversation_id") != selected_id:
                st.session_state.current_conversation_id = selected_id
                # mark that we switched conversations so the in-memory chat history
                # (`msgs`) can be refreshed to show the selected convo immediately
                st.session_state["_conversation_changed"] = True

    if st.button("+ New conversation"):
        new_conv = create_conversation()
        st.session_state.current_conversation_id = new_conv["id"]
        st.session_state["_conversation_changed"] = True

    if st.button("🗑️ Delete conversation"):
        cid = st.session_state.get("current_conversation_id")
        if cid:
            delete_conversation(cid)
            # pick first remaining or create new
            convs = load_conversations()
            if convs:
                st.session_state.current_conversation_id = convs[0].get("id")
            else:
                nc = create_conversation()
                st.session_state.current_conversation_id = nc["id"]
            st.session_state["_conversation_changed"] = True

    st.divider()

    # (Removed old sidebar blocks for image recognition to move to main area)

    # 3. Sidebar: Settings & Cache (Grouped)
    with st.expander("⚙️ Preferences & Performance", expanded=False):
        st.subheader("Language")
        language = st.selectbox(
            "Reply Language",
            ["自动识别 (Auto)", "简体中文", "English", "Français"],
            index=0,
            label_visibility="collapsed"
        )
        st.session_state.target_language = language

        st.divider()

        st.subheader("📊 Smart Cache stats")
        stats = get_cache_stats()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Hit Rate", f"{stats['hit_rate']:.1f}%")
        with col2:
            st.metric("Total", stats['total_queries'])

        # Display cache layers details if items exist
        cache_layers = 0
        cache_sizes = []
        for cache_name in ["response_cache", "retrieval_cache", "generation_cache",
                           "grade_cache", "hallucination_cache", "answer_grade_cache"]:
            if cache_name in st.session_state:
                cache_layers += 1
                cache_sizes.append(len(st.session_state[cache_name]))

        if cache_layers > 0:
            total_items = sum(cache_sizes)
            st.info(f"💾 {total_items} items in {cache_layers} layers")

        if st.button("🗑️ Reset All Caches", use_container_width=True):
            clear_all_caches()
            st.success("✅ Caches cleared!")

# 3. Main interface: Render chat history
# Prefer showing the currently opened conversation (if present); fall back to in-memory Streamlit history
current_conv = None
cid = st.session_state.get("current_conversation_id")
if cid:
    current_conv = find_conversation_by_id(cid)

# If the conversation just changed, refresh the in-memory Streamlit chat history
# so the UI shows the persisted conversation immediately without calling rerun.
if st.session_state.get("_conversation_changed") and current_conv:
    try:
        # reset the in-memory messages list used by StreamlitChatMessageHistory
        st.session_state["messages"] = []
        # repopulate
        for m in current_conv.get("messages", []):
            if m.get("role") == "user":
                msgs.add_user_message(m.get("text", ""))
            else:
                msgs.add_ai_message(m.get("text", ""))
    except Exception as _e:
        # best-effort: if the underlying structure differs, ignore and fall back
        print(f"Warning: failed to refresh in-memory messages: {_e}")
    finally:
        st.session_state["_conversation_changed"] = False

if current_conv:
    for i, m in enumerate(current_conv.get("messages", [])):
        role = m.get("role", "user")
        text = m.get("text", "")
        with st.chat_message(role):
            st.markdown(text)
            # Add Save button for assistant messages that aren't already saved
            if role == "assistant" and not m.get("saved"):
                if st.button("⭐ Save to Permanent DB", key=f"save_{i}"):
                    # Find previous user message for question info
                    q_text = "N/A"
                    if i > 0:
                        q_text = current_conv.get("messages")[
                            i-1].get("text", "")

                    if move_to_permanent_storage(q_text, text):
                        st.success("Successfully saved to permanent store!")
                        # Mark as saved in current session to hide button
                        m["saved"] = True
                    else:
                        st.error("Failed to save.")
else:
    for i, msg in enumerate(msgs.messages):
        role = "user" if msg.type == "human" else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)
            if role == "assistant":
                if st.button("⭐ Save to Permanent DB", key=f"save_mem_{i}"):
                    q_text = msgs.messages[i-1].content if i > 0 else "N/A"
                    if move_to_permanent_storage(q_text, msg.content):
                        st.success("Successfully saved to permanent store!")
                    else:
                        st.error("Failed to save.")

# 4. Bottom text input with thinking process display and cache
if prompt := st.chat_input("e.g.: Which Lee Kum Kee sauces are soy-free?"):
    st.chat_message("user").markdown(prompt)
    msgs.add_user_message(prompt)

    # Persist user message to conversation
    cid = st.session_state.get("current_conversation_id")
    if cid:
        conv = find_conversation_by_id(cid)
        if conv is None:
            # Create with a snippet of the first prompt
            title = prompt[:30] + ("..." if len(prompt) > 30 else "")
            conv = create_conversation(title=title)
            st.session_state.current_conversation_id = conv["id"]

        # If the conversation is still named "New conversation", update it to the first prompt
        if conv.get("title") == "New conversation":
            conv["title"] = prompt[:30] + ("..." if len(prompt) > 30 else "")

        conv.setdefault("messages", []).append(
            {"role": "user", "text": prompt, "ts": time.time()})
        save_conversation(conv)

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
                    elif node == "plan_route":
                        st.write("🧭 Routing (rules / optional JSON)...")
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

            # --- Redis Background Persistence ---
            try:
                from important_data_store import record_important_qa
                record_important_qa(prompt, final_response,
                                    source="chat_interaction")
            except Exception as e:
                print(f"Redis persistence failed: {e}")

            # Persist assistant response to conversation
            cid = st.session_state.get("current_conversation_id")
            if cid:
                conv = find_conversation_by_id(cid)
                if conv is None:
                    conv = create_conversation()
                    st.session_state.current_conversation_id = conv["id"]
                conv.setdefault("messages", []).append(
                    {"role": "assistant", "text": final_response, "ts": time.time()})
                save_conversation(conv)

            # Store important text QA snapshot in Redis (TTL ~3 days)
            record_important_qa(
                question=prompt, answer=final_response, source="chat")

        # Restore original language setting
        st.session_state.target_language = original_lang

# --- Unified Input Area (Gemini Style) ---

# Floating Plus Button UI Overlay
st.markdown('<div class="plus-button-container">+</div>',
            unsafe_allow_html=True)

# Invisible Uploader that covers the plus button
uploaded_file = st.file_uploader(
    "Upload",
    type=["jpg", "jpeg", "png"],
    key="unified_uploader",
    label_visibility="collapsed"
)

if uploaded_file and uploaded_file.name != st.session_state.get("last_processed_file"):
    with st.chat_message("assistant"):
        with st.status("📸 Analyzing image...", expanded=True) as status:
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

                status.update(label="✅ Image Analysis Complete",
                              state="complete", expanded=False)

                if response:
                    st.markdown(response)
                    msgs.add_user_message("📸 [User uploaded an image]")
                    msgs.add_ai_message(response)
                    st.session_state.last_processed_file = uploaded_file.name

                    # Persist to conversation store
                    cid = st.session_state.get("current_conversation_id")
                    if cid:
                        conv = find_conversation_by_id(cid)
                        if conv is None:
                            conv = create_conversation()
                            st.session_state.current_conversation_id = conv["id"]
                        conv.setdefault("messages", []).append(
                            {"role": "user", "text": "📸 [User uploaded an image]", "ts": time.time()})
                        conv.setdefault("messages", []).append(
                            {"role": "assistant", "text": response, "ts": time.time()})
                        save_conversation(conv)

                    # Store important image QA snapshot in Redis
                    record_important_qa(
                        question="[image] Product allergen analysis",
                        answer=response,
                        source="image",
                    )
            except Exception as e:
                st.error(f"Recognition failed: {str(e)}")
            finally:
                st.session_state.target_language = original_lang
