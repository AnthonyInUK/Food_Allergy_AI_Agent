# 🥗 Food Allergy AI Agent

An intelligent multimodal assistant powered by GPT-4o-vision and RAG architecture to help users identify allergens in food products.

## ✨ Features

- 🔍 **Smart Query Routing**: Automatically dispatches between SQL database (29k+ products) and vector knowledge base
- 📸 **Vision Recognition**: Upload food packaging photos for instant allergen analysis
- 🌍 **Multi-language**: Supports Chinese, English, and French
- ⚡ **Ultra-fast**: Semantic cache with 1000x speedup for common queries
- 🎯 **Quality Assurance**: 4-stage validation pipeline with hallucination detection
- 🔄 **Self-correction**: Automatic web search fallback when local data insufficient

## 🚀 Tech Stack

- **Framework**: LangGraph, LangChain
- **LLM**: GPT-4o, GPT-4o-vision
- **Databases**: SQLite (29k+ records), ChromaDB (vector store)
- **Interface**: Streamlit
- **Search**: Tavily AI

## 🔑 Setup (HuggingFace Spaces)

### Required Secrets

Go to Settings → Secrets and add:

```
OPENAI_API_KEY = "your-openai-api-key"
TAVILY_API_KEY = "your-tavily-api-key"
```

### Files Structure

```
.
├── main.py              # Streamlit UI
├── graph_logic.py       # LangGraph workflow
├── agent_logic.py       # SQL Agent & Vision
├── requirements.txt     # Dependencies
├── data/
│   ├── food_data.db    # SQLite database
│   └── chroma_db/      # ChromaDB vector store
└── .streamlit/
    └── config.toml     # Streamlit configuration
```

## 📊 Performance

- **Simple queries** (allergen check, product image): ~10ms
- **Complex queries** (comparison, analysis): ~7-8s
- **Cached queries**: ~50ms
- **Cache hit rate**: 60-80%

## ⚠️ Disclaimer

This tool is for informational purposes only. Always verify ingredient labels for severe allergies.

## 📧 Contact

Created by Anthony | [HuggingFace](https://hf.co/AnthonyInBC)

