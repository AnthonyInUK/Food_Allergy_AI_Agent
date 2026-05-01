---
title: Food Allergy AI Agent
emoji: 🥗
colorFrom: red
colorTo: pink
sdk: docker
pinned: false
license: mit
---

# 🥗 Food Allergy AI Agent

A sophisticated AI-powered assistant built with **LangGraph** and **RAG architecture**. This application helps users identify food allergens through natural language conversation or packaging photo uploads, combining structured databases (29k+ products) with real-time web search for reliable safety guidance.

## Key Features

*   **Intelligent Query Routing**: Automatically determines the optimal search path—querying structured SQL databases for brand statistics or vector databases for specific ingredient analysis.
*   **Multimodal Recognition (Vision)**: Seamlessly identifies products and extracts allergen data from uploaded food packaging or ingredient list photos using advanced vision reasoning.
*   **Self-Correction Retrieval (Self-RAG)**:
    *   **Local Knowledge**: Prioritizes verified data from `ChromaDB` and `SQLite`.
    *   **Autonomous Web Search**: Dynamically triggers **Tavily AI** to bridge knowledge gaps when local data is insufficient or outdated.
*   **Dual-Stage Quality Audit**:
    *   **Fact-Check Logic**: Validates generated responses against source documents to eliminate hallucination.
    *   **Utility Assessment**: Ensures every answer directly addresses the user's specific safety concerns.
*   **Real-time Process Transparency**: Displays the step-by-step reasoning chain (routing, retrieval, auditing) for a professional and trustworthy user experience.
*   **Automated Localization**: Instantly translates technical ingredient data from multiple languages (e.g., German, French) into user-preferred language.

## Tech Stack

*   **Orchestration**: LangChain, LangGraph
*   **Inference Engine**: GPT-4o (Reasoning & Vision)
*   **Databases**: PostgreSQL (Structured), ChromaDB (Vector)
*   **API**: FastAPI (`api_server.py`)；**Web UI**: Next.js (`frontend/`)
*   **Connectivity**: Tavily Search API
*   **Deployment**: Docker, Hugging Face Spaces

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/AnthonyInUK/Food_Allergy_AI_Agent.git
cd Food_Allergy_AI_Agent
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
Create a `.env` file in the root directory:
```text
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
# Optional: store important QA snapshots in Redis (TTL default: 3 days)
REDIS_URL=redis://localhost:6379/0
IMPORTANT_DATA_TTL_SECONDS=259200
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/food_ai

# Unified runtime toggles
ENABLE_RESPONSE_CACHE=true
ENABLE_REDIS_RESPONSE_CACHE=true
ENABLE_IMPORTANT_QA_REDIS=true
ENABLE_PG_CONVERSATION_CHECKPOINT=true
ENABLE_NODE_RETRY=true
NODE_RETRY_MAX_ATTEMPTS=3
NODE_RETRY_INITIAL_INTERVAL=0.5
NODE_RETRY_BACKOFF_FACTOR=2.0
NODE_RETRY_MAX_INTERVAL=8.0

# Retrieval/indexing knobs (for large dataset chunking strategy)
VECTOR_TOP_K=5
VECTOR_CHUNK_SIZE=800
VECTOR_CHUNK_OVERLAP=120

# Optional: CLIP-style image retrieval (Chroma collection `product_images`, default off)
# After enabling, run: python scripts/index_product_images.py --limit 500
ENABLE_IMAGE_VECTOR_RETRIEVAL=false
CHROMA_IMAGE_COLLECTION=product_images
CLIP_MODEL_NAME=clip-ViT-B-32
IMAGE_VECTOR_TOP_K=5
```

### 3.1 Migrate Existing SQLite Data (Optional)
If you already have data in `data/food_data.db`, migrate it to PostgreSQL:
```bash
python scripts/migrate_sqlite_to_postgres.py
```

### 3.2 Index product images for “search by photo” (Optional)
Requires `sentence-transformers`, PostgreSQL with `products.image_url`, and `ENABLE_IMAGE_VECTOR_RETRIEVAL=true` at runtime.
```bash
pip install -r requirements.txt
python scripts/index_product_images.py --limit 500
```

### 4. Launch backend API
```bash
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```
### 4.1 Launch Next.js frontend (optional)
```bash
cd frontend && npm install && npm run dev
```
Set `NEXT_PUBLIC_API_URL` (see `frontend/.env.local.example`) to point at the API.

## Docker Deployment

```bash
docker build -t food-agent .
docker run -p 7860:7860 -e OPENAI_API_KEY="..." -e TAVILY_API_KEY="..." food-agent
```

## Unified Operational Switches

- Runtime toggles are centralized in `app_config.py`.
- Query current effective toggles via `GET /api/runtime-config`.
- You can now uniformly enable/disable:
  - Redis response cache
  - Redis important QA snapshot storage
  - PostgreSQL conversation checkpoint fallback
  - LangGraph retry policy
- For large datasets, tune `VECTOR_CHUNK_SIZE` / `VECTOR_CHUNK_OVERLAP` / `VECTOR_TOP_K` as your indexing/query control knobs.

## Project Structure

- `api_server.py`: FastAPI REST/WebSocket 后端。
- `frontend/`: Next.js 聊天界面。
- `graph_logic.py`: LangGraph workflow definition (Routing, RAG, Quality Control).
- `agent_logic.py`: SQL Agent implementation, Vision processing, and LLM configuration.
- `data/`: SQLite databases and vector indexing files.

---
*Disclaimer: This tool is for informational purposes only. Individuals with severe allergies must always manually verify the physical ingredient labels on actual products.*
