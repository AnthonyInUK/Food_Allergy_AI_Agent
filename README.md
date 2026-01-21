---
title: Food Allergy AI Agent
emoji: 🥗
colorFrom: red
colorTo: pink
sdk: streamlit
sdk_version: "1.32.0"
app_file: main.py
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
*   **Databases**: SQLite (Structured), ChromaDB (Vector)
*   **Interface**: Streamlit
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
```

### 4. Launch Application
```bash
streamlit run main.py
```

## Docker Deployment

```bash
docker build -t food-agent .
docker run -p 7860:7860 -e OPENAI_API_KEY="..." -e TAVILY_API_KEY="..." food-agent
```

## Project Structure

- `main.py`: Streamlit UI and interaction layer.
- `graph_logic.py`: LangGraph workflow definition (Routing, RAG, Quality Control).
- `agent_logic.py`: SQL Agent implementation, Vision processing, and LLM configuration.
- `data/`: SQLite databases and vector indexing files.

---
*Disclaimer: This tool is for informational purposes only. Individuals with severe allergies must always manually verify the physical ingredient labels on actual products.*
