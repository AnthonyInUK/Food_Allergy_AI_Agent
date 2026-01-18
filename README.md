# 🥗 Food Allergy AI Agent

这是一个基于 **LangGraph** 和 **Self-RAG** 架构的智能食品过敏专家 Agent。它能够通过文本对话或图片上传，帮助用户快速查询食品成分及过敏原，并在本地数据库信息不足时自动启动联网搜索。

## 🌟 核心功能

*   **智能路径路由 (Smart Routing)**：自动识别用户意图，决定是查询结构化 SQL 数据库（统计/列表类）还是向量数据库（成分/常识类）。
*   **多模态识别 (Vision)**：支持上传食品包装或配料表图片，利用 GPT-4o-vision 自动识别产品并分析过敏风险。
*   **自我修正检索 (Self-RAG)**：
    *   **本地检索**：优先从 `ChromaDB` 和 `SQLite` 中获取数据。
    *   **联网搜索**：当本地资料不足或不相关时，自动触发 **Tavily AI** 联网搜索。
*   **双重质量审计**：
    *   **幻觉检查 (Hallucination Grader)**：确保生成的回答完全基于事实，拒绝胡编乱造。
    *   **有用性评估 (Answer Grader)**：确保回答直接解决了用户的问题。
*   **DeepSeek 风格思考流**：实时展示 Agent 的思考过程（路由、检索、核查等节点状态）。
*   **多语言翻译**：自动将数据库中的德语、法语等配料信息翻译为中文。

## 🛠️ 技术栈

*   **框架**：LangChain, LangGraph
*   **大模型**：GPT-4o (Reasoning & Vision)
*   **数据库**：SQLite (结构化数据), ChromaDB (向量检索)
*   **前端**：Streamlit
*   **搜索**：Tavily Search API
*   **部署**：Docker, Hugging Face Spaces

## 🚀 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/AnthonyInUK/Food_Allergy_AI_Agent.git
cd Food_Allergy_AI_Agent
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 配置环境变量
在根目录下创建 `.env` 文件：
```text
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
```

### 4. 运行应用
```bash
streamlit run main.py
```

## 🐳 Docker 部署

```bash
docker build -t food-ai-agent .
docker run -p 7860:7860 -e OPENAI_API_KEY="..." -e TAVILY_API_KEY="..." food-ai-agent
```

## 📈 项目结构

- `main.py`: Streamlit UI 界面与交互逻辑。
- `graph_logic.py`: LangGraph 工作流定义（路由、RAG、质量检查）。
- `agent_logic.py`: SQL Agent 逻辑、Vision 识别及 LLM 配置。
- `data/`: 存储 SQLite 数据库及向量索引文件。

---
*声明：本工具仅供参考，过敏患者在食用前请务必仔细核对食品实物包装上的成分表。*

