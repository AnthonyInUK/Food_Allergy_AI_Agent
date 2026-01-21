# 🚀 Deploy to HuggingFace Spaces

## 📋 Pre-deployment Checklist

- ✅ README.md has HuggingFace metadata (already done)
- ✅ requirements.txt is ready
- ✅ .streamlit/config.toml is configured
- ⚠️ Need to configure API keys as Secrets

## 🔑 Step 1: Prepare Your Repository

Your Space: https://huggingface.co/spaces/AnthonyInBC/Anthony_space

## 📤 Step 2: Upload via Git

### Option A: Using Git (Recommended)

```bash
# 1. Install git-lfs (for large files like database)
brew install git-lfs  # macOS
git lfs install

# 2. Clone your HuggingFace Space
cd ~/Desktop/llm
git clone https://huggingface.co/spaces/AnthonyInBC/Anthony_space
cd Anthony_space

# 3. Copy your latest code
cp -r ../foodAIAgent/* .

# 4. Track large files with git-lfs
git lfs track "data/*.db"
git lfs track "data/chroma_db/**"

# 5. Commit and push
git add .
git commit -m "Update: Performance optimization with 6-layer cache and fast-path routing"
git push
```

### Option B: Using HuggingFace Web UI

1. Go to https://huggingface.co/spaces/AnthonyInBC/Anthony_space
2. Click "Files and versions"
3. Click "Add file" → "Upload files"
4. Drag and drop:
   - main.py
   - graph_logic.py
   - agent_logic.py
   - requirements.txt
   - README.md
   - .streamlit/config.toml
   - data/ folder (if not too large)

## 🔐 Step 3: Configure Secrets

1. Go to your Space settings: https://huggingface.co/spaces/AnthonyInBC/Anthony_space/settings
2. Scroll to "Repository secrets"
3. Add these secrets:

```
Name: OPENAI_API_KEY
Value: sk-proj-...

Name: TAVILY_API_KEY
Value: tvly-...
```

## ✅ Step 4: Verify Deployment

1. Wait for build to complete (~2-3 minutes)
2. Visit: https://huggingface.co/spaces/AnthonyInBC/Anthony_space
3. Test queries:
   - "Lee Kum Kee dark soy sauce, can I drink it?"
   - "Show me what Lee Kum Kee products look like"

## 📊 Expected Performance

- Fast queries: ~10-50ms
- Complex queries: ~7-8s
- Image recognition: ~3-5s

## 🐛 Troubleshooting

**If app doesn't start:**
- Check logs in "Logs" tab
- Verify secrets are set correctly
- Ensure requirements.txt has all dependencies

**If database missing:**
- Upload data/food_data.db via git-lfs
- Or re-create database on HuggingFace

## 📝 Notes

- HuggingFace Spaces has 16GB RAM limit
- Your database (~50MB) is well within limits
- ChromaDB files should be included in git-lfs

