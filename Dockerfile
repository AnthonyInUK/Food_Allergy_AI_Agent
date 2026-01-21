FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    git-lfs \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# 初始化 git-lfs
RUN git lfs install

# 复制依赖并安装
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 复制项目文件（包括数据库）
COPY . .

# 确保数据库文件存在且可读
RUN if [ ! -f "data/food_data.db" ]; then echo "WARNING: food_data.db not found"; fi && \
    if [ ! -d "data/chroma_db" ]; then echo "WARNING: chroma_db directory not found"; fi

# Hugging Face Space 必须监听 7860 端口
EXPOSE 7860

# 启动命令
CMD ["python", "-m", "streamlit", "run", "main.py", "--server.port=7860", "--server.address=0.0.0.0"]