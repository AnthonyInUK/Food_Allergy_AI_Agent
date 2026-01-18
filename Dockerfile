FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制并安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目所有代码和数据
COPY . .

# 暴露端口
EXPOSE 7860

# 启动命令（注意：Hugging Face 默认端口是 7860）
CMD ["streamlit", "run", "main.py", "--server.port=7860", "--server.address=0.0.0.0"]