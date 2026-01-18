FROM python:3.11-slim

# 创建一个非 root 用户 (Hugging Face 安全要求)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# 安装系统依赖
USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*
USER user

# 复制依赖并安装
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 复制项目文件
COPY --chown=user . .

# Hugging Face Space 必须监听 7860 端口
EXPOSE 7860

# 启动命令
CMD ["streamlit", "run", "main.py", "--server.port=7860", "--server.address=0.0.0.0"]