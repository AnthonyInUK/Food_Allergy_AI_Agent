# 生产部署（HTTPS + 环境变量）

## 1. 准备域名与 `.env`

1. 将 **A 记录** 指到服务器公网 IP（`PUBLIC_DOMAIN` 与证书域名一致）。
2. 在仓库根目录复制并编辑 `.env`（参考根目录 `.env.example`），至少填写：

| 变量 | 说明 |
|------|------|
| `PUBLIC_DOMAIN` | 对外域名，如 `app.example.com`（勿带 `https://`） |
| `PUBLIC_API_URL` | 浏览器里访问 API 的**源**，须与 Caddy 对外一致，如 `https://app.example.com` |
| `ACME_EMAIL` | Let's Encrypt 联系邮箱（用于证书到期通知） |
| `CORS_ALLOW_ORIGINS` | 允许的前端源，与 `PUBLIC_API_URL` 同源时填同一 URL，多个用英文逗号分隔 |
| `OPENAI_API_KEY` / `TAVILY_API_KEY` | 与开发一致 |
| `REDIS_URL` | 默认 `redis://redis:6379/0`（compose 内服务名 `redis`） |
| `DATABASE_URL` | **推荐显式设置**。未设置时 compose 默认为 `postgresql+psycopg://postgres:postgres@postgres:5432/food_ai`（须与下方 `POSTGRES_*` 一致） |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | 可选，默认 `postgres` / `postgres` / `food_ai`；若修改密码或库名，请同步改 `DATABASE_URL` 与 postgres 健康检查期望的库名（见 compose 内 `postgres` 服务） |

可选：`LANGSMITH_*`、`UVICORN_WORKERS`（默认 2）、`ENABLE_PG_CONVERSATION_CHECKPOINT`（默认 `true`，对话检查点写 PG）。

### 1.1 PostgreSQL 与产品数据

生产栈已包含 **`postgres` 服务**；结构化查询（SQL Agent、图片分析里的产品库等）统一走 **`DATABASE_URL`** 指向的 PostgreSQL，不再依赖容器内的 SQLite。

- **卷**：数据库文件在 **`foodai_pg`**；应用数据（Chroma、可选的本地文件等）仍在 **`foodai_data`**（`/app/data`）。
- **首次有数据**：若你本地仍有 `data/food_data.db`，可在仓库根目录对**已启动**的 Postgres 执行迁移（需本机能访问该 `DATABASE_URL`，例如临时在 compose 里给 postgres 映射 `5432:5432` 后执行）：

  ```bash
  export DATABASE_URL='postgresql+psycopg://postgres:postgres@127.0.0.1:5432/food_ai'
  python scripts/migrate_sqlite_to_postgres.py
  ```

  无 SQLite 备份时，需自行向 `products` 表导入数据或从其它源初始化 schema（迁移脚本内的 `ensure_schema` 会创建 `products` 表）。

## 2. 构建并启动

```bash
docker compose -f docker-compose.prod.yml --env-file .env up -d --build
```

首次启动 Caddy 会向 Let's Encrypt 申请证书，**80/443 须对公网开放**。

## 3. 验证

- 浏览器打开 `https://$PUBLIC_DOMAIN`，应能加载前端并调用 API。
- 证书与 Caddy 数据在卷 `caddy_data` / `caddy_config`；Chroma 等在卷 `foodai_data`（`/app/data`）；PostgreSQL 在卷 `foodai_pg`。

## 4. 开发环境

本机开发仍用 `./scripts/dev.sh` 或 `docker compose`（`docker-compose.yml`，带 `--reload`）。

## 5. 仅 HTTP 内网（无域名）

可暂时不用 Caddy，直接暴露 `api:8000` / `frontend:3000` 并自行处理 TLS；或把 `PUBLIC_DOMAIN` 改为内网域名并在 Caddy 使用 `tls internal`（需自行改 `deploy/Caddyfile`，此处不展开）。

## 6. Hugging Face Space 仍显示 Streamlit？

仓库已改为 **`Dockerfile` + `uvicorn api_server:app`（7860）**，不再包含 `main.py` / Streamlit。若 Space 里仍是旧界面，通常是下面之一：

1. **Space 未用 Docker SDK**  
   打开 Space → **Settings**，确认 **SDK** 为 **Docker**（若选 Streamlit/Gradio，HF 会按旧方式找 `main.py`，与根目录 `Dockerfile` 无关）。

2. **仍是旧镜像**  
   在 Space **Settings** 里 **Factory reboot** / 重新触发构建，或推一个新 commit 让 CI 再 `git push` 一次 Space，等 **Build** 完成后再打开 **App**。

3. **当前架构**  
   Space 上跑的是 **仅 API**；聊天 UI 在本机或别处的 **Next.js**（`NEXT_PUBLIC_API_URL` 指向该 Space 的 URL）。根路径 `/` 返回 JSON 说明，交互在 **`/docs`** 或前端里完成。

4. **CI 强制重建**  
   `.github/workflows/deploy-to-hf.yml` 在推送到 Space 后会调用 `restart_space(..., factory_reboot=True)`，避免 HF 长期复用旧容器里已删除的 Streamlit 镜像层。需配置仓库 Secret `HUGGINGFACE_TOKEN`（write 权限）。
