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
