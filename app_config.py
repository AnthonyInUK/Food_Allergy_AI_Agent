import os
from functools import lru_cache
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv(override=True)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except Exception:
        return default


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    return value if value else default


class RuntimeSettings:
    """Centralized runtime toggles for cache/checkpoint/retry/indexing knobs."""

    def __init__(self) -> None:
        # Cache toggles
        self.enable_response_cache = _env_bool("ENABLE_RESPONSE_CACHE", True)
        self.enable_redis_response_cache = _env_bool("ENABLE_REDIS_RESPONSE_CACHE", True)
        self.response_cache_ttl_seconds = _env_int("RESPONSE_CACHE_TTL_SECONDS", 3 * 24 * 60 * 60)
        self.inproc_cache_ttl_seconds = _env_int("CACHE_TTL_SECONDS", 3600)

        # Retry toggles
        self.enable_node_retry = _env_bool("ENABLE_NODE_RETRY", True)
        self.node_retry_max_attempts = _env_int("NODE_RETRY_MAX_ATTEMPTS", 3)
        self.node_retry_initial_interval = _env_float("NODE_RETRY_INITIAL_INTERVAL", 0.5)
        self.node_retry_backoff_factor = _env_float("NODE_RETRY_BACKOFF_FACTOR", 2.0)
        self.node_retry_max_interval = _env_float("NODE_RETRY_MAX_INTERVAL", 8.0)

        # Checkpoint/persistence toggles
        self.enable_pg_conversation_checkpoint = _env_bool("ENABLE_PG_CONVERSATION_CHECKPOINT", True)
        self.enable_important_qa_redis = _env_bool("ENABLE_IMPORTANT_QA_REDIS", True)
        self.important_data_ttl_seconds = _env_int("IMPORTANT_DATA_TTL_SECONDS", 3 * 24 * 60 * 60)

        # Vector store (Chroma) disk persistence — directory must survive restarts / Docker rebuilds
        self.chroma_persist_directory = _env_str("CHROMA_PERSIST_DIRECTORY", "data/chroma_db")
        self.chroma_text_collection_name = _env_str("CHROMA_TEXT_COLLECTION", "food_products")

        # CLIP-style image retrieval (separate Chroma collection; default off)
        self.enable_image_vector_retrieval = _env_bool("ENABLE_IMAGE_VECTOR_RETRIEVAL", False)
        self.chroma_image_collection_name = _env_str("CHROMA_IMAGE_COLLECTION", "product_images")
        self.clip_model_name = _env_str("CLIP_MODEL_NAME", "clip-ViT-B-32")
        self.image_vector_top_k = _env_int("IMAGE_VECTOR_TOP_K", 5)
        self.image_index_batch_size = _env_int("IMAGE_INDEX_BATCH_SIZE", 32)
        self.image_download_timeout_sec = _env_int("IMAGE_DOWNLOAD_TIMEOUT_SEC", 20)

        # Retrieval/indexing knobs (for large datasets/chunking)
        self.vector_top_k = _env_int("VECTOR_TOP_K", 5)
        self.vector_chunk_size = _env_int("VECTOR_CHUNK_SIZE", 800)
        self.vector_chunk_overlap = _env_int("VECTOR_CHUNK_OVERLAP", 120)

        # 质量优先：为省延迟而调低 top_k / 关联网 等时，不把召回与路由压到「可能变差」的区间（可用 QUALITY_FIRST=false 完全按 env 数值）
        self.quality_first = _env_bool("QUALITY_FIRST", True)
        self.vector_top_k_floor = _env_int("VECTOR_TOP_K_FLOOR", 5)

        # 启动时预热向量检索（加载 embedding + Chroma + 一次空查询），缩短首个用户请求的检索段延迟
        self.enable_retrieval_warmup = _env_bool("ENABLE_RETRIEVAL_WARMUP", True)

        # NDJSON 流里是否启用 LLM 正文 token 增量（子线程合并队列；默认关，避免与 ASGI 线程池叠套时偶发卡死 / 长时间无包）
        self.enable_stream_llm_deltas = _env_bool("ENABLE_STREAM_LLM_DELTAS", False)

        # Debug logging controls
        self.enable_debug_log = _env_bool("ENABLE_DEBUG_LOG", False)
        self.debug_log_sample_rate = _env_float("DEBUG_LOG_SAMPLE_RATE", 0.1)

    def effective_vector_top_k(self) -> int:
        """检索条数：QUALITY_FIRST 时对 VECTOR_TOP_K 做下限保护，避免过小 k 伤召回。"""
        k = max(1, int(self.vector_top_k))
        if not self.quality_first:
            return k
        return max(k, max(1, int(self.vector_top_k_floor)))

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "cache": {
                "enable_response_cache": self.enable_response_cache,
                "enable_redis_response_cache": self.enable_redis_response_cache,
                "response_cache_ttl_seconds": self.response_cache_ttl_seconds,
                "inproc_cache_ttl_seconds": self.inproc_cache_ttl_seconds,
            },
            "retry": {
                "enable_node_retry": self.enable_node_retry,
                "node_retry_max_attempts": self.node_retry_max_attempts,
                "node_retry_initial_interval": self.node_retry_initial_interval,
                "node_retry_backoff_factor": self.node_retry_backoff_factor,
                "node_retry_max_interval": self.node_retry_max_interval,
            },
            "checkpoint": {
                "enable_pg_conversation_checkpoint": self.enable_pg_conversation_checkpoint,
                "enable_important_qa_redis": self.enable_important_qa_redis,
                "important_data_ttl_seconds": self.important_data_ttl_seconds,
            },
            "retrieval": {
                "chroma_persist_directory": self.chroma_persist_directory,
                "chroma_text_collection_name": self.chroma_text_collection_name,
                "enable_image_vector_retrieval": self.enable_image_vector_retrieval,
                "chroma_image_collection_name": self.chroma_image_collection_name,
                "clip_model_name": self.clip_model_name,
                "image_vector_top_k": self.image_vector_top_k,
                "vector_top_k": self.vector_top_k,
                "vector_top_k_floor": self.vector_top_k_floor,
                "effective_vector_top_k": self.effective_vector_top_k(),
                "quality_first": self.quality_first,
                "enable_retrieval_warmup": self.enable_retrieval_warmup,
                "enable_stream_llm_deltas": self.enable_stream_llm_deltas,
                "vector_chunk_size": self.vector_chunk_size,
                "vector_chunk_overlap": self.vector_chunk_overlap,
            },
            "debug": {
                "enable_debug_log": self.enable_debug_log,
                "debug_log_sample_rate": self.debug_log_sample_rate,
            },
        }


@lru_cache(maxsize=1)
def get_runtime_settings() -> RuntimeSettings:
    return RuntimeSettings()
