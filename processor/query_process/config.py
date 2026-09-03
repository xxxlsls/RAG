"""
  @Author:LiShuo
  @Time:2026/8/28
  @Desc:
"""


"""查询流程配置管理模块

集中管理所有配置项，支持环境变量覆盖。所有属性均采用懒加载模式。
"""


"""
类别	配置项	说明
文本处理	max_context_chars	上下文最大字符数
Rerank	rerank_max_top_k	重排序最大 TopK
        rerank_min_top_k	重排序最小 TopK
        rerank_gap_abs	断崖检测阈值（绝对）
RRF	    rrf_k	RRF 平滑常数
        rrf_max_results	最大返回结果数
检索	    embedding_search_limit	向量搜索返回数量
        hyde_search_limit	HyDE 搜索返回数量
商品确认	item_name_high_confidence	高置信度阈值
        item_name_mid_confidence	中置信度阈值
        item_name_max_options	最大候选数量
LLM	    openai_api_base	API 地址
        default_model	默认模型名
数据库	milvus_url	Milvus 地址
        chunks_collection	切片集合名
"""

from dataclasses import dataclass, field
from typing import Optional
import os

from dotenv import load_dotenv
load_dotenv()


@dataclass
class QueryConfig:
    """查询流程配置。"""

    # ==================== 文本处理配置 ====================
    max_context_chars: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
    )

    # ==================== Rerank 配置 ====================
    rerank_max_top_k: int = field(
        default_factory=lambda: int(os.getenv("RERANK_MAX_TOP_K", "10"))
    )
    rerank_min_top_k: int = field(
        default_factory=lambda: int(os.getenv("RERANK_MIN_TOP_K", "3"))
    )
    rerank_gap_abs: float = field(
        default_factory=lambda: float(os.getenv("RERANK_GAP_ABS", "0.5"))
    )

    # ==================== RRF 配置 ====================
    rrf_k: int = field(
        default_factory=lambda: int(os.getenv("RRF_K", "60"))
    )
    rrf_max_results: int = field(
        default_factory=lambda: int(os.getenv("RRF_MAX_RESULTS", "10"))
    )

    # ==================== 检索配置 ====================
    embedding_search_limit: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_SEARCH_LIMIT", "10"))
    )
    hyde_search_limit: int = field(
        default_factory=lambda: int(os.getenv("HYDE_SEARCH_LIMIT", "5"))
    )

    # ==================== 商品确认节点配置 ====================
    item_name_high_confidence: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_HIGH_CONFIDENCE", "0.7"))
    )
    item_name_mid_confidence: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_MID_CONFIDENCE", "0.6"))
    )
    item_name_max_options: int = field(
        default_factory=lambda: int(os.getenv("ITEM_NAME_MAX_OPTIONS", "5"))
    )
    item_name_dense_weight: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_DENSE_WEIGHT", "0.5"))
    )
    item_name_sparse_weight: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_SPARSE_WEIGHT", "0.5"))
    )

    # ==================== LLM 配置 ====================
    openai_api_base: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_BASE", "")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    default_model: str = field(
        default_factory=lambda: os.getenv("LLM_DEFAULT_MODEL", "")
    )
    item_model: str = field(
        default_factory=lambda: os.getenv("ITEM_MODEL", "")
    )
    mcp_dashscope_base_url: str = field(
        default_factory=lambda: os.getenv("MCP_DASHSCOPE_BASE_URL", "")
    )

    # ==================== Milvus 配置 ====================
    milvus_url: str = field(
        default_factory=lambda: os.getenv("MILVUS_URL", "")
    )
    chunks_collection: str = field(
        default_factory=lambda: os.getenv("CHUNKS_COLLECTION", "")
    )
    item_name_collection: str = field(
        default_factory=lambda: os.getenv("ITEM_NAME_COLLECTION", "")
    )
    entity_name_collection: str = field(
        default_factory=lambda: os.getenv("ENTITY_NAME_COLLECTION", "")
    )

    @classmethod
    def from_env(cls) -> "QueryConfig":
        """从环境变量加载配置。"""
        return cls()


# ==================== 全局单例 ====================

_config: Optional[QueryConfig] = None


def get_config() -> QueryConfig:
    """获取配置单例。"""
    global _config
    if _config is None:
        _config = QueryConfig.from_env()
    return _config