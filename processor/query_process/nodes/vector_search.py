"""向量检索节点

对用户查询进行向量化（稠密 + 稀疏），根据已确认的商品名在 Milvus 中执行混合搜索，返回相关切片。
"""

import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Dict, Any, List, Tuple, Union

from processor.query_process.base import BaseNode
from processor.query_process.config import get_config
from processor.query_process.exceptions import StateFieldError
from processor.query_process.state import QueryGraphState
from utils.client.ai_clients import AIClients
from utils.client.storage_clients import StorageClients
from utils.embedding_util import generate_bge_m3_hybrid_vectors
from utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query, item_names_filter


class VectorSearchNode(BaseNode):
    """向量检索节点：多路并行检索中的核心检索通道"""

    name = "search_embedding"

    def __init__(self):
        super().__init__(config=get_config())

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        # ① 参数校验：调用 _validate_state 获取 validated_query 和 validate_item_names
        validated_query, validate_item_names = self._validate_state(state)

        # ② 获取嵌入模型
        try:
            bge_m3_client= AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"嵌入模型获取失败: {e}")
            return state

        # ③ 获取 Milvus 客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"Milvus 客户端获取失败: {e}")
            return state


        # ④ 对问题向量化
        try:
            embedding_result = generate_bge_m3_hybrid_vectors(
                model=bge_m3_client,
                embedding_documents=[validated_query]
            )
            logger.info(f"嵌入结果: {embedding_result}")
        except (ValueError, RuntimeError) as e:
            # 注意：generate_bge_m3_hybrid_vectors 实际抛的是 ValueError / RuntimeError
            self.logger.error(f"问题向量化失败: {e}")
            return state

        # ⑤ 构建过滤表达式以及表达式参数（基于已确认的商品名做 IN 过滤）
        filter_expr, filter_expr_param = item_names_filter(item_names=validate_item_names)

        try:
            # ⑥ 创建混合搜索请求（稠密 + 稀疏两路）
            hybrid_search_request = create_hybrid_search_requests(
                dense_vector=embedding_result['dense'][0],
                sparse_vector=embedding_result['sparse'][0],
                expr=filter_expr,
                expr_params=filter_expr_param
            )

            # ⑦ 执行混合搜索（结果结构与查询向量一一对应，本节点只有一个查询，取 [0]）
            hybrid_search_reps = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=self.config.chunks_collection,
                search_requests=hybrid_search_request,
                output_fields=['chunk_id', 'content', 'item_name', 'title']
            )

            # ⑧ 判空：结果为空或第一路结果为空 → return state
            if not hybrid_search_reps or not hybrid_search_reps[0]:
                return state

            # ⑨ 更新 state 返回（只返回变更字段，供后续 rrf 节点融合）
            return {"embedding_chunks": hybrid_search_reps[0]}
        except Exception as e:
            self.logger.error(f"混合检索失败 原因:{str(e)}")
            return state

    def _validate_state(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        """校验输入参数"""
        # ① 获取 state 的 rewritten_query 和 item_names
        rewritten_query = state.get("rewritten_query")
        item_names = state.get("item_names")

        # ② 校验：
        if not rewritten_query or type(rewritten_query) != str:
            raise StateFieldError( node_name=self.name, field_name='rewritten_query', expected_type=str,message="校验重写问题失败")
        if not item_names or type(item_names) != list:
            raise StateFieldError( node_name=self.name, field_name='item_names', expected_type=list,message="校验商品名失败")

        # ③ 返回 (rewritten_query, item_names)
        return rewritten_query, item_names


# ================================================================== #
#                        测试入口                                      #
# ================================================================== #

if __name__ == '__main__':
    state = {
        "rewritten_query": "bios密码的作用",
        "item_names": ["华为擎云 L540"]
    }

    vector_search = VectorSearchNode()
    result = vector_search.process(state)

    for r in result.get('embedding_chunks', []):
        print(json.dumps(r, ensure_ascii=False, indent=2))
