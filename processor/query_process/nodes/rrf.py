"""RRF 融合排序节点

使用 Reciprocal Rank Fusion（倒数排名融合）算法融合多路检索结果，
为后续 rerank 节点提供统一的候选集。
"""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Dict, Any, List, Tuple

from processor.query_process.base import BaseNode
from processor.query_process.config import get_config
from processor.query_process.state import QueryGraphState


class RrfNode(BaseNode):
    """RRF 融合排序节点

    流程: 收集多路检索结果 → 格式规整 → 带权重 RRF 融合 → 按得分降序返回
    """

    name = "rrf"

    def __init__(self):
        super().__init__(config=get_config())
        # ① 从 config 读取 RRF 相关参数（平滑常数、最大返回数）
        rrf_k = self.config.rrf_k
        rrf_max_results = self.config.rrf_max_results
        self.rrf_max_results = rrf_max_results
        self.rrf_k = rrf_k

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """执行 RRF 融合"""
        # ① 获取各路检索结果
        vector_search_chunks = state.get('embedding_chunks') or []
        hyde_search_chunks = state.get('hyde_embedding_chunks') or []

        # ② 为不同路的搜索结果设置不同的权重（默认等权 1.0:1.0）
        search_source = {
            "vector_search_result": (self._normalize_input(vector_search_chunks), 1.0),
            "hyde_search_result":   (self._normalize_input(hyde_search_chunks),   1.0),
            }

        # ③ 构建 rrf_inputs（list of (docs, weight) 元组）
        rrf_inputs = list(search_source.values())

        # ④ 利用 RRF 公式计算每个 chunk 的总得分
        rrf_merge_results:List[Tuple[Dict[str, Any], float]] = self._rrf_merge(
            rrf_inputs, self.rrf_k, self.rrf_max_results
        )

        # ⑤ 获取 rrf_chunks（只取文档，不要分数）
        rrf_chunks = [doc for doc, _ in rrf_merge_results]
        logger.info(f"RRF 融合完成，返回 {len(rrf_chunks)} 条结果")

        # ⑥ 记录分数范围（便于调试）
        if rrf_merge_results:
            scores = [s for _, s in rrf_merge_results]
            logger.info(f"分数范围: [{min(scores):.6f}, {max(scores):.6f}]")
        # ⑦ 更新 state 并返回
        state['rrf_chunks'] = rrf_chunks
        return state

    def _normalize_input(self, rrf_input: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """统一处理各路检索到的结果（提取 entity 字段）

        上游返回结构：[{"entity": {"chunk_id": ..., "content": ...}, "distance": ...}, ...]
        本节点需要结构：[{"chunk_id": ..., "content": ...}, ...]
        """
        # ① 定义结果容器
        diff_path_result = []

        # ② 判空：输入为空 → return []
        if not rrf_input:
            return []

        # ③ 遍历该路的所有结果：
        #   - 跳过非 dict 的元素
        #   - 跳过 entity 为空的元素
        #   - 将 entity 追加到结果列表
        for item in rrf_input:
            if not isinstance(item, dict):
                continue
            if not item.get("entity"):
                continue
            diff_path_result.append(item["entity"])

        # ④ 返回
        return diff_path_result

    def  _rrf_merge(self, rrf_inputs, _rrf_k, _top_k) -> List[Tuple[Dict[str, Any], float]]:
        """利用 RRF 公式计算每一个文档的总得分

        RRF 公式: score(d) = Σ weight_i / (k + rank_i(d))
        - rank 从 1 开始（符合排名语义）
        - 同一 chunk 在多路命中时，分数累加
        - 使用 setdefault 保留首次遇到的文档版本（去重）

        Args:
            rrf_inputs: list of (docs, weight) 元组
            _rrf_k: 平滑常数（默认 60）
            _top_k: 融合后最大返回文档数

        Returns:
            [(doc, score), ...] 按 score 降序，最多 top_k 条
        """
        # ① 定义两个字典：
        #   - chunk_scores: chunk_id → RRF 累计分数
        #   - chunk_data:   chunk_id → 文档数据（首次遇到的版本）
        chunk_scores = {}
        chunk_data = {}

        # ② 双层循环：外层遍历各路 (rrf_input, weight)，内层 enumerate(rrf_input, 1) 得到 (rank, doc)
        for rrf_input, weight in rrf_inputs:
            for i, doc in enumerate(rrf_input, 1):
                chunk_id = doc.get('chunk_id')
                if not chunk_id: continue
                # RRF 公式累加
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + weight / (_rrf_k + i)
                # 保留首次遇到的文档版本
                chunk_data.setdefault(chunk_id, doc)

        # ③ 按得分降序排序，截取前 top_k 条
        sorted_results = sorted(
            [(chunk_data[cid], score) for cid, score in chunk_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_results[:_top_k] if _top_k else sorted_results


# ================================================================== #
#                        测试入口                                      #
# ================================================================== #

if __name__ == '__main__':
    # 模拟两路检索结果
    # chunk_1 命中 2 路（预期最高分）
    # chunk_2 命中 2 路
    # chunk_3, chunk_4 各命中 1 路
    mock_state = {
        "embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "向量搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_2", "content": "向量搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_3", "content": "向量搜索结果#3"}},
        ],
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": "chunk_2", "content": "HyDE搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_1", "content": "HyDE搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_4", "content": "HyDE搜索结果#3"}},
        ],
    }

    print("【输入状态】:")
    print(f"  embedding_chunks: {len(mock_state['embedding_chunks'])} 条")
    print(f"  hyde_embedding_chunks: {len(mock_state['hyde_embedding_chunks'])} 条")

    rrf_node = RrfNode()
    result = rrf_node.process(mock_state)

    print("\n【融合结果】:")
    for i, chunk in enumerate(result["rrf_chunks"], 1):
        print(f"[{i}] {chunk.get('chunk_id')} - {chunk.get('content')}")
