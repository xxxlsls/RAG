"""Rerank 重排序节点

使用 Reranker 模型（交叉编码器）对 RRF 融合结果和网络搜索结果进行精排，
并通过最大断崖检测算法实现动态 TopK 截断。
"""

import logging

from django.template.defaultfilters import upper
from modelscope.preprocessors.templates.utils import upper_bound

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Dict, Any, List

from processor.import_process.base import BaseNode
from processor.query_process.config import get_config
from processor.query_process.state import QueryGraphState
from utils.client.ai_clients import AIClients


class RerankNode(BaseNode):
    """Rerank 重排序节点

    流程: 合并多源文档 → Reranker 精排 → 断崖检测动态截断
    """

    name: str = "rerank"

    def __init__(self):
        super().__init__(config=get_config())

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """执行重排序"""
        # ① 获取查询文本（优先 rewritten_query，更完整）
        user_query = state.get('rewritten_query', '') or state.get('original_query', '')

        # ② 合并多源文档（本地 RRF + 网络搜索）
        merged_multi_docs = self._merge_multi_source_docs(state)

        # ③ Reranker 精排
        reranked_docs: List[Dict[str, Any]] = self._rerank_merged_docs(user_query, merged_multi_docs)

        # ④ 断崖检测动态截断
        cutoff_docs = self._cliff_cutoff(reranked_docs)

        # ⑤ 写入 state 并返回
        return {"reranked_docs": cutoff_docs}

    def _merge_multi_source_docs(self, state: QueryGraphState) -> List[Dict[str, Any]]:
        """合并本地 RRF 文档和 Web 搜索文档，统一格式

        本地文档从 rrf_chunks 取（含 chunk_id/content/title）
        网络文档从 web_search_docs 取（含 url/snippet/title）
        统一为 {content, title, chunk_id, url, source} 结构
        """
        final_docs = []

        # ① 遍历本地 RRF 文档
        for rrf_doc in state.get('rrf_chunks'):
            if not isinstance(rrf_doc, dict): continue
            content = rrf_doc.get('content', '').strip()
            if not content: continue
            title = rrf_doc.get('title', '').strip()
            chunk_id = rrf_doc.get('chunk_id')
            format_rrf_doc = self._format_rrf_docs(content=content, title=title, chunk_id=chunk_id, source="local")
            final_docs.append(format_rrf_doc)

        # ② 遍历网络搜索文档
        for web_doc in state.get('web_search_docs'):
            if not isinstance(web_doc, dict): continue
            snippet = web_doc.get('snippet', '').strip()
            if not snippet: continue
            title = web_doc.get("title", "").strip()
            if not title: continue
            url = web_doc.get('url', '').strip()
            if not url: continue
            format_web_doc = self._format_rrf_docs(content=snippet, title=title, url=url, source="web")
            final_docs.append(format_web_doc)

        # ③ 记录日志
        logger.info(f"收集到准备进行 Rerank 精排的文档 {len(final_docs)} 篇")

        return final_docs

    def _format_rrf_docs(self, content: str, title: str = "", chunk_id=None,
                         url: str = "", source: str = "") -> Dict[str, Any]:
        """构建统一的文档结构"""
        return {"content": content, "title": title, "chunk_id": chunk_id, "url": url, "source": source}

    def _rerank_merged_docs(self, user_query: str, merged_multi_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """使用 Reranker 模型对合并文档进行精排

        注意：
        - compute_score 单文档时返回 float 而非 list，需要防护
        - 使用 normalize=True 直接归一化到 0~1（sigmoid）
        - 失败时降级：score 设为 None，不阻塞流程
        """
        # ① 判空
        if not merged_multi_docs: return []

        # ② 获取 Reranker 模型
        rerank_model = AIClients.get_bge_m3_rerank_client()
        if rerank_model is None:
            logger.error("重排序模型获取失败")
            return []

        # ③ 构建 (query, doc_content) 对
        query_doc_pairs = [(user_query, doc.get('content')) for doc in merged_multi_docs]
        try:
            # ④ 计算相关性得分（normalize=True 直接归一化到 0~1）
            rerank_scores = rerank_model.compute_score(sentence_pairs=query_doc_pairs, normalize=True)

            # ⑤ 单文档防护：compute_score 单篇返回 float 而非 list
            if isinstance(rerank_scores, (float, int)):
                rerank_scores = [rerank_scores]

            # ⑥ 映射分数到文档，按分数降序排序
            score_docs = [{**doc,"score": score} for doc, score in zip(merged_multi_docs, rerank_scores)]
            sorted_score_docs = sorted(score_docs, key=lambda x: x["score"], reverse=True)
            return sorted_score_docs

        except Exception as e:
            # ⑦ 降级处理：Reranker 失败不阻塞，score 设为 None
            logger.error(f"Rerank 重排序失败: {str(e)}")
            return [{**doc, "score": None} for doc in merged_multi_docs]

    def _cliff_cutoff(self, reranked_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """断崖检测动态截断（最大断崖检测版）

        扫描所有相邻文档分数差，找到最大落差位置进行截断，
        同时保证至少返回 lower_bound 个文档。

        与"首次断崖"版的区别：不 break，遍历整个扫描范围找最大 gap。

        参数:
            reranked_docs: 按分数降序排列的文档列表
            rerank_min_top_k: 最少返回文档数
            rerank_max_top_k: 最多返回文档数

        返回值:
            截断后的文档列表
        """
        # ① 计算上下界
        upper_bound = min(self.config.rerank_max_top_k, len(reranked_docs))
        lower_bound = min(self.config.rerank_min_top_k, upper_bound)

        # ② 边界情况：只有 0 或 1 篇，直接返回
        if upper_bound <= 1:
            return reranked_docs[:upper_bound]

        # ③ 初始化：默认截断到上界（不断崖就全保留）
        cut_off = upper_bound
        max_gap = 0.0

        # ④ 遍历相邻文档，找最大断崖
        for i in range(0, upper_bound - 1):
            current_score = reranked_docs[i].get("score")
            next_score = reranked_docs[i + 1].get("score")
            gap = current_score - next_score
            if next_score is None or current_score is None: continue
            if gap >= self.config.rerank_gap_abs and gap > max_gap:
                max_gap = gap
                cut_off = i + 1
                logger.info(f"位置{cut_off}发生断崖，gap={max_gap:.4f}")

        # ⑤ 兜底：至少保留 lower_bound 个
        # cut_off  = max(cut_off, lower_bound)
        return reranked_docs[:cut_off]


# ================================================================== #
#                        测试入口                                      #
# ================================================================== #

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    mock_state = {
        "rewritten_query": "怎么测这块主板的短路问题？",
        "rrf_chunks": [
            {"chunk_id": "local_1", "title": "主板维修手册",
             "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。"},
            {"chunk_id": "local_2", "title": "闲聊",
             "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"},
        ],
        "web_search_docs": [
            {"url": "https://example.com/repair", "title": "短路查修指南",
             "snippet": "主板通电前先打各主供电电感的对地阻值，阻值偏低就是短路。"},
            {"url": "https://example.com/news", "title": "科技新闻",
             "snippet": "苹果发布新款手机，A系列芯片性能提升20%。"},
        ],
    }

    print("【输入状态】:")
    print(f"  查询: {mock_state['rewritten_query']}")
    print(f"  本地文档: {len(mock_state['rrf_chunks'])} 篇")
    print(f"  网络文档: {len(mock_state['web_search_docs'])} 篇")

    node = RerankNode()
    result = node.process(mock_state)

    print("\n【重排序结果】:")
    for i, doc in enumerate(result.get("reranked_docs", []), 1):
        score = doc.get('score')
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"[{i}] score={score_str} | {doc.get('source', '?'):5} | {doc.get('content', '')[:50]}...")
