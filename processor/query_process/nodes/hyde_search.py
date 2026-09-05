"""HyDE 检索节点

使用 Hypothetical Document Embedding 技术：
先让 LLM 生成假设性文档，再将其与原查询拼接后向量化检索，提升语义召回质量。
"""

import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Dict, Any, List, Tuple, Union

from langchain_core.messages import HumanMessage, SystemMessage

from processor.query_process.base import BaseNode
from processor.query_process.config import get_config
from processor.query_process.exceptions import StateFieldError
from processor.query_process.prompt import USER_HYDE_PROMPT_TEMPLATE
from processor.query_process.state import QueryGraphState
from utils.client.ai_clients import AIClients
from utils.client.storage_clients import StorageClients
from utils.embedding_util import generate_bge_m3_hybrid_vectors
from utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query, item_names_filter


class HyDeSearchNode(BaseNode):
    """HyDE 检索节点：三路并行检索中的语义增强通道

    流程: 参数校验 → LLM 生成假设文档 → 拼接原查询 → 向量化 → 混合检索
    """

    name: str = "search_embedding_hyde"

    def __init__(self):
        super().__init__(config=get_config())

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        # ① 参数校验：调用 _validate_query_inputs 获取 validated_query 和 validate_item_names
        validated_query, validated_item_names = self._validate_query_inputs(state)

        # ② 生成假设性文档（LLM 失败时返回空字符串，不阻塞流程）
        hy_document = self._generate_hy_document(validated_query, validated_item_names)


        # ③ 获取嵌入模型
        try:
            bge_m3client = AIClients.get_bge_m3_client()
        except ConnectionError:
            self.logger.error("BGE-M3 模型连接失败")
            return state

        # ④ 获取 Milvus 客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError:
            self.logger.error("Milvus 连接失败")
            return state

        # ⑤ 拼接 + 向量化（原问题 + 换行符 + 假设文档）
        embedding_document = f"{validated_query}\n{hy_document}"
        try:
            embedding_result = generate_bge_m3_hybrid_vectors(model=bge_m3client,
                                           embedding_documents=[embedding_document]
                                           )
            logger.info(f"嵌入结果: {embedding_result}")
        except Exception as e:
            self.logger.error(f"嵌入失败 原因:{str(e)}")
            return state

        # ⑥ 构建过滤表达式以及表达式参数（复用 milvus_util 的模板参数方式）
        filter_expr, filter_expr_param = item_names_filter(item_names=validated_item_names)

        try:
            # ⑦ 创建混合搜索请求（稠密 + 稀疏两路）
            hybrid_search_requests = create_hybrid_search_requests(
                dense_vector = embedding_result['dense'][0],
                sparse_vector = embedding_result['sparse'][0],
                expr=filter_expr,
                expr_params=filter_expr_param
            )

            # ⑧ 执行混合搜索（只有一个查询，取 [0]）
            hybrid_search_result = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=self.config.chunks_collection,
                search_requests=hybrid_search_requests,
                output_fields=['chunk_id', 'content', 'item_name', 'title']
            )

            # ⑨ 判空：结果为空或第一路结果为空 → return state
            if not hybrid_search_result or not hybrid_search_result[0]:
                return state

            # ⑩ 更新 state 返回（只返回变更字段，供后续 rrf 节点融合）
            return {"hyde_embedding_chunks": hybrid_search_result[0]}
        except Exception as e:
            self.logger.error(f"HyDE 混合检索失败 原因:{str(e)}")
            return state

    def _generate_hy_document(self, validated_query: str, validate_item_names: List[str]) -> str:
        """使用 LLM 生成假设性文档"""
        # ① 获取 LLM 客户端（普通模式，不需要 JSON 输出）
        llm_client = AIClients.get_openai_llm(response_format=False)

        # ② 判空：客户端不存在 → return ""
        if not llm_client:
            return ""

        # ③ 组装提示词

        user_prompt = USER_HYDE_PROMPT_TEMPLATE.format(
            item_hint=validate_item_names,
            rewritten_query=validated_query
        )
        system_prompt = f"您是一位{validate_item_names}的技术文档领域的专家，主要擅长编写技术文档、操作手册、文档规格说明"

        try:
            # ④ 调用 LLM（消息列表：[SystemMessage(...), HumanMessage(...)]）
            llm_response = llm_client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])

            # ⑤ 提取内容
            content = getattr(llm_response, 'content', "").strip()

            # ⑥ 内容为空 → return ""，否则返回内容
            if not content:
                return ""
            else:
                return content
        except Exception as e:
            # LLM 失败不阻塞：返回空字符串，后续拼接退化为纯原问题检索
            self.logger.error(f"LLM调用失败:{str(e)}")
            return ""

    def _validate_query_inputs(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        """校验输入参数"""
        # ① 获取 state 的 rewritten_query 和 item_names
        rewritten_query = state.get("rewritten_query")
        item_names = state.get("item_names")

        # ② 校验：
        #   - rewritten_query 为空或非 str → raise StateFieldError(
        #         node_name=self.name, field_name='rewritten_query', expected_type=str)
        #   - item_names 为空或非 list → raise StateFieldError(
        #         node_name=self.name, field_name='item_names', expected_type=list)
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name='rewritten_query', expected_type=str)
        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name='item_names', expected_type=list)


        # ③ 返回 (rewritten_query, item_names)
        return rewritten_query, item_names


# ================================================================== #
#                        测试入口                                      #
# ================================================================== #

if __name__ == '__main__':
    state = {
        "rewritten_query": "RS-12 数字万用表如何测量直流电压？",
        "item_names": ["RS-12 数字万用表"]
    }

    hyde_search = HyDeSearchNode()
    result = hyde_search.process(state)

    for r in result.get('hyde_embedding_chunks', []):
        print('HyDE检索结果:')
        print(json.dumps(r, ensure_ascii=False, indent=2))
