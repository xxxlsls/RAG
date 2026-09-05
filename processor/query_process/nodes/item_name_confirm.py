"""商品名称确认节点

从用户查询中提取商品名称，通过向量相似度匹配与数据库中已有商品对齐确认。
"""

import logging
import json
import re
from json import JSONDecodeError
from typing import Dict, Any, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from processor.query_process.base import BaseNode
from processor.query_process.config import get_config
from processor.query_process.prompt import ITEM_NAME_EXTRACT_TEMPLATE
from processor.query_process.state import QueryGraphState
from utils.client.ai_clients import AIClients
from utils.client.storage_clients import StorageClients
from utils.embedding_util import generate_bge_m3_hybrid_vectors
from utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query
from utils.mongo_history_util import get_recent_messages

logging.basicConfig(level=logging.INFO)
logger  = logging.getLogger(__name__)


class ItemNameExtractor:
    """商品名提取器：基于用户原始问题和历史对话提取商品名"""
    def __init__(self):
        self._config = get_config()


    def extract_item_name(self, original_query: str, history_text: str) -> Dict[str, Any]:
        """
        LLM 根据用户原始问题提取商品名

        Args:
            original_query: 用户原始问题
            history_text: 格式化后的历史对话文本

        Returns:
            {"item_names": [...], "rewritten_query": "..."}
        """
        # ① 默认返回值（兜底）
        result = {"item_names":[], "rewritten_query": original_query}

        # ② 获取 LLM 客户端（JSON Mode）
        llm_client = AIClients.get_openai_llm(response_format=True)
        if llm_client is None:
            return result

        # ③ 组装提示词
        human_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(
                   history_text=history_text if history_text else "暂无上下文",
                   query=original_query
               )
        system_prompt = "你是一个专业的客服助手，擅长理解用户意图和提取关键信息。"

        # ④ 调用 LLM
        llm_response = llm_client.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])

        # ⑤ 空值检查
        if not llm_response.content.strip():
            return result

        # ⑥ 清洗解析
        # 注意：_clean_parse 内部抛的是 ValueError，而 JSONDecodeError 只是它的子类，
        # 单写 except JSONDecodeError 抓不到，必须一起兜住
        try:
            parsed_result = self._clean_parse(llm_response.content)
            result["rewritten_query"] = parsed_result.get("rewritten_query") or original_query
            result["item_names"] = parsed_result.get("item_names") or []
        except (JSONDecodeError, ValueError) as e:
            logger.error(f"清洗以及解析LLM的输出失败：{str(e)}")

        # ⑦ 返回 result
        return result

    def _clean_parse(self, llm_content: str) -> Dict[str, Any]:
        """清洗并解析 LLM 响应"""
        # ① 去掉 markdown 代码块围栏
        cleaned = re.sub(r"^```(?:json)?\s*", "", llm_content.strip())
        content = re.sub(r"\s*```$", "", cleaned)

        # ② 反序列化 + 清洗字段
        try:
            parsed_llm_result = json.loads(content)
            return {"item_names": parsed_llm_result.get("item_names", []),
                    "rewritten_query": parsed_llm_result.get("rewritten_query", "")}
        except JSONDecodeError as e:
            raise ValueError(f"JSON反序列LLM的输出失败：{str(e)}")


class ItemNameAligner:
    """商品名对齐器：向量匹配 + 评分对齐 + 分数差异过滤"""
    def __init__(self):
        self._config = get_config()


    def match_align_filter(self, item_names: List[str]) -> Tuple[List[str], List[str]]:
        """执行匹配、对齐、过滤三步流程"""
        # ① 查询向量数据库
        search_result = self._match_vector(item_names)

        # ② 评分对齐
        confirmed, options = self._item_name_score_align(search_result)

        # ③ 分数差异过滤（仅当 confirmed 有多个时）
        if len(confirmed) > 1:
            confirmed = self._item_name_score_filter(confirmed, search_result)

        # ④ 返回
        return confirmed, options

    def _match_vector(self, item_names: List[str]) -> List[Dict[str, Any]]:
        """根据 LLM 提取的商品名，查询向量数据库"""
        # ① 定义搜索结果
        search_results = []

        # ② 获取 milvus_client
        milvus_client = StorageClients.get_milvus_client()

        # ③ 获取嵌入模型
        embedding_model = AIClients.get_bge_m3_client()

        # ④ 生成混合向量（{"dense": [...], "sparse": [...]}，列表与 item_names 按下标一一对应）
        hybrid_embedding_result = generate_bge_m3_hybrid_vectors(embedding_model, item_names)

        # ⑤ 遍历每个商品名，执行混合检索（collection_name 用 config 里的 item_name_collection）
        collection_name = get_config().item_name_collection
        for index, extract_item_name in enumerate(item_names):
            # 5.1 创建混合检索请求（传当前商品名的稠密 + 稀疏向量）
            search_request = create_hybrid_search_requests(
                hybrid_embedding_result["dense"][index],
                hybrid_embedding_result["sparse"][index]
            )
            # 5.2 执行混合检索，单次查询只有一条结果列表，取 res[0]
            res = execute_hybrid_search_query(milvus_client, collection_name, search_request)
            # 5.3 解析成统一结构 {"matches": [{"item_name": 商品名, "score": 融合分}]}
            matches = [
                {"item_name": hit.entity.get("item_name"), "score": hit.distance}
                for hit in (res[0] if res else [])
            ]
            # 5.4 加入 search_results（顺序与 item_names 一致，携带提取名供精确匹配用）
            search_results.append({"extracted_name": extract_item_name, "matches": matches})
        # ⑥ 返回
        return search_results

    def _item_name_score_align(self, search_results: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """根据评分对齐商品名称（三档：高置信确认 / 中置信候选 / 低置信忽略）"""
        # ① 定义两个容器
        confirmed = []
        options = []

        # ② 遍历搜索结果
        for item_name_search_result in search_results:
            extracted_name = item_name_search_result.get("extracted_name")
            # 按分数降序排列 matches（混合检索不保证返回顺序）
            matches = sorted(item_name_search_result.get("matches"), key=lambda x: x["score"], reverse=True)

            # 2.1 高置信区（≥ 0.7）
            high = [m for m in matches if m["score"] >= 0.7]

            if high:
                # 场景 A: 高置信中存在与提取名完全匹配 → 直接确认它（最精准）
                exact = next((h for h in high if str(h["item_name"]) == extracted_name), None)
                if exact:
                    picked = exact["item_name"]
                    if picked not in confirmed:
                        confirmed.append(picked)
                # 场景 B: 只有一条高置信结果 → 直接确认它（无歧义）
                elif len(high) == 1:
                    picked = high[0]["item_name"]
                    if picked not in confirmed:
                        confirmed.append(picked)
                # 场景 C: 多条高置信且无精确匹配 → 前 3 条进候选让用户选（有歧义）
                else:
                    for h in high[:3]:
                        picked = h["item_name"]
                        if picked not in options and picked not in confirmed:
                            options.append(picked)
            else:
                # 2.2 中置信区（0.6 ≤ score < 0.7）：取前 3 条进候选；低于 0.6 直接忽略（匹配不可靠）
                mid = [m for m in matches
                       if m["score"] >= 0.6
                       and m["item_name"] not in options
                       and m["item_name"] not in confirmed]
                for m in mid[:3]:
                    options.append(m["item_name"])

        # ③ 返回（options 最多 3 个）
        return confirmed, options[:3]


    def _item_name_score_filter(self, confirmed: List[str], search_results: List[Dict[str, Any]]):
        """分数差异过滤，剔除误判"""
        # ① 收集每个商品名的最高分数
        item_name_score = {}
        for search_result in search_results:
            for match in search_result["matches"]:
                if match["item_name"] in confirmed:
                    item_name_score[match["item_name"]] = max(item_name_score.get(match["item_name"], 0), match["score"])
        # ② 排序 + 取基准分
        sorted_item_name_score = sorted(item_name_score.items(), key=lambda x: x[1], reverse=True)
        max_item_name_score = sorted_item_name_score[0][1]

        # ③ 保留分数差 ≤ 0.15 的
        return [name for name, score in sorted_item_name_score if max_item_name_score - score <= 0.15]


class ItemNameConfirmNode(BaseNode):
    """商品名称确认节点

    流程: 获取历史 → LLM提取商品名 → 向量匹配 → 评分对齐 → 更新状态 → 历史回填
    """
    name = "item_name_confirm"

    def __init__(self):
        super().__init__(config=get_config())
        self._item_name_extractor = ItemNameExtractor()
        self._item_name_aligner = ItemNameAligner()

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # ① 获取用户的原始问题和 session_id
        original_query = state.get("original_query")
        session_id = state.get("session_id")

        # ② 获取历史对话
        chat_history = get_recent_messages(session_id, limit=10)
        history_text = "\n".join(
            f"{msg.get('role')}: {msg.get('text')}" for msg in chat_history
        )  # "role: text\n..." 格式

        # ③ 调用 LLM 提取商品名
        clean_llm_result = self._item_name_extractor.extract_item_name(original_query, history_text)
        item_names = clean_llm_result.get("item_names") or []
        rewritten_query = clean_llm_result.get("rewritten_query") or original_query

        # ④ 向量匹配 + 过滤
        if item_names:
            confirmed, options = self._item_name_aligner.match_align_filter(item_names)
        else:
            confirmed, options = [], []

        # ⑤ 决策分支，更新 state
        self._decide(state, item_names, confirmed, options, rewritten_query)

        # ⑥ 历史回填
        state["history"] = chat_history


        # ⑦ 返回 state
        return state

    def _decide(self, state: QueryGraphState, item_names: List[str],
                confirmed: List[str], options: List[str], rewritten_query: str):
        """根据对齐结果更新 state"""
        if confirmed:
            state['rewritten_query'] = rewritten_query
            state['item_names'] = confirmed
        elif options:
            state['answer'] = f"我不确定您指的是哪款产品。您是在询问以下产品吗：{','.join(options)}？"
        else:
            state['answer'] = "抱歉，我无法识别..."
        pass


# ================================================================== #
#                        测试入口                                      #
# ================================================================== #

if __name__ == "__main__":
    state = {
        "original_query": "如何使用 万用表",
        "session_id": "1234567890",
        "history": []
    }
    # 创建 ItemNameConfirmNode 实例
    item_name_confirm_node = ItemNameConfirmNode()
    #调用 process 并打印结果
    result = item_name_confirm_node.process(state)
    print(result)


