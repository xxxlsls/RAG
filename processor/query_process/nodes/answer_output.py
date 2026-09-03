"""答案输出节点

组装提示词、调用 LLM 生成答案，支持流式/非流式输出，写入历史记录。
"""

import logging

from numpy.distutils.conv_template import header
from sympy.abc import delta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import List, Dict, Any, Tuple

from processor.import_process.base import BaseNode
from processor.query_process.config import get_config
from processor.query_process.prompt import ANSWER_PROMPT
from processor.query_process.state import QueryGraphState
from utils.client.ai_clients import AIClients
from utils.task_util import set_task_result
from utils.sse_util import push_sse_event, SSEEvent
from utils.mongo_history_util import save_chat_message


class AnswerOutputNode(BaseNode):
    """答案输出节点

    流程: 检查已有答案 → 构建提示词 → LLM 生成 → 写入历史 → 发送结束事件
    """

    name: str = "answer_output"

    def __init__(self):
        super().__init__(config=get_config())

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """执行答案生成"""
        # ① 获取 task_id 和 is_stream
        # task_id = "task_abc123"
        task_id = state.get("task_id")
        # is_stream = False
        is_stream = state.get("is_stream")

        # ② 检查已有答案（商品确认节点可能已生成提示答案，直接推送即可）
        # state["answer"] = "请问您是指A还是B？"  （商品确认节点生成的澄清提示）
        if state.get("answer"):
             self._push_existing_answer(state)

        # ③ 没有已有答案 → 构建提示词 + 调用 LLM
        else:
            prompt = self._build_prompt(state)
            state["prompt"] = prompt
            self._generate_answer(state, prompt)

        # ④ 写入历史记录（不管答案从哪来，都要记录）
        self._write_history(state)

        # ⑤ 流式模式发送结束事件
        if is_stream:
            push_sse_event(task_id, SSEEvent.FINAL, {"answer": state.get("answer", "")})

        return state

    # ================================================================== #
    #                    已有答案推送                                       #
    # ================================================================== #

    def _push_existing_answer(self, state: QueryGraphState):
        """非流式模式：存入任务结果；流式模式：让 FINAL 统一推送。"""
        if not state.get("is_stream"):
            set_task_result(state["task_id"], "answer", state["answer"])

    # ================================================================== #
    #                    提示词构建                                         #
    # ================================================================== #

    def _build_prompt(self, state: QueryGraphState) -> str:
        """根据检索结果、历史对话组装 LLM 提示词

        字符预算分配优先级：文档（最高）> 历史对话 > 截断
        """
        # ① 获取字符预算
        char_budget = self.config.max_context_chars


        # ② 获取问题和商品名
        # question = "RS-12数字万用表如何测量电压？"
        question = state.get("rewritten_query") or state.get("original_query", "")
        # item_names = ["RS-12数字万用表"]
        item_names = state.get("item_names") or []

        # ③ 格式化检索文档（先用预算，文档优先级最高）
        # state["reranked_docs"] = [{'content': '主板短路通常表现为...', 'title': '主板维修手册', 'chunk_id': 'local_1', 'url': '', 'source': 'local', 'score': 0.9937}, ...]
        context_str, char_budget = self._format_reranked_docs(state.get("reranked_docs") or [], char_budget)

        # ④ 格式化历史对话（用剩余预算）
        # state["history"] = [{'role': 'user', 'text': '万用表是什么？'}, {'role': 'assistant', 'text': '万用表是一种多功能电子测量仪器...'}]
        history_str, char_budget = self._format_chat_history(state.get("history") or [], char_budget)

        # ⑤ 组装提示词（ANSWER_PROMPT 有 4 个占位符：context, history, item_names, question）
        return ANSWER_PROMPT.format(
            context=context_str or "无参考内容",
            history=history_str if history_str else "暂无历史对话",
            item_names=", ".join(item_names),
            question=question,
        )

    def _format_reranked_docs(self, reranked_docs: List[Dict], char_budget: int) -> Tuple[str, int]:
        """格式化重排序文档，带字符预算控制

        每篇文档格式：
        [1] [source=local] [chunk_id=xxx] [title=xxx] [score=0.9234]
        文档内容...

        超出预算后截断，返回 (格式化文本, 剩余预算)
        """
        # ① 初始化
        formatted_lines = []
        used_chars = 0

        # ② 遍历文档
        # reranked_docs 数据结构 = [{'content': '主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。', 'title': '主板维修手册', 'chunk_id': 'local_1', 'url': '', 'source': 'local', 'score': 0.993734466224161}, {'content': '主板通电前先打各主供电电感的对地阻值，阻值偏低就是短路。', 'title': '短路查修指南', 'chunk_id': None, 'url': 'https://example.com/repair', 'source': 'web', 'score': 0.9813106915732508}]
        for idx,doc in enumerate(reranked_docs,1):
            content = doc.get("content", "").strip()
            if not content: continue

            # 格式化文档
            meta_tags = [f"[{idx}]"]
            for field, template in [("source", "[source={}]"),
                                    ("chunk_id", "[chunk_id={}]"),
                                    ("url", "[url={}]"),
                                    ("title", "[title={}]"),
                                    ("score", "[score={}]")]:
                field_value = str(doc.get(field, "")).strip()
                if field_value:
                    meta_tags.append(template.format(field_value))
                    doc_entry = " ".join(meta_tags) + "\n" + content

            if used_chars + len(doc_entry) > char_budget:
                break
            formatted_lines.append(doc_entry)

            used_chars += len(doc_entry) + 1
            char_budget -= used_chars

        # ③ 返回格式化文本和剩余预算
        return "\n\n".join(formatted_lines), char_budget

    def _format_chat_history(self, chat_history: List[Dict], char_budget: int) -> Tuple[str, int]:
        """格式化历史对话

        格式：
        用户: xxx
        助手: xxx
        """
        # ① 初始化
        formatted_lines = []
        used_chars = 0
        role_label_map = {"user": "用户", "assistant": "助手"}

        # ② 遍历历史消息
        for message in chat_history:
            role = message.get("role", "")
            text = message.get("text", "")
            if not text or role not in role_label_map: continue

            role = role_label_map.get(role, "")
            formatted_lines.append(f"{role}: {text}")
            used_chars += len(formatted_lines[-1]) + 1
            char_budget -= used_chars

        # ③ 返回格式化文本和剩余预算
        return "\n".join(formatted_lines), char_budget

    # ================================================================== #
    #                    LLM 生成                                         #
    # ================================================================== #

    def _generate_answer(self, state: QueryGraphState, prompt: str):
        """调用 LLM 生成答案（流式/非流式）"""
        # ① 获取 LLM 客户端（response_format=False，答案不需要 JSON 格式）
        llm_client = AIClients.get_openai_llm(response_format=False)
        if llm_client is None:
                logger.error("LLM 客户端初始化失败")
                state["answer"] = "抱歉，系统暂时无法生成回答。"
                return

        # ② 根据模式选择生成方式
        task_id = state["task_id"]
        if state.get("is_stream"):
            state["answer"] = self._stream_generate(llm_client, prompt, task_id)
        else:
            state["answer"] = self._invoke_generate(llm_client, prompt)
            set_task_result(task_id, "answer", state["answer"])

    def _invoke_generate(self, llm_client, prompt: str) -> str:
        """非流式生成"""
        try:
            response = llm_client.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"生成回答出错: {e}")
            return "抱歉，生成回答时出现错误。"

    def _stream_generate(self, llm_client, prompt: str, task_id: str) -> str:
        """流式生成，逐 chunk 推送 delta 事件"""
        accumulated_answer = ""

        try:
            for chunk in llm_client.stream(prompt):
                delta_text = getattr(chunk, "content", "") or ""
                if delta_text:
                    accumulated_answer += delta_text
                    push_sse_event(task_id, "delta", {"delta": delta_text})
        except Exception as e:
            logger.error(f"流式生成出错: {e}")
            return "抱歉，生成回答时出现错误。"
        return accumulated_answer

    # ================================================================== #
    #                    历史记录                                           #
    # ================================================================== #

    def _write_history(self, state: QueryGraphState):
        """将用户问题和助手回答写入 MongoDB 历史记录"""
        # ① 获取会话信息
        # session_id = "sess_001"
        session_id = state.get("session_id", "")
        # rewritten_query = "RS-12数字万用表如何测量电压？"
        rewritten_query = state.get("rewritten_query", "") or state.get("original_query", "")
        # item_names = ["RS-12数字万用表"]
        item_names = state.get("item_names") or []

        try:
            # ② 写用户问题
            save_chat_message(
                session_id=session_id,
                role="user",
                text=state.get("original_query", ""),
                rewritten_query=rewritten_query,
                item_names=item_names,
            )


            # ③ 写助手回复
            if state.get("answer"):
                save_chat_message(
                    session_id=session_id,
                    role="assistant",
                    text=state["answer"],
                    rewritten_query=rewritten_query,
                    item_names=item_names,
                )
        except Exception as e:
            # ④ 写入失败不阻塞流程
            logger.warning(f"写入历史记录失败: {e}")


# ================================================================== #
#                        测试入口                                      #
# ================================================================== #

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    mock_state = {
        "task_id": "test_task_001",
        "session_id": "test_session_001",
        "is_stream": True,
        "original_query": "万用表怎么测电压？",
        "rewritten_query": "RS-12数字万用表如何测量电压？",
        "item_names": ["RS-12数字万用表"],
        "reranked_docs": [
            {
                "content": "数字万用表测量电压步骤：1. 将旋钮转到V档位；2. 黑表笔插COM孔，红表笔插V孔；3. 将表笔并联到被测点两端。",
                "source": "local",
                "chunk_id": "chunk_001",
                "title": "万用表使用手册",
                "score": 0.9234
            },
            {
                "content": "测量直流电压时需注意正负极性，红表笔接正极，黑表笔接负极。",
                "source": "web",
                "url": "https://example.com/guide",
                "title": "电压测量指南",
                "score": 0.8756
            }
        ],
        "history": [
            {"role": "user", "text": "万用表是什么？"},
            {"role": "assistant", "text": "万用表是一种多功能电子测量仪器..."}
        ],
    }

    print("【输入状态】:")
    print(f"  query: {mock_state['rewritten_query']}")
    print(f"  item_names: {mock_state['item_names']}")
    print(f"  reranked_docs: {len(mock_state['reranked_docs'])} 篇")

    node = AnswerOutputNode()
    result = node.process(mock_state)

    print("\n【生成结果】:")
    print("-" * 60)
    print(result.get("answer", "无答案"))
    print("-" * 60)
