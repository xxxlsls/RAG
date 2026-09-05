"""
  @Author:LiShuo
  @Time:2026/9/4
  @Desc:
"""
"""查询流程节点基类"""

from abc import ABC, abstractmethod
from typing import TypeVar, Optional
import logging

from processor.query_process.config import QueryConfig, get_config
from processor.query_process.exceptions import QueryProcessError
from utils.task_util import add_running_task, add_done_task, get_task_status, \
    get_done_task_list, get_running_task_list
from utils.sse_util import push_sse_event

T = TypeVar("T")


class BaseNode(ABC):
    name: str = "base_node"

    def __init__(self, config: Optional[QueryConfig] = None):
        self.config = config or get_config()
        self.logger = logging.getLogger(f"query.{self.name}")

    def __call__(self, state: T) -> T:
        task_id = state.get("task_id", "")
        is_stream = state.get("is_stream", False)

        try:
            self.logger.info(f"--- {self.name} 开始 ---")
            if task_id:
                add_running_task(task_id, self.name)
                if is_stream:
                    self._push_progress(task_id)

            result = self.process(state)

            self.logger.info(f"--- {self.name} 完成 ---")
            if task_id:
                add_done_task(task_id, self.name)
                if is_stream:
                    self._push_progress(task_id)

            return result
        except Exception as e:
            self.logger.error(f"{self.name} 执行失败: {e}")
            raise QueryProcessError(message=str(e), node_name=self.name, cause=e)

    @abstractmethod
    def process(self, state: T) -> T:
        pass

    def log_step(self, step_name: str, message: str = ""):
        log_msg = f"[{step_name}]"
        if message:
            log_msg += f" {message}"
        self.logger.info(log_msg)

    @staticmethod
    def _push_progress(task_id: str):
        push_sse_event(task_id, "progress", {
            "status": get_task_status(task_id),
            "done_list": get_done_task_list(task_id),
            "running_list": get_running_task_list(task_id),
        })


def setup_logging(level: int = logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )