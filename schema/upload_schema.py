from typing import List, Dict

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """文件上传响应 —— POST /upload 返回"""
    message: str = Field(..., description="响应消息（如：文件上传成功）")
    task_id: str = Field(..., description="任务ID（任务凭证，轮询的钥匙）")

class TaskStatusResponse(BaseModel):
    """任务状态响应 —— GET /task/status 返回"""
    status:str = Field(...,description = "根据任务ID 获取任务状态")
    running_list:List[str] = Field(...,description = "获取指定任务运行中的节点列表")
    done_list:List[str] = Field(...,description = "获取指定任务已完成的节点列表")
    durations:Dict[str, float] = Field(default={},description = "获取所有节点的耗时")


# ─── 临时验证（确认合同与 task_util 数据形状吻合后删除）───
if __name__ == '__main__':
    from utils.task_util import get_task_info

    # 用一个不存在的 task_id 取数据：预期得到 空串/空列表/空字典 的兜底值
    task_info = get_task_info("fake_id_123")
    print("get_task_info 原始返回:", task_info)

    # ** 解包构造模型：验证字段名与字典 key 完全一致、类型校验通过
    response = TaskStatusResponse(**task_info)
    print("TaskStatusResponse:", response.model_dump())
