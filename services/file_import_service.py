import logging
import os.path
import shutil
import uuid
from datetime import datetime
from os import getenv
from typing import Tuple


from fastapi import UploadFile


from core.paths import get_local_base_dir
from processor.import_process.exceptions import FileProcessingError
from processor.import_process.main_graph import kb_import_graph_app
from processor.import_process.state import ImportGraphState, create_default_state
from utils.client.storage_clients import StorageClients
from utils.task_util import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    add_done_task,
    add_running_task,
    update_task_status,
)

logger = logging.getLogger(__name__)


class ImportFileService:
    """文件导入服务：处理文件上传（本地 + MinIO）并运行导入图谱"""

    def _get_date_dir(self) -> str:
        """返回日期目录：本地存储基础目录 + 当天日期(%Y%m%d)"""
        i = get_local_base_dir()
        data_path = os.path.join(i, datetime.now().strftime("%Y%m%d"))

        return data_path

    def process_upload_file(self, file: UploadFile) -> Tuple[str, str, str]:
        """
        处理上传文件：
        1. 生成 task_id，构建归档目录
        2. 标记 upload_file 节点为运行中
        3. 保存文件到本地磁盘
        4. 同步上传到 MinIO
        5. 标记 upload_file 节点完成
        6. 返回 (task_id, file_dir, import_file_path)
        """
        # ① 算日期目录（调 _get_date_dir）
        data_dir = self._get_date_dir()

        # ② 生成 task_id = uuid.uuid4().hex[:8]
        task_id = uuid.uuid4().hex[:8]

        # ③ 拼任务专属目录 = 日期目录 / task_id
        file_dir = os.path.join(data_dir, task_id)

        # ④ 标记 upload_file 节点为运行中
        add_running_task(task_id, "upload_file")

        # ⑤ 调 _save_upload_file_to_local → 拿到 import_file_path
        import_file_path = self._save_upload_file_to_local(file, file_dir)

        # ⑥ 调 _save_upload_file_to_minio
        self._save_upload_file_to_minio(import_file_path, file.filename)

        # ⑦ 标记 upload_file 节点完成
        add_done_task(task_id, "upload_file")

        # ⑧ 返回三件套
        return task_id, file_dir, import_file_path

    def _save_upload_file_to_local(self, file: UploadFile, file_dir: str) -> str:
        """保存上传文件到本地临时目录，返回落盘后的文件路径"""
        # ① 创建目录（如果不存在）
        os.makedirs(file_dir, exist_ok=True)

        # ② 拼文件路径 = file_dir / file.filename
        import_file_path = os.path.join(file_dir, file.filename)

        # ③ 写入文件（try/except 包裹）
        try:
            with open(import_file_path, "wb") as f:
                # 解释：使用 shutil.copyfileobj 将 UploadFile 的底层文件流 (file.file) 
                # 分块复制到本地磁盘文件 (f) 中。这种流式复制方式避免了将整个文件
                # 一次性读入内存，显著降低了内存占用，是处理大文件上传的最佳实践。
                shutil.copyfileobj(file.file, f)
        except IOError as e:
            logger.error(f"Error writing file: {e}")
            raise FileProcessingError(f"Error writing file: {e}")

        # ④ 返回文件路径
        return import_file_path

    def _save_upload_file_to_minio(self, import_file_path: str, filename: str):
        """同步上传原始文件到 MinIO（失败仅记录日志，不阻断导入流程）"""
        # ① 获取 minio 客户端（try/except 包裹，失败直接 return）
        try:
            minio_client = StorageClients.get_minio()
        except (ConnectionError, EnvironmentError) as e:
            logger.error(f"获取minio客户端失败: {e}")
            return

        # ② 获取 bucket_name（存储桶名称）和 object_name（对象名称/路径）
        bucket_name = getenv("MINIO_BUCKET_NAME")
        data_str = datetime.now().strftime("%Y%m%d")
        object_name = f"origin_files/{data_str}/{filename}"

        # ③ 上传（try/except 包裹，失败只记日志）
        try:
            minio_client.fput_object(
                bucket_name,
                object_name,
                import_file_path
            )
        except Exception as e:
            logger.error(f"上传文件到 MinIO 失败: {e}")

    def run_import_graph(self, task_id: str, file_dir: str, import_file_path: str):
        """运行导入 LangGraph 流水线（在后台任务中执行）"""
        try:
            # ① 设置任务状态为 processing
            update_task_status(task_id, TASK_STATUS_PROCESSING)

            # ② 构造初始 state（用 create_default_state）
            global_graph_init_status: ImportGraphState = create_default_state(
                task_id=task_id,
                file_dir=file_dir,
                import_file_path=import_file_path,
            )

            # ③ 流式执行图谱
            for event in kb_import_graph_app.stream(global_graph_init_status):
                for node_name, state in event.items():
                    print(f"[{task_id}] Completed Node: {node_name}")

            # ④ 执行完成，设置状态为 completed
            update_task_status(task_id, TASK_STATUS_COMPLETED)

        # ⑤ 异常处理：设置状态为 failed
        except Exception as e:
            update_task_status(task_id, TASK_STATUS_FAILED)
            print(f"[{task_id}] Error: {e}")


