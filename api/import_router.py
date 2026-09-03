"""
    router 是总装车间——把前面所有零件（paths、schema、service、deps、task_util）装配成一个能跑的 FastAPI 服务
    """
import uvicorn
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Depends
from fastapi.responses import FileResponse
import os

from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from core.deps import get_import_file_service
from core.paths import get_front_page_dir
from schema.upload_schema import UploadResponse, TaskStatusResponse
from services.file_import_service import ImportFileService

from utils.task_util import get_task_info


def register_router(app:FastAPI):
    @app.get("/import")
    async def import_root():
        return FileResponse(path=os.path.join(get_front_page_dir(), "import.html"))

    @app.post("/upload",response_model=UploadResponse)
    async def upload_file_endpoint(
            background_tasks: BackgroundTasks,
            file: UploadFile = File(...),
            service: ImportFileService = Depends(get_import_file_service),
    ):
        # 1同步处理上传
        task_id, file_dir, import_file_path = service.process_upload_file(file)
        #2异步启动图谱
        background_tasks.add_task(service.run_import_graph, task_id, file_dir, import_file_path)
        # 3. 立即返回
        return UploadResponse(message="文件上传成功", task_id=task_id)

    @app.get("/status/{task_id}", response_model=TaskStatusResponse)
    async def get_status_endpoint(task_id: str):
        task_info = get_task_info(task_id)
        return TaskStatusResponse(**task_info)


def create_app() -> FastAPI:
    app = FastAPI(description="知识库导入", version="v1.0")

    # CORS 中间件（防御性配置）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,  # 注意：* 和 credentials=True 不能同时用
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 挂载静态资源
    front_page_dir = get_front_page_dir()
    if front_page_dir and os.path.exists(front_page_dir):
        app.mount("/front", StaticFiles(directory=front_page_dir))

    # 注册路由
    register_router(app)
    return app

if __name__ == '__main__':
    uvicorn.run(
        app=create_app(),
        host="127.0.0.1", port=8000,
        log_level="info"
    )