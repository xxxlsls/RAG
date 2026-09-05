"""
查询路由
"""
import os
import asyncio

import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, Depends
from fastapi.responses import FileResponse, StreamingResponse, RedirectResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from core.deps import get_query_service
from core.paths import get_front_page_dir
from schema.query_schema import QueryResponse, StreamSubmitResponse, QueryRequest
from services.query_service import QueryService
from utils.sse_util import sse_generator, create_sse_queue
from processor.query_process.base import setup_logging


def register_routes(app):

    @app.get("/")
    async def index():
        # 根路径直接重定向到聊天页，避免访问 http://127.0.0.1:8001/ 时 404
        return RedirectResponse(url="/chat")

    @app.get("/chat")
    async def chat_page():
        path = os.path.join(get_front_page_dir(), "chat.html")
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Chat page not found")
        return FileResponse(path)

    @app.post("/query", response_model=QueryResponse | StreamSubmitResponse)
    async def query(
            request: QueryRequest,
            background_tasks: BackgroundTasks,
            service: QueryService = Depends(get_query_service),
            ):
        # ① 获取 session_id（前端没传就自动生成）
        session_id = request.session_id or service.generate_session_id()

        # ② 生成 task_id
        task_id = service.generate_task_id()

        # ③ 流式模式
        if request.is_stream:
            #3.1 必须在返回响应前创建 SSE 队列
            create_sse_queue(task_id)
            # 3.2 后台运行查询图谱
            background_tasks.add_task(
                service.run_query_graph, task_id, session_id, request.query, True
            )
            # 3.3 返回 task_id 给前端
            return StreamSubmitResponse(message="处理中", task_id=task_id, session_id=session_id)


        # ④ 非流式模式：丢到线程池避免阻塞事件循环，await 等待执行完成
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, service.run_query_graph, task_id, session_id, request.query, False
        )

        # ⑤ 获取答案（此时图谱已执行完毕）
        answer = service.get_answer(task_id)

        # ⑥ 返回答案
        return QueryResponse(message="处理完成", session_id=session_id, answer=answer)


    @app.get("/stream/{task_id}")
    async def stream(task_id: str, request: Request) -> StreamingResponse:
        # SSE 必须显式禁用缓存与代理缓冲，否则 Nginx / 浏览器会攒包，
        # 导致 delta / progress 事件一次性涌出，失去流式效果
        headers = {
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(
            sse_generator(task_id, request),
            media_type="text/event-stream",
            headers=headers,
        )

    @app.get("/history/{session_id}")
    async def get_history(
            session_id: str, limit: int = 50,
            service: QueryService = Depends(get_query_service),
    ):
        try:
            items = service.get_history(session_id, limit)
            return {"session_id": session_id, "items": items}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"history error: {e}")

    @app.delete("/history/{session_id}")
    async def clear_chat_history(
            session_id: str,
            service: QueryService = Depends(get_query_service),
    ):
        count = service.clear_history(session_id)
        return {"message": "History cleared", "deleted_count": count}


def create_app() -> FastAPI:
    app = FastAPI(title="Query Service", description="知识库查询服务")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # 挂载前端静态文件路径
    front_page_dir = get_front_page_dir()
    if front_page_dir and os.path.exists(front_page_dir):
        app.mount("/front", StaticFiles(directory=front_page_dir, html=True), name="front")

    register_routes(app)
    return app


if __name__ == "__main__":
    setup_logging()
    uvicorn.run(app=create_app(), host="127.0.0.1", port=8001)

