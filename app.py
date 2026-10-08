from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
import hmac
from agent import (
    ask_agent,
    forget_shared_thread,
    get_shared_agent,
    shutdown_agent,
    update_shared_knowledge,
)
import uvicorn
import asyncio
import history
import json
import os
import shutil
import sys
import tempfile
from rag import add_to_knowledge_base, delete_document, list_documents

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 单个上传文件的大小上限（MB）。默认 50：扫描件每页都要 OCR，一次上传会
# 长时间占着知识库写锁，把所有人的提问堵在门外，所以要有个天花板。
# 需要放宽就设环境变量 MAX_UPLOAD_MB。每次请求现读，测试可以临时改小。
def max_upload_bytes() -> int:
    try:
        mb = int(os.getenv("MAX_UPLOAD_MB", "50"))
    except ValueError:
        mb = 50
    if mb < 1:
        mb = 50
    return mb * 1024 * 1024


def public_error(action: str, exc: BaseException) -> str:
    """记完整异常到终端，返回给浏览器的句子里不带路径和堆栈。"""
    sys.stderr.write(f"[ERROR] {action}：{type(exc).__name__}: {exc}\n")
    return f"{action}失败，请查看运行服务的终端日志"


def require_kb_token(x_kb_token: Optional[str] = Header(default=None)) -> None:
    """上传、删除文件、删除会话都要带与 .env 里 KB_TOKEN 一致的口令。"""
    expected = os.getenv("KB_TOKEN", "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="未配置 KB_TOKEN，已拒绝上传和删除")
    provided = x_kb_token or ""
    try:
        matched = hmac.compare_digest(provided, expected)
    except (TypeError, ValueError):
        matched = False
    if not matched:
        raise HTTPException(status_code=401, detail="口令不正确")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时不用预热，第一次提问会自动拉起 MCP 子进程；
    # 关闭时把子进程收干净，避免残留
    yield
    await shutdown_agent()


app = FastAPI(lifespan=lifespan)

# 页面和接口都在本机 8000，不对外站开放跨域，也不带 cookie。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    # 前端可传会话 ID 来隔离不同用户的上下文；不传则共用 default 线程
    session_id: Optional[str] = "default"
    # 保留字段以兼容旧前端；对话历史现在由服务端 MemorySaver 按 session_id 管理
    history: Optional[List[Dict[str, str]]] = []
    # 设为 true 时，响应里会额外带上 model ⇄ tools 循环的完整轨迹
    trace: bool = False
    # "chat"（快速）或 "reasoner"（深度思考，会把完整思考过程流式推给前端）
    model: Optional[str] = "chat"


class ChatResponse(BaseModel):
    response: str
    # trace=true 时才有值，每行是 loop 的一步
    trace: Optional[List[str]] = None


def _record(session_id: str, question: str, answer: str) -> None:
    """把这一轮问答落盘，供左侧「历史会话」和重启后的上下文重建使用。

    本地 sqlite 写入是微秒级，这里直接同步调用：放到 finally 里 await
    反而会在客户端断连（生成器被强行关掉）时出问题。
    写历史失败绝不能影响回答本身，所以整段兜住。
    """
    try:
        history.record_turn(session_id, question, answer)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[WARNING] 记录对话历史失败：{type(exc).__name__}: {exc}\n")


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    session_id = req.session_id or "default"
    try:
        answer = await ask_agent(
            req.message, thread_id=session_id, trace=req.trace, model=req.model
        )
        trace_lines = None
        if req.trace:
            agent = await get_shared_agent()
            trace_lines = list(agent.last_trace)
        _record(session_id, req.message, answer)
        return ChatResponse(response=answer, trace=trace_lines)
    except Exception as e:
        raise HTTPException(status_code=500, detail=public_error("回答", e))


@app.post("/chat/stream")
async def chat_stream_endpoint(req: ChatRequest):
    """SSE 流式问答：边跑边把「步骤 / 正文片段」推给前端。

    每个事件是一行 `data: {json}`：
      step      中间步骤（正在检索 / 正在计算）
      tool      工具返回全文（前端默认折叠，展开可看全部）
      reasoning 模型思考内容（deepseek-reasoner 才有）
      token     回答正文片段
      reset     之前吐的正文作废（模型那是在自言自语，接着要调工具）
      error     出错
      done      收尾
    """
    try:
        agent = await get_shared_agent()
    except Exception as e:
        raise HTTPException(status_code=500, detail=public_error("启动助手", e))

    session_id = req.session_id or "default"

    async def event_source():
        parts = []  # 正文累积；reset 表示模型刚才那段自言自语作废
        try:
            async for item in agent.astream_answer(
                req.message, thread_id=session_id, model=req.model
            ):
                kind = item.get("type")
                if kind == "token":
                    parts.append(item.get("text") or "")
                elif kind == "reset":
                    parts.clear()
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        except Exception as e:
            # 响应头早就发出去了，HTTP 状态码已经改不了，只能用事件把错误送出去
            payload = {"type": "error", "text": public_error("回答", e)}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        finally:
            # 客户端中途断开时生成器会被关掉，这里同样会执行，那一轮也不会丢
            _record(session_id, req.message, "".join(parts))
        yield 'data: {"type": "done"}\n\n'

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    _: None = Depends(require_kb_token),
):
    # 检查文件扩展名
    allowed_ext = ['.pdf', '.txt', '.md', '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp']
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_ext:
        raise HTTPException(status_code=400, detail="不支持的文件格式，仅支持 PDF, TXT, MD, PNG, JPG, JPEG, BMP, TIFF, WEBP")

    limit = max_upload_bytes()
    limit_mb = limit // (1024 * 1024)

    # 只取文件名，挡掉客户端塞进来的路径；空名字给个兜底
    display_name = os.path.basename(file.filename or "").strip() or "未命名文档"

    # 先按声明的大小拦一道，省掉整段读盘
    declared = getattr(file, "size", None)
    if declared is not None and declared > limit:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大（{declared / 1048576:.1f} MB），单个文件上限 {limit_mb} MB",
        )

    # 保存临时文件（边写边数字节，不依赖声明值）
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_path = tmp.name
            written = 0
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"文件过大，单个文件上限 {limit_mb} MB",
                    )
                tmp.write(chunk)

        # 占着和提问同一把锁：先让子进程放开库文件，写完再丢掉旧句柄。
        # 传原始文件名进去：临时文件名不能当来源写进 metadata。
        await update_shared_knowledge(
            lambda: add_to_knowledge_base(tmp_path, display_name)
        )
        return {"message": f"文件 {file.filename} 上传并索引成功"}
    except HTTPException:
        raise  # 413 / 400 原样透出，别被下面的 500 吞掉
    except Exception as e:
        raise HTTPException(status_code=500, detail=public_error("上传", e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)  # 删除临时文件


# ---------- 会话历史 ----------
@app.get("/sessions")
async def list_sessions_endpoint():
    """左侧「历史会话」列表：按最近活动时间倒序。"""
    try:
        return {"sessions": history.list_sessions()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=public_error("读取会话列表", e))


@app.get("/sessions/{session_id}")
async def get_session_endpoint(session_id: str):
    session = history.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    messages = history.get_messages(session_id)
    session = dict(session)
    session["message_count"] = len(messages)
    return {"session": session, "messages": messages}


@app.delete("/sessions/{session_id}")
async def delete_session_endpoint(
    session_id: str,
    _: None = Depends(require_kb_token),
):
    try:
        removed = history.delete_session(session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=public_error("删除会话", e))
    # 历史删了，MemorySaver 里那条线程也一起抹掉，否则模型还记得删掉的内容
    await forget_shared_thread(session_id)
    return {"deleted": removed}


# ---------- 知识库文件 ----------
class DocumentRequest(BaseModel):
    name: str


@app.get("/documents")
async def list_documents_endpoint():
    """当前知识库里已入库的文件（按来源名聚合，带块数）。"""
    try:
        return await asyncio.to_thread(list_documents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=public_error("读取知识库文件列表", e))


@app.post("/documents/delete")
async def delete_document_endpoint(
    req: DocumentRequest,
    _: None = Depends(require_kb_token),
):
    name = os.path.basename((req.name or "").strip())
    if not name:
        raise HTTPException(status_code=400, detail="缺少文件名")
    try:
        # 走和上传同一条路：先让 MCP 子进程放开库文件，删完再重新打开
        return await update_shared_knowledge(lambda: delete_document(name))
    except Exception as e:
        raise HTTPException(status_code=500, detail=public_error("删除文档", e))


# 挂载静态文件目录（用绝对路径，避免工作目录不同导致 404）
app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "static"), html=True), name="static")

if __name__ == "__main__":
    # 默认只听本机。要给局域网用再设 KB_HOST=0.0.0.0，并先改掉 KB_TOKEN。
    host = os.getenv("KB_HOST", "127.0.0.1")
    port = int(os.getenv("KB_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
