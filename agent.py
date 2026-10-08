import asyncio
import inspect
import os
import sys
from contextlib import AsyncExitStack
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import Tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

from history import recent_messages

try:  # mcp 依赖 anyio，正常都在
    from anyio import BrokenResourceError, ClosedResourceError
    _DEAD_SESSION_ERRORS = (BrokenResourceError, ClosedResourceError, EOFError)
except ImportError:  # pragma: no cover
    _DEAD_SESSION_ERRORS = (EOFError,)

# override=True：机器级环境变量里可能存着过期的 key，而 load_dotenv 默认
# 「已存在就不覆盖」，会让 .env 里的新 key 被悄悄压掉（踩过一次）。
# 这里明确让 .env 当唯一事实来源。
load_dotenv(override=True)

SYSTEM_PROMPT = """你是专业的知识库问答助手，回答必须严谨准确，禁止编造内容。

【工具使用规则】
1. 内部文档、制度、产品相关问题，必须优先调用 rag_search 工具检索知识库
2. 数值计算类问题，必须调用 calculator 工具
3. 知识库没有的内容，如实告知用户，禁止编造
4. 检索结果中的 [来源: ...] 必须在回答里保留（文件名、页码），不要编造来源
5. 如果检索结果以「[提示] 重排模型认为没有高相关段落」开头，说明这是低置信度的兜底结果：
   能从中明确回答就回答（照样带来源），答不上来就如实说没找到，不要据此编造

【计算类问题的标准流程】
- 如果用户要求"求和/最大/最高/最低/平均"等计算，必须分两步：
  第一步：只调用一次 rag_search。查询字符串里必须保留「最高/最大/最低/最小/所有/求和/平均」这些词
  （例如用户问最高 PSNR，就检索「最高 PSNR」，不要只写「PSNR」）。
  带上这些词时，一次检索就会返回知识库里的全部相关条目，不要换关键词再检索。
  第二步：把提取出的数值整理成一个算式，调用 calculator 计算，最后给出结果。
- 不要自己心算，一律交给 calculator。
- calculator 支持 max / min / sum / mean / avg / abs / round / pow 函数。
  求最大值直接写 max(26.47, 26.02, 25.66)，求和直接写 sum([1.5, 2.5])，
  求平均直接写 mean([1.5, 2.5]) 或 avg([1.5, 2.5])，
  不要用"逐个相减再比较"这类绕远路的方式。
- 如果 calculator 返回「计算出错」，说明表达式语法有问题：
  读一遍它给出的语法说明，修正表达式后再调用；**绝不要把同一个表达式原样重试**。
- 只需要查一个数值（例如"最高的 PSNR 是多少"）时，不需要调用 calculator，
  检索到后直接回答即可，并带来源。

【重要】
- 当 rag_search 返回包含具体数值（如 PSNR、SSIM 等）的文本时，你必须从该文本中提取出用户所需的具体数值，并以简洁的方式回答，不要复制整段文本。
- 如果 rag_search 返回以 [知识库异常] 开头，说明知识库本身有问题（路径错误或库为空），
  请把该提示原样转达给用户，不要反复重试检索。
- 一旦你从工具返回中获得了足够回答问题的信息，必须立即回答，不要再重复调用工具。
"""


# 前端「快速 / 深度思考」开关对应的模型。
# deepseek-reasoner 会把思考过程放进 reasoning_content 流式吐出来；
# 实测它同样支持 function calling（工具照样能调），所以可以随时切。
MODEL_CHOICES = {
    "chat": "deepseek-chat",
    "reasoner": "deepseek-reasoner",
}
DEFAULT_MODEL_KEY = "chat"


def resolve_model(mode: str | None) -> tuple[str, str]:
    """把前端给的模式名/模型名解析成 (key, 模型名)，认不出来的一律走默认档。"""
    wanted = (mode or "").strip().lower()
    if wanted in MODEL_CHOICES:
        return wanted, MODEL_CHOICES[wanted]
    for key, name in MODEL_CHOICES.items():
        if wanted == name:
            return key, name
    return DEFAULT_MODEL_KEY, MODEL_CHOICES[DEFAULT_MODEL_KEY]


def get_llm(model: str | None = None):
    return ChatDeepSeek(
        model=model or MODEL_CHOICES[DEFAULT_MODEL_KEY],
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0.1,
    )


def _server_params() -> StdioServerParameters:
    """构造 MCP 子进程启动参数"""
    server_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return StdioServerParameters(
        command=sys.executable,
        args=[server_script],
        env=env,
        # 关键：显式指定工作目录，否则子进程继承父进程 cwd，
        # rag.py 里的相对路径会漂移到别处（踩过一次）
        cwd=os.path.dirname(server_script),
    )


def _oneline(text, limit: int = 160) -> str:
    """压成一行并截断，方便在终端里看"""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[:limit] + f"…（共 {len(flat)} 字符）"


# 工具返回发给前端的最大长度：前端默认折叠，展开能看到全文，
# 但留个上限免得一次 SSE 帧大到离谱（知识库全量检索可能上万字）。
_TOOL_TEXT_MAX = 20000


def _unwrap_args(args):
    """单参数工具会被 langchain 包成 {"__arg1": ...} 或 {"arg1": ...}，拆开展示"""
    if isinstance(args, dict) and len(args) == 1:
        only = next(iter(args.values()))
        if isinstance(only, str):
            return only
    return args


def _describe_message(m) -> list[str]:
    """把一条消息翻译成人类可读的 trace 行"""
    out = []
    if isinstance(m, AIMessage):
        for tc in (m.tool_calls or []):
            out.append(f"│  → 决定调用 {tc.get('name')}  参数: {_unwrap_args(tc.get('args', {}))}")
        if m.content:
            out.append(f"│  → 模型文本: {_oneline(m.content)}")
    elif isinstance(m, ToolMessage):
        out.append(f"│  ← {m.name} 返回: {_oneline(m.content)}")
    return out


# 流式输出时给前端看的步骤文案
_TOOL_LABELS = {
    "rag_search": "正在检索知识库…",
    "calculator": "正在计算…",
    "reload_knowledge_base": "正在重新打开知识库…",
}


def _tool_label(name) -> str:
    return _TOOL_LABELS.get(name or "", f"正在调用 {name or '工具'}…")


def _tool_text(output) -> str:
    """工具返回可能是 ToolMessage，也可能已经是字符串"""
    if isinstance(output, ToolMessage):
        return str(output.content)
    if isinstance(output, str):
        return output
    return str(getattr(output, "content", output))


# 重建上下文时最多回溯多少条历史消息（10 轮问答）
HISTORY_SEED_LIMIT = 20


class KnowledgeAgent:
    """常驻的知识库 Agent。

    MCP 子进程和 LangGraph Agent 只创建一次，之后所有提问复用同一个实例，
    不必每问一句就重启子进程、重载一遍 embedding 模型。
    对话历史由 MemorySaver 按 thread_id 隔离，不同会话互不干扰；
    进程重启后 MemorySaver 是空的，这时用 history.py 里落盘的历史补回上下文。
    """

    def __init__(self):
        self.memory = MemorySaver()
        self._session: ClientSession | None = None
        self._agent = None
        # 非默认模型的图（「深度思考」档），按 key 缓存，随会话一起失效
        self._extra_agents: dict[str, object] = {}
        self._start_lock = asyncio.Lock()
        # 提问和「重新打开知识库」共用，避免同时打同一条 MCP stdio
        self._call_lock = asyncio.Lock()
        # 持有 MCP 会话的常驻任务（见 start() 的说明）
        self._owner_task: asyncio.Task | None = None
        self._stop_event: asyncio.Event | None = None
        self._start_error: BaseException | None = None
        # 最近一次 trace=True 的执行轨迹（调试用）
        self.last_trace: list[str] = []
        # 最近一次执行的完整 messages 列表（loop 的原始记录）
        self.last_messages: list = []

    @property
    def is_started(self) -> bool:
        return self._agent is not None

    async def start(self) -> "KnowledgeAgent":
        """拉起 MCP 子进程并构建 Agent（幂等，并发调用只会真正启动一次）。

        为什么要把生命周期交给一个常驻任务：mcp 的 stdio_client / ClientSession
        内部是 anyio 的任务组 + cancel scope，而 anyio 禁止「在 A 任务里进、
        在 B 任务里出」。以前是在请求任务里懒启动、由 lifespan 的关闭任务收尾，
        于是每次 Ctrl+C 都会炸
        `RuntimeError: Attempted to exit cancel scope in a different task as it was entered in`。
        现在无论谁调用 start()/aclose()，真正的进和出都发生在 _owner 这一个任务里。
        """
        async with self._start_lock:
            if self._is_alive():
                return self
            await self._close_unlocked()  # 清掉上次留下的残骸

            ready = asyncio.Event()
            stop = asyncio.Event()
            self._start_error = None
            self._stop_event = stop
            self._owner_task = asyncio.create_task(
                self._run_session(ready, stop), name="mcp-session-owner"
            )

            await ready.wait()
            if self._agent is None:
                error = self._start_error
                await self._close_unlocked()
                raise RuntimeError(f"MCP 子进程启动失败：{error!r}") from error
            return self

    def _is_alive(self) -> bool:
        return (
            self._agent is not None
            and self._owner_task is not None
            and not self._owner_task.done()
        )

    async def _run_session(self, ready: asyncio.Event, stop: asyncio.Event) -> None:
        """常驻任务：MCP 会话在这里进，也在这里出。"""
        try:
            async with AsyncExitStack() as stack:
                read, write = await stack.enter_async_context(
                    stdio_client(_server_params())
                )
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                self._session = session
                self._agent = self._build_agent(session)
                ready.set()
                await stop.wait()  # 一直挂着，直到 aclose() 让它收工
        except BaseException as exc:  # noqa: BLE001 - 任务里不能漏异常
            name = type(exc).__name__
            if not ready.is_set():
                self._start_error = exc
                sys.stderr.write(f"[ERROR] 启动 MCP 子进程失败：{name}: {exc}\n")
            else:
                sys.stderr.write(f"[WARNING] MCP 子进程会话异常退出：{name}: {exc}\n")
        finally:
            if not ready.is_set():
                ready.set()
            # 只在「自己还是当前 owner」时清状态，别把后来重启的会话抹掉
            if self._owner_task is asyncio.current_task():
                self._owner_task = None
                self._session = None
                self._agent = None
                self._extra_agents.clear()

    async def _close_unlocked(self) -> None:
        """让 owner 任务收工并等它结束（调用方保证没有并发 start）。"""
        task, self._owner_task = self._owner_task, None
        stop, self._stop_event = self._stop_event, None
        self._session = None
        self._agent = None
        self._extra_agents.clear()
        if task is None:
            return
        if stop is not None:
            stop.set()
        try:
            await asyncio.wait_for(task, timeout=10)
        except asyncio.TimeoutError:
            # wait_for 超时会自己 cancel，这里只提示，绝不往上抛：
            # 关闭阶段的异常会让 uvicorn 打印一大坨 traceback 后非正常退出
            sys.stderr.write("[WARNING] MCP 子进程收尾超时，已强制取消\n")
        except asyncio.CancelledError:
            sys.stderr.write("[WARNING] MCP 子进程收尾被取消\n")
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(
                f"[WARNING] MCP 子进程收尾出错（已忽略）：{type(exc).__name__}: {exc}\n"
            )

    def _build_agent(self, session: ClientSession, model_key: str = DEFAULT_MODEL_KEY):
        async def rag_search_remote(query: str) -> str:
            result = await session.call_tool("rag_search", arguments={"query_or_json": query})
            return result.content[0].text if result.content else "无返回内容"

        async def calculator_remote(expression: str) -> str:
            result = await session.call_tool("calculator", arguments={"expr_or_json": expression})
            return result.content[0].text if result.content else "无返回内容"

        tools = [
            Tool(
                name="rag_search",
                description=(
                    "检索知识库，输入查询字符串。"
                    "问「最高/最大/最低/最小/所有/求和/平均」时，查询字符串本身必须带上这些词"
                    "（例如「所有 PSNR」，不要只写「PSNR」），这样一次检索就会返回全部相关条目，不要再重复检索。"
                    "返回文本含 [来源: 文件名 第N页]，回答时要保留来源。"
                ),
                func=lambda x: "同步调用不支持，请使用异步",
                coroutine=rag_search_remote,
            ),
            Tool(
                name="calculator",
                description=(
                    "计算数学表达式，输入表达式字符串。"
                    "支持 + - * / // % ** 和括号，以及 max/min/sum/mean/avg/abs/round/pow 函数。"
                    "例：max(23.38, 26.47, 26.02)、sum([1.5, 2.5])、mean([1.5, 2.5])"
                ),
                func=lambda x: "同步调用不支持，请使用异步",
                coroutine=calculator_remote,
            ),
        ]
        return create_agent(
            get_llm(MODEL_CHOICES.get(model_key, MODEL_CHOICES[DEFAULT_MODEL_KEY])),
            tools,
            checkpointer=self.memory,
            system_prompt=SYSTEM_PROMPT,
        )

    async def _graph_for(self, model: str | None):
        """取这次提问要用的 Agent 图；「深度思考」档第一次用到时才建。"""
        model_key, _name = resolve_model(model)
        if self._agent is None or self._session is None:
            await self.start()
        if model_key == DEFAULT_MODEL_KEY:
            return self._agent
        graph = self._extra_agents.get(model_key)
        if graph is None:
            graph = self._build_agent(self._session, model_key)
            self._extra_agents[model_key] = graph
        return graph

    async def _reload_unlocked(self) -> str:
        result = await self._session.call_tool(
            "reload_knowledge_base", arguments={}
        )
        if getattr(result, "isError", False):
            detail = result.content[0].text if result.content else "重新打开知识库失败"
            raise RuntimeError(detail)
        return result.content[0].text if result.content else "知识库已重新打开"

    async def reload_knowledge(self) -> str:
        """让 MCP 子进程关掉向量库连接，下次检索重新打开。"""
        if self._session is None:
            return "知识库服务未启动"
        async with self._call_lock:
            return await self._reload_unlocked()

    async def update_knowledge(self, update_fn):
        """先放开子进程的库文件，再写入，最后让子进程丢掉旧句柄。

        整段都占着和提问同一把锁，避免写到一半时检索又把库打开。
        写入本身丢到线程里跑：扫描件 OCR 一页要秒级，不能把事件循环卡住。
        返回值就是 update_fn 的返回值（调用方可能要拿删除条数之类的结果）。
        """
        if self._session is None:
            return await asyncio.to_thread(update_fn)
        async with self._call_lock:
            await self._reload_unlocked()
            try:
                return await asyncio.to_thread(update_fn)
            finally:
                if self._session is not None:
                    await self._reload_unlocked()

    async def _seed_from_history(
        self, thread_id: str, config: dict, graph=None
    ) -> list:
        """记忆里没有这条线程时，用落盘的历史补回上下文。

        MemorySaver 是纯内存的，服务一重启上下文就没了；而前端的 session_id
        还存在 localStorage 里，用户点开旧会话接着问，模型会一脸茫然。
        这里在开跑之前检查一下 checkpoint，空的话就把最近若干轮historical
        消息垫在最前面，让模型接得上。
        """
        try:
            snapshot = await (graph or self._agent).aget_state(config)
            existing = (getattr(snapshot, "values", None) or {}).get("messages")
        except Exception:
            existing = None
        if existing:
            return []  # 记忆还在，什么都不用补

        try:
            rows = recent_messages(thread_id, HISTORY_SEED_LIMIT)
        except Exception as exc:  # noqa: BLE001 - 历史读不到不该挡住提问
            sys.stderr.write(f"[WARNING] 读取历史会话失败：{type(exc).__name__}: {exc}\n")
            return []

        seed: list = []
        for row in rows:
            content = (row.get("content") or "").strip()
            if not content:
                continue
            if row.get("role") == "user":
                seed.append(HumanMessage(content=content))
            elif row.get("role") == "assistant":
                seed.append(AIMessage(content=content))
        # 最后一条必须是 AI 的：否则紧接着又跟一条用户提问，
        # 会出现两条连续 user 消息，有些接口会直接报错
        if seed and isinstance(seed[-1], HumanMessage):
            seed.pop()
        if seed:
            sys.stderr.write(
                f"[INFO] 会话 {thread_id}：记忆为空，从历史记录补回 {len(seed)} 条上下文\n"
            )
        return seed

    async def _input_payload(
        self, user_input: str, thread_id: str, config: dict, graph=None
    ) -> dict:
        """这一轮的输入：通常就是一句提问；需要时前面垫上历史上下文。"""
        seed = await self._seed_from_history(thread_id, config, graph)
        return {"messages": [*seed, HumanMessage(content=user_input)]}

    async def aclose(self) -> None:
        """关闭 MCP 子进程，释放资源。

        任何任务里调用都安全：真正退出 cancel scope 的是 owner 任务自己。
        而且这里保证不抛异常——以前这一步会顺着 FastAPI 的 shutdown 抛出去，
        演变成 `Application shutdown failed. Exiting.`。
        """
        async with self._start_lock:
            await self._close_unlocked()

    async def forget_thread(self, thread_id: str) -> bool:
        """删掉某条会话在 MemorySaver 里的记忆（没有这条线程也不报错）。"""
        for name in ("adelete_thread", "delete_thread"):
            fn = getattr(self.memory, name, None)
            if not callable(fn):
                continue
            try:
                result = fn(thread_id)
                if inspect.isawaitable(result):
                    await result
                return True
            except Exception as exc:  # noqa: BLE001
                sys.stderr.write(
                    f"[WARNING] {name} 删除会话 {thread_id} 失败：{type(exc).__name__}: {exc}\n"
                )
        # 老版本 langgraph 没有 delete_thread，直接清内存里的 key
        # （InMemorySaver 的 storage/writes 都是 {(thread_id, ns, id): ...}）
        storage = getattr(self.memory, "storage", None)
        if isinstance(storage, dict):
            for key in [k for k in storage if k and k[0] == thread_id]:
                storage.pop(key, None)
            writes = getattr(self.memory, "writes", None)
            if isinstance(writes, dict):
                for key in [k for k in writes if k and k[0] == thread_id]:
                    writes.pop(key, None)
            return True
        return False

    async def aask(
        self,
        user_input: str,
        thread_id: str = "default",
        trace: bool = False,
        model: str | None = None,
    ) -> str:
        """问一次，返回模型的纯回答文本（不含问题前缀）。

        trace=True 时会用 stream_mode="updates" 逐步流式执行，
        把 model / tools 每一轮的节点、工具调用、工具返回全部打印出来，
        同时记录到 self.last_trace。
        model 传 "reasoner"/"deepseek-reasoner" 就走深度思考档。
        与重新打开知识库共用一把锁，避免上传和提问同时打 MCP 子进程。
        """
        async with self._call_lock:
            return await self._aask_unlocked(user_input, thread_id, trace, model)

    async def _aask_unlocked(
        self,
        user_input: str,
        thread_id: str = "default",
        trace: bool = False,
        model: str | None = None,
    ) -> str:
        graph = await self._graph_for(model)
        config = {"configurable": {"thread_id": thread_id}}
        self.last_trace = []
        payload = await self._input_payload(user_input, thread_id, config, graph)

        if not trace:
            result = await graph.ainvoke(payload, config=config)
            self.last_messages = list(result["messages"])
            return result["messages"][-1].content

        # ---- 带轨迹的逐步执行 ----
        self._emit(f"┏━━ 开始执行   thread_id={thread_id}")
        self._emit(f"┃ 输入: {user_input}")

        step = 0
        answer = ""
        collected: list = []
        async for chunk in graph.astream(
            payload,
            config=config,
            stream_mode="updates",
        ):
            step += 1
            for node_name, payload in chunk.items():
                self._emit(f"┠─ 第 {step} 步 · 节点 [{node_name}]")
                msgs = payload.get("messages", []) if isinstance(payload, dict) else []
                for m in msgs:
                    collected.append(m)
                    for line in _describe_message(m):
                        self._emit(line)
                    # 最后一条「不带工具调用的 AIMessage」就是最终回答
                    if isinstance(m, AIMessage) and not m.tool_calls and m.content:
                        answer = m.content

        self._emit(f"┗━ 共 {step} 步，loop 结束（model ⇄ tools）")
        self.last_messages = collected
        if not answer:  # 兜底：万一最后一步没留下纯文本
            for m in reversed(collected):
                if isinstance(m, AIMessage) and m.content:
                    answer = m.content
                    break
        return answer

    async def astream_answer(
        self, user_input: str, thread_id: str = "default", model: str | None = None
    ):
        """流式执行一次提问，逐条产出给前端的事件字典。

        事件类型：
          step      中间步骤（正在检索 / 正在计算）
          tool      工具返回内容（全文，前端可折叠查看）
          reasoning 模型的思考内容（deepseek-reasoner 档才有）
          token     回答正文片段
          reset     之前吐出的正文作废：模型那是在自言自语，真正要做的是调工具

        model 传 "reasoner" 走深度思考档，用户能在界面上看到完整思考过程。
        整个迭代都占着 _call_lock，和上传、重开知识库互斥；
        客户端中途断开时生成器被关掉，锁会在 async with 退出时自动释放。
        """
        graph = await self._graph_for(model)

        config = {"configurable": {"thread_id": thread_id}}
        collected: list = []
        emitted = False

        async with self._call_lock:
            payload = await self._input_payload(user_input, thread_id, config, graph)
            async for event in graph.astream_events(
                payload,
                config=config,
                version="v2",
            ):
                kind = event.get("event")

                if kind == "on_chat_model_stream":
                    chunk = event["data"].get("chunk")
                    if chunk is None:
                        continue
                    thought = (getattr(chunk, "additional_kwargs", None) or {}).get(
                        "reasoning_content"
                    )
                    if thought:
                        yield {"type": "reasoning", "text": thought}
                    content = chunk.content if isinstance(chunk.content, str) else ""
                    # 工具调用的参数也是流式吐出来的，不能当回答给用户看
                    if content and not getattr(chunk, "tool_call_chunks", None):
                        emitted = True
                        yield {"type": "token", "text": content}

                elif kind == "on_chat_model_end":
                    output = event["data"].get("output")
                    if isinstance(output, AIMessage):
                        collected.append(output)

                elif kind == "on_tool_start":
                    if emitted:
                        yield {"type": "reset"}
                        emitted = False
                    yield {"type": "step", "text": _tool_label(event.get("name"))}

                elif kind == "on_tool_end":
                    output = event["data"].get("output")
                    if isinstance(output, ToolMessage):
                        collected.append(output)
                    # 全文给前端：默认折叠起来，用户想展开看就展开
                    yield {
                        "type": "tool",
                        "name": event.get("name"),
                        "text": _tool_text(output)[:_TOOL_TEXT_MAX],
                    }

        if collected:
            self.last_messages = collected

    def _emit(self, line: str) -> None:
        """trace 行：既打印也留档"""
        print(line)
        self.last_trace.append(line)


# ---------- 进程级共享实例 ----------
_shared_agent: KnowledgeAgent | None = None
_shared_lock: asyncio.Lock | None = None


def _get_shared_lock() -> asyncio.Lock:
    global _shared_lock
    if _shared_lock is None:
        _shared_lock = asyncio.Lock()
    return _shared_lock


async def get_shared_agent() -> KnowledgeAgent:
    """拿到进程内共享的 Agent（不存在则启动）"""
    global _shared_agent
    async with _get_shared_lock():
        if _shared_agent is None:
            _shared_agent = KnowledgeAgent()
        await _shared_agent.start()
        return _shared_agent


async def shutdown_agent() -> None:
    """进程退出前调用，关掉 MCP 子进程"""
    global _shared_agent
    if _shared_agent is not None:
        await _shared_agent.aclose()
        _shared_agent = None


async def reload_shared_knowledge() -> None:
    """若检索子进程已在跑，让它丢掉旧的向量库句柄。未启动则什么都不做。"""
    agent = _shared_agent
    if agent is None or not agent.is_started:
        return
    try:
        await agent.reload_knowledge()
    except _DEAD_SESSION_ERRORS as exc:
        sys.stderr.write(
            f"[WARNING] MCP 子进程连接已断开（{type(exc).__name__}），已关闭，下次提问会重新打开知识库。\n"
        )
        await agent.aclose()
    except Exception as exc:
        sys.stderr.write(f"[WARNING] 通知知识库重新打开失败：{type(exc).__name__}: {exc}\n")


async def update_shared_knowledge(update_fn):
    """在不和提问抢同一个库文件的前提下执行写入。子进程没启动就直接写。

    写入统一走线程：解析 + OCR 是阻塞活，放主线程会把事件循环卡死。
    返回值原样透传 update_fn 的结果。
    """
    agent = _shared_agent
    if agent is None or not agent.is_started:
        return await asyncio.to_thread(update_fn)
    try:
        return await agent.update_knowledge(update_fn)
    except _DEAD_SESSION_ERRORS as exc:
        sys.stderr.write(
            f"[WARNING] MCP 子进程连接已断开（{type(exc).__name__}），已关闭后继续写入。\n"
        )
        await agent.aclose()
        return await asyncio.to_thread(update_fn)


async def forget_shared_thread(thread_id: str) -> bool:
    """把某条会话的记忆从 MemorySaver 里抹掉（用于「删除会话」）。"""
    agent = _shared_agent
    if agent is None or not agent.is_started:
        return False
    try:
        return await agent.forget_thread(thread_id)
    except Exception as exc:  # noqa: BLE001 - 删记忆失败不该影响删除历史
        sys.stderr.write(
            f"[WARNING] 清除会话记忆失败：{type(exc).__name__}: {exc}\n"
        )
        return False


async def ask_agent(
    user_input: str,
    thread_id: str = "default",
    trace: bool = False,
    model: str | None = None,
) -> str:
    """问一次（复用常驻 Agent），返回模型的纯回答。

    trace=True 时会在终端打印完整的 model ⇄ tools 循环过程。
    model="reasoner" 走深度思考档。
    MCP 子进程意外挂掉时自动重启一次再重试。
    """
    agent = await get_shared_agent()
    try:
        answer = await agent.aask(user_input, thread_id, trace=trace, model=model)
    except _DEAD_SESSION_ERRORS as e:
        sys.stderr.write(f"[WARNING] MCP 子进程连接已断开（{type(e).__name__}），正在重启...\n")
        await agent.aclose()
        await agent.start()
        answer = await agent.aask(user_input, thread_id, trace=trace, model=model)
    # 只返回回答本身：以前这里会拼上 "{问题}：\n"，Web 界面的助手气泡
    # 会把用户刚问过的问题又重复一遍。CLI 那边自己拼前缀（见 run_cli）。
    return answer


HELP_TEXT = """
可用命令：
  /trace        开关轨迹模式（开启后每次提问都会打印 model ⇄ tools 循环全过程）
  /graph        打印 Agent 的图结构
  /steps        打印上一条提问的完整 messages 轨迹（loop 的原始记录）
  /help         显示本帮助
  exit          退出
""".strip()


async def run_cli():
    trace = "--trace" in sys.argv or "-t" in sys.argv
    print("=== 知识库问答助手已启动（LangGraph版本），输入 exit 退出 ===")
    print("[启动中] 正在拉起知识库工具服务并加载模型，首次约 10~20 秒...")
    agent = await get_shared_agent()
    print(f"[就绪] 可以开始提问了。输入 /help 查看命令（当前轨迹模式：{'开' if trace else '关'}）")

    thread_id = "default"
    last_messages = []
    try:
        while True:
            try:
                user_input = input("\n请提问：")
            except EOFError:
                break

            cmd = user_input.strip().lower()
            if cmd == "exit":
                break
            if not cmd:
                continue

            # ---- 内置命令 ----
            if cmd.startswith("/"):
                if cmd in ("/trace", "/trace on", "/trace off"):
                    if cmd == "/trace off":
                        trace = False
                    elif cmd == "/trace on":
                        trace = True
                    else:
                        trace = not trace
                    print(f"[轨迹模式] {'已开启，下次提问会打印完整 loop' if trace else '已关闭'}")
                elif cmd == "/graph":
                    g = agent._agent.get_graph()
                    print("节点:", list(g.nodes.keys()))
                    for e in g.edges:
                        print(f"  {e.source} -> {e.target}" + (f"  [{e.data}]" if e.data else ""))
                elif cmd == "/steps":
                    if not last_messages:
                        print("[提示] 还没有提问记录")
                    else:
                        print(f"上一条提问的 messages 轨迹（共 {len(last_messages)} 条）：")
                        for i, m in enumerate(last_messages):
                            extra = ""
                            if isinstance(m, AIMessage) and m.tool_calls:
                                extra = "  tool_calls=" + str([t["name"] for t in m.tool_calls])
                            elif isinstance(m, ToolMessage):
                                extra = f"  来自 {m.name}，{len(str(m.content))} 字符"
                            print(f"  [{i}] {type(m).__name__}{extra}")
                elif cmd in ("/help", "/?"):
                    print(HELP_TEXT)
                else:
                    print(f"[未知命令] {user_input}\n\n{HELP_TEXT}")
                continue

            # ---- 正常提问 ----
            try:
                answer = await agent.aask(user_input, thread_id, trace=trace)
            except Exception as e:
                print(f"\n[出错] {type(e).__name__}: {e}")
                continue
            last_messages = agent.last_messages
            print("\n回答：", answer)
    finally:
        await agent.aclose()


if __name__ == "__main__":
    try:
        asyncio.run(run_cli())
    except KeyboardInterrupt:
        print("\n程序已安全退出")
