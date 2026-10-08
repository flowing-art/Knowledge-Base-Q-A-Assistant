from fastmcp import FastMCP
from rag import reset_vector_db, search_knowledge
import ast
import json
import operator

mcp = FastMCP("知识库工具服务")


# ---------- 安全表达式求值 ----------
# 不用 eval + 字符白名单，改成 AST 白名单求值：
# 既能支持 max/min/sum 这类聚合函数，又不会被注入。
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _sum(*args):
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        return sum(args[0])
    return sum(args)


def _mean(*args):
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        values = list(args[0])
    else:
        values = list(args)
    if not values:
        raise ValueError("mean() 至少需要一个数")
    return sum(values) / len(values)


_FUNCS = {
    "max": max,
    "min": min,
    "sum": _sum,
    "mean": _mean,
    "avg": _mean,
    "abs": abs,
    "round": round,
    "pow": pow,
}
SUPPORTED_HINT = (
    "支持 + - * / // % ** 、括号、数字列表，以及 max / min / sum / mean / avg / abs / round / pow 函数"
)


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError("只允许数字常量")
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的运算符 {type(node.op).__name__}")
        return op(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARYOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"不支持的一元运算符 {type(node.op).__name__}")
        return op(_safe_eval(node.operand))
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_safe_eval(e) for e in node.elts]
    if isinstance(node, ast.Call):
        name = node.func.id if isinstance(node.func, ast.Name) else None
        if name not in _FUNCS:
            raise ValueError(f"不支持的函数 {name or '<非函数调用>'}")
        if node.keywords:
            raise ValueError("不支持关键字参数")
        return _FUNCS[name](*[_safe_eval(a) for a in node.args])
    raise ValueError(f"不支持的表达式 {type(node).__name__}")


def calculate(expression: str):
    """返回 (是否成功, 结果或错误说明)"""
    expr = expression.strip().replace("，", ",").replace("（", "(").replace("）", ")")
    if not expr:
        raise ValueError("表达式为空")
    tree = ast.parse(expr, mode="eval")
    return _safe_eval(tree)


@mcp.tool()
def rag_search(query_or_json: str) -> str:
    """检索知识库，支持直接传入查询字符串或 JSON 格式 {"query": "..."}"""
    try:
        data = json.loads(query_or_json)
        if isinstance(data, dict) and "query" in data:
            query = data["query"]
        else:
            query = query_or_json
    except json.JSONDecodeError:
        query = query_or_json
    try:
        result = search_knowledge(query)
    except Exception as e:
        # 把异常转成可读文本返回，避免 FastMCP 直接抛整段英文堆栈
        # （LLM 看到英文堆栈容易自己翻译并跑偏）
        return f"[知识库异常] 检索失败：{type(e).__name__}: {e}"
    return f"知识库检索到的相关内容：\n{result}"


@mcp.tool()
def calculator(expr_or_json: str) -> str:
    """计算数学表达式。支持直接传入表达式或 JSON 格式 {"expression": "..."}。

    支持四则运算、括号、** 幂运算，以及 max/min/sum/mean/avg/abs/round/pow 函数。
    例如：max(1, 2, 3)、sum([1.5, 2.5])、mean([1.5, 2.5])、26.47 + 0.939
    """
    try:
        data = json.loads(expr_or_json)
        if isinstance(data, dict) and "expression" in data:
            expression = data["expression"]
        else:
            expression = expr_or_json
    except json.JSONDecodeError:
        expression = expr_or_json
    try:
        result = calculate(expression)
    except Exception as e:
        # 错误信息里带上支持的语法，让模型第一次失败就能自我纠正，
        # 而不是换个空格反复重试同一个表达式
        return f"计算出错：{e}。{SUPPORTED_HINT}。请修正表达式后重新调用，不要原样重试。"
    return f"计算结果：{result}"


@mcp.tool()
def reload_knowledge_base() -> str:
    """关掉本进程里已打开的向量库。下次检索会重新打开，从而读到刚上传的文档。"""
    reset_vector_db()
    return "知识库已重新打开，后续检索会读到最新文档。"


if __name__ == "__main__":
    # 启动 MCP 服务（默认 stdio 模式，或指定 --transport sse）
    # show_banner=False：不打印 FastMCP 那个大 ASCII 框，保持输出干净
    mcp.run(show_banner=False)
