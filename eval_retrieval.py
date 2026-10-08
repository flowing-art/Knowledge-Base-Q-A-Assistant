"""对当前知识库跑一组固定问法，写出重排分和最终判定。

用法（在项目目录、Agent 环境）：

    python eval_retrieval.py

结果写到 eval/retrieval.md。会加载本地向量库和重排模型，不访问外网。
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import rag  # noqa: E402

# 期望只用来对照，表里记的是这次实测。
CASES = [
    ("开发者都有谁", "兜底"),
    ("开发者都有谁？", "兜底"),
    ("开发人员都有谁", "兜底"),
    ("全体开发者都是谁", "通过"),
    ("全体开发者都有谁", "通过"),
    ("这份软著的第一开发者是谁", "通过"),
    ("IGDFormer 的 PSNR", "通过"),
    ("今天天气怎么样", "拒绝"),
]


def _top_score(query: str):
    """返回 (最高分, 候选数)。库为空时分数是 None。"""
    if not os.path.exists(rag.VECTOR_DB_PATH):
        return None, 0
    db = rag._get_vector_db()
    docs = db.similarity_search(query, k=20)
    reranker = rag._get_reranker()
    if reranker is None or not docs:
        return None, len(docs)
    pairs = [(query, doc.page_content) for doc in docs]
    scores = [float(score) for score in reranker.predict(pairs)]
    return max(scores), len(docs)


def _verdict(answer: str) -> str:
    if answer.startswith("[提示]"):
        return "兜底"
    if answer.startswith("知识库中没有") or answer.startswith("知识库中未找到"):
        return "拒绝"
    return "通过"


def main() -> None:
    lines = [
        "# 检索评测",
        "",
        "同一套本地知识库（`test.txt` 的论文指标 + 扫描版软著登记表）。",
        "重排模型是 `BAAI/bge-reranker-base`，通过线是 `0.1`。",
        "低于这条线时，汉字二元组还能对上原文，就记为「兜底」，否则「拒绝」。",
        "",
        "| 问法 | 最高重排分 | 判定 | 原先预期 |",
        "|---|---:|---|---|",
    ]
    mismatches = []
    for query, expected in CASES:
        score, _count = _top_score(query)
        answer = rag.search_knowledge(query)
        verdict = _verdict(answer)
        score_text = "—" if score is None else f"{score:.3f}"
        lines.append(f"| {query} | {score_text} | {verdict} | {expected} |")
        if verdict != expected:
            mismatches.append(f"{query}：预期 {expected}，实际 {verdict}（{score_text}）")
        print(f"{verdict}\t{score_text}\t{query}")

    lines.extend([
        "",
        "「开发者都有谁」和「全体开发者都是谁」问的是同一份表。",
        "只改两个字，重排分会从远低于 0.1 跳到远高于 0.1。",
        "兜底只在原文里真有这些字时交回模型，无关问题仍拒绝。",
        "",
    ])
    if mismatches:
        lines.append("和预期不一致：")
        lines.extend(f"- {item}" for item in mismatches)
        lines.append("")

    out_dir = os.path.join(BASE_DIR, "eval")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "retrieval.md")
    with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
