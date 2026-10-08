import os
import sys
import hashlib
import logging
import threading
import time
from dotenv import load_dotenv

# override=True：机器级环境变量可能存着过期值，默认不覆盖会压掉 .env
load_dotenv(override=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 必须用绝对路径！MCP 子进程的工作目录继承自父进程（可能是任意目录），
# 用 "./chroma_db" 会随 cwd 漂移，导致打开到一个空库或新建一个空库。
VECTOR_DB_PATH = os.path.join(BASE_DIR, "chroma_db")
MODEL_CACHE_DIR = os.path.join(BASE_DIR, "models_cache")
# 权重只放工作区的 hub 子目录，避免 HuggingFace / torch 写到用户主目录（通常在 C 盘）。
HUB_CACHE_DIR = os.path.join(MODEL_CACHE_DIR, "hub")
_GENERATION_PATH = os.path.join(BASE_DIR, ".kb_generation")
_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin", "model.bin")
# 图片直接整张 OCR；PDF 的扫描页按这个 DPI 渲染成图再识别
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
OCR_DPI = 200

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 强制离线，并设置缓存目录。放在 import 之前，避免库在导入时就把缓存定到 C 盘。
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = MODEL_CACHE_DIR
os.environ["HF_HUB_CACHE"] = HUB_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = HUB_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HUB_CACHE_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = MODEL_CACHE_DIR
os.environ["TORCH_HOME"] = os.path.join(MODEL_CACHE_DIR, "torch")

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

# 保证同目录的 ocr.py 一定能被找到（MCP 子进程的 cwd 不一定是本目录）
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from ocr import ocr_image, page_needs_ocr

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# ---------- 单例模型管理 ----------
_embeddings = None
_reranker = None
_reranker_checked = False
_vector_db = None
_loaded_generation = None

# chromadb 1.5.x 的 SharedSystemClient 不是线程安全的：
# 它先把「还没 start() 的空壳 System」放进全局字典，再执行 start()，
# 期间另一个线程拿到这个空壳就会报
# AttributeError: 'RustBindingsAPI' object has no attribute 'bindings'，
# 最终被包装成 "Could not connect to tenant default_tenant"。
# Agent 并行发起多个 rag_search 时必然踩到，所以这里统一加锁 + 单例。
# 用 RLock：_get_vector_db() 持锁期间会再调 _get_embeddings()，普通 Lock 会死锁。
_init_lock = threading.RLock()


def _snapshot_dirs(repo_dirname: str) -> list[str]:
    """本地快照目录。旧缓存直接放在 models_cache 下，新缓存放在 models_cache/hub 下。"""
    roots = (
        os.path.join(HUB_CACHE_DIR, repo_dirname),
        os.path.join(MODEL_CACHE_DIR, repo_dirname),
    )
    found = []
    for base in roots:
        snapshots_dir = os.path.join(base, "snapshots")
        if not os.path.isdir(snapshots_dir):
            continue
        for name in sorted(os.listdir(snapshots_dir)):
            snap = os.path.join(snapshots_dir, name)
            if os.path.isdir(snap):
                found.append(snap)
    return found


def _find_local_weights(repo_dirname: str) -> str | None:
    """返回含权重文件的快照；只有 config.json 的半成品目录不算。"""
    for snap in _snapshot_dirs(repo_dirname):
        if any(os.path.isfile(os.path.join(snap, name)) for name in _WEIGHT_FILES):
            return snap
    return None


def _read_generation() -> str:
    try:
        with open(_GENERATION_PATH, encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def _write_generation() -> None:
    with open(_GENERATION_PATH, "w", encoding="utf-8") as handle:
        handle.write(str(time.time_ns()))


def _release_db(db) -> None:
    """停掉 Chroma 的 System，并清掉进程内缓存。

    只删 Python 引用不够：SharedSystemClient 会按路径复用旧连接，
    上传后的新文档就看不见。本进程只用这一个库，清掉整张表是安全的。
    """
    if db is None:
        return
    client = getattr(db, "_client", None)
    system = getattr(client, "_system", None) if client is not None else None
    if system is not None:
        try:
            system.stop()
        except Exception:
            pass
    try:
        from chromadb.api.shared_system_client import SharedSystemClient

        clear = getattr(SharedSystemClient, "clear_system_cache", None)
        if callable(clear):
            clear()
            return
        cache = getattr(SharedSystemClient, "_identifier_to_system", None)
        refcounts = getattr(SharedSystemClient, "_refcount_to_system", None)
        if isinstance(refcounts, dict):
            refcounts.clear()
        if isinstance(cache, dict):
            cache.clear()
    except Exception as exc:
        sys.stderr.write(f"[WARNING] 释放向量库句柄时出错: {exc}\n")


def _release_current_unlocked() -> None:
    global _vector_db, _loaded_generation
    db = _vector_db
    _vector_db = None
    _loaded_generation = None
    _release_db(db)


def reset_vector_db() -> None:
    """关掉本进程的向量库连接。下次检索会重新打开磁盘上的库。"""
    with _init_lock:
        _release_current_unlocked()


def mark_knowledge_updated() -> None:
    """文档落盘后调用：推进代数，并丢掉本进程的旧连接。"""
    with _init_lock:
        _write_generation()
        _release_current_unlocked()


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        with _init_lock:
            if _embeddings is None:
                local_model_path = _find_local_weights("models--BAAI--bge-small-zh-v1.5")
                if local_model_path is None:
                    local_model_path = os.path.join(
                        HUB_CACHE_DIR, "models--BAAI--bge-small-zh-v1.5"
                    )

                sys.stderr.write(f"[DEBUG] 从本地加载 Embedding 模型：{local_model_path}\n")
                _embeddings = HuggingFaceEmbeddings(
                    model_name=local_model_path,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
    return _embeddings


def _get_vector_db():
    """进程内单例的 Chroma 客户端（加锁创建，规避 chromadb 的并发缺陷）。

    `.kb_generation` 变了说明有新文档，关掉旧连接再打开，避免一直读启动时的库。
    """
    global _vector_db, _loaded_generation
    generation = _read_generation()
    if _vector_db is not None and _loaded_generation == generation:
        return _vector_db
    with _init_lock:
        generation = _read_generation()
        if _vector_db is not None and _loaded_generation == generation:
            return _vector_db
        _release_current_unlocked()
        _vector_db = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=_get_embeddings(),
        )
        _loaded_generation = generation
        return _vector_db


def _get_reranker():
    """加载重排序模型（本地无完整权重则跳过，且只提示一次）"""
    global _reranker, _reranker_checked
    if _reranker_checked:
        return _reranker
    with _init_lock:
        if _reranker_checked:
            return _reranker
        # 先看本地缓存里有没有完整权重；没有就直接跳过，别去联网
        # （离线环境下会卡很久再报 502）。hub 和旧的平铺目录都认。
        local_model_path = _find_local_weights("models--BAAI--bge-reranker-base")
        if local_model_path is None:
            snaps = _snapshot_dirs("models--BAAI--bge-reranker-base")
            if snaps:
                sys.stderr.write(
                    "[INFO] 找到 Reranker 缓存但不完整（没有权重文件），跳过重排序。\n"
                    f"       缓存位置：{snaps[0]}\n"
                    "       请运行 ceshi.py，把完整模型下到工作区 models_cache\\hub。\n"
                )
            else:
                sys.stderr.write(
                    "[INFO] 本地无 Reranker 缓存，跳过重排序（仅用向量检索）。\n"
                    "       需要重排序请运行 ceshi.py，模型会下到工作区 models_cache\\hub。\n"
                )
            _reranker = None
            _reranker_checked = True
            return _reranker

        try:
            sys.stderr.write(f"[DEBUG] 加载 Reranker 模型：{local_model_path}\n")
            _reranker = CrossEncoder(local_model_path, device="cpu", max_length=512)
        except Exception as exc:
            sys.stderr.write(f"[WARNING] Reranker 加载失败，将跳过重排序: {exc}\n")
            _reranker = None
        _reranker_checked = True
    return _reranker


# ---------- 文档加载（扫描件走 OCR） ----------
def _load_pdf(file_path: str):
    """逐页加载 PDF：有文字层的页直接用，没有文字层的扫描页渲染成图再 OCR。

    只对「没有文字层」的页做 OCR：既省时间，也避免把本来干净的电子版 PDF
    识别出一堆错字。
    """
    documents = PyPDFLoader(file_path).load()
    try:
        import pymupdf
    except ImportError:  # 包旧版本仍叫 fitz
        try:
            import fitz as pymupdf
        except ImportError:
            sys.stderr.write("[WARNING] 未安装 PyMuPDF，扫描页无法 OCR，只返回文字层内容。\n")
            return documents

    ocr_pages = []
    with pymupdf.open(file_path) as pdf:
        for index in range(pdf.page_count):
            page_text = documents[index].page_content if index < len(documents) else ""
            if not page_needs_ocr(page_text):
                continue
            pixmap = pdf.load_page(index).get_pixmap(dpi=OCR_DPI)
            text = ocr_image(pixmap.tobytes("png"))
            if not text:
                continue
            ocr_pages.append(index)
            metadata = {"source": file_path, "page": index, "ocr": True}
            if index < len(documents):
                documents[index].page_content = text
                documents[index].metadata.update(metadata)
            else:
                from langchain_core.documents import Document

                documents.append(Document(page_content=text, metadata=metadata))
    if ocr_pages:
        pages = "、".join(str(p + 1) for p in ocr_pages)
        sys.stderr.write(f"[INFO] {os.path.basename(file_path)}：OCR 识别了第 {pages} 页\n")
    return documents


def _load_image(file_path: str):
    """图片整张 OCR。"""
    from langchain_core.documents import Document

    with open(file_path, "rb") as handle:
        raw = handle.read()
    text = ocr_image(raw)
    if not text.strip():
        raise ValueError(f"图片 {os.path.basename(file_path)} 没有识别出文字，换张清晰点的图试试")
    return [Document(page_content=text, metadata={"source": file_path, "ocr": True})]


def _load_documents(file_path: str, display_name: str | None = None):
    """按扩展名选择加载方式；PDF 的扫描页和图片会自动走 OCR。

    display_name 是用户上传时的真实文件名，会覆盖 metadata 里的 source。
    /upload 落盘用的是 NamedTemporaryFile，不覆盖的话引用会显示成
    "tmpab12cd.pdf" 这种随机名，用户根本对不上是哪份文件。
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        documents = _load_pdf(file_path)
    elif ext in (".txt", ".md"):
        documents = TextLoader(file_path, encoding="utf-8").load()
    elif ext in IMAGE_EXTS:
        documents = _load_image(file_path)
    else:
        raise ValueError("仅支持PDF/TXT/MD/PNG/JPG/JPEG/BMP/TIFF/WEBP格式文档")

    if display_name:
        for doc in documents:
            doc.metadata["source"] = display_name
    return documents


def _split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", " ", ""],
    )
    return text_splitter.split_documents(documents)


def _doc_ids(name: str, count: int) -> list[str]:
    """按「来源文件名 + 序号」算确定性 id：同一份文件重传会覆盖旧块，而不是叠加。"""
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()[:12]
    return [f"{digest}-{index}" for index in range(count)]


def _delete_source(db, name: str) -> int:
    """删掉某个来源文件已有的所有块，返回实际删掉的条数。"""
    try:
        existing = db.get(where={"source": name})
        ids = list(existing.get("ids") or [])
        if ids:
            db.delete(ids=ids)
        return len(ids)
    except Exception as exc:
        sys.stderr.write(
            f"[WARNING] 清理旧文档 {name} 失败（{type(exc).__name__}: {exc}），本次改为直接追加\n"
        )
        return 0


# ---------- 知识库构建与添加 ----------
def build_knowledge_base(file_path: str, display_name: str | None = None):
    """首次构建知识库（会覆盖已有数据）"""
    name = display_name or os.path.basename(file_path)
    documents = _load_documents(file_path, name)
    split_docs = _split_documents(documents)

    embeddings = _get_embeddings()
    reset_vector_db()
    created = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH,
        ids=_doc_ids(name, len(split_docs)),
    )
    with _init_lock:
        _release_db(created)
    mark_knowledge_updated()
    sys.stderr.write(f"知识库构建完成，共存入 {len(split_docs)} 个文档块\n")


def add_to_knowledge_base(file_path: str, display_name: str | None = None):
    """增量添加文档到已有知识库。

    display_name 是用户上传时的原始文件名：既写进 metadata 的 source
    （引用时显示的文件名），也用来算确定性 id。
    同名文件再次上传时先删掉旧块，避免同一份文件在库里越堆越多。
    """
    name = display_name or os.path.basename(file_path)
    documents = _load_documents(file_path, name)
    split_docs = _split_documents(documents)
    ids = _doc_ids(name, len(split_docs))

    embeddings = _get_embeddings()

    if os.path.exists(VECTOR_DB_PATH) and os.listdir(VECTOR_DB_PATH):
        db = _get_vector_db()
        removed = _delete_source(db, name)
        if removed:
            sys.stderr.write(f"[INFO] {name}：同名文件重新上传，先清掉旧的 {removed} 个文档块\n")
        db.add_documents(split_docs, ids=ids)
    else:
        created = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=VECTOR_DB_PATH,
            ids=ids,
        )
        with _init_lock:
            _release_db(created)
            _release_current_unlocked()
    mark_knowledge_updated()
    sys.stderr.write(f"成功添加 {len(split_docs)} 个文档块到知识库\n")


def list_documents() -> dict:
    """列出知识库里所有已入库的文件（按来源名聚合）。

    本进程直接开库读：和上传/删除走同一条路，不必为了看一眼文件列表
    就把 MCP 子进程拉起来。库不存在或为空就返回空列表，不当成错误。
    """
    if not (os.path.exists(VECTOR_DB_PATH) and os.listdir(VECTOR_DB_PATH)):
        return {"documents": [], "total_chunks": 0}

    db = _get_vector_db()
    data = db.get()
    metadatas = data.get("metadatas") or []
    counter: dict[str, int] = {}
    for meta in metadatas:
        meta = meta or {}
        raw = meta.get("source") or meta.get("file_path") or "未知来源"
        name = os.path.basename(str(raw)) or "未知来源"
        counter[name] = counter.get(name, 0) + 1

    documents = [
        {"name": name, "chunks": chunks}
        for name, chunks in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    return {"documents": documents, "total_chunks": len(metadatas)}


def delete_document(name: str) -> dict:
    """按来源文件名删掉一份文档的全部块，返回删掉的块数。

    调用方必须走 update_shared_knowledge：先让 MCP 子进程放开库文件，
    否则两个进程同时写同一个库会打架（和上传是同一套规矩）。
    """
    safe_name = os.path.basename((name or "").strip())
    if not safe_name:
        raise ValueError("缺少文件名")
    if not (os.path.exists(VECTOR_DB_PATH) and os.listdir(VECTOR_DB_PATH)):
        return {"name": safe_name, "removed": 0}

    db = _get_vector_db()
    removed = _delete_source(db, safe_name)
    if removed:
        mark_knowledge_updated()
        sys.stderr.write(f"[INFO] 已从知识库删除 {safe_name} 的 {removed} 个文档块\n")
    return {"name": safe_name, "removed": removed}


def _format_doc(doc) -> str:
    """带上文件名和页码，方便回答时引用。页码按 PDF 的 0 起始转成第 1 页。

    OCR 出来的内容额外标一个 (OCR)，提醒模型和用户这段是机器识别的，可能有错字。
    """
    meta = doc.metadata or {}
    raw_source = meta.get("source") or meta.get("file_path") or ""
    name = os.path.basename(str(raw_source)) if raw_source else "未知来源"
    suffix = " (OCR)" if meta.get("ocr") else ""
    page = meta.get("page")
    if isinstance(page, bool) or page is None:
        where = f"{name}{suffix}"
    elif isinstance(page, (int, float)):
        where = f"{name} 第{int(page) + 1}页{suffix}"
    else:
        where = f"{name} 第{page}页{suffix}"
    return f"[来源: {where}]\n{doc.page_content}"


# ---------- 检索（带 Rerank） ----------
# 出现这些词，说明用户要的是"全局统计"，向量 top_k 检索必然漏数据，直接全量返回
_AGGREGATE_KEYWORDS = (
    "最高", "最低", "最大", "最小", "所有", "全部", "全部数值", "求和", "之和",
    "汇总", "一共", "多少种", "排名", "对比", "排序",
    "平均", "均值", "平均值", "平均数",
)

# 重排后的最低相关度（bge-reranker 输出 sigmoid 后的 0~1 概率）。
# 低于它的段落直接丢掉：向量检索永远凑够 top_k 段，其中不相关的那些
# 会诱导模型拿无关内容硬答；实测完全无关的段落得分接近 0。
RERANK_MIN_SCORE = 0.1

# 重排全军覆没时的字面兜底。
# bge-reranker-base 对措辞敏感到离谱：同一份软著登记表，
# 「开发者都有谁」得 0.026 被判无关，「全体开发者都是谁」得 0.851 通过——
# 用户换个说法就被回答「知识库里没有」。所以重排说"全都不相关"时先别认输，
# 再用汉字二元组的重合度捞一次：文档里真的出现过这些字，就交回模型自己判断。
LEXICAL_MIN_HITS = 2          # 至少共享 2 个相邻字对
LEXICAL_MIN_RATIO = 0.34      # 或者共享比例够高
LEXICAL_FALLBACK_HITS = 1     # 只共享 1 个字对时，要求重排分数不能是彻底的 0
LEXICAL_FALLBACK_SCORE = 0.01


def _bigrams(text: str) -> list[str]:
    """抽相邻两个字组成的片段，丢掉标点和空白（「开发者都有谁」→ 开发/发者/者都/都有/有谁）。"""
    chars = [c for c in (text or "") if c.isalnum()]
    return [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]


def _lexical_rescue(query: str, scored_docs: list, limit: int = 2) -> list:
    """按字面重合度从"被重排判死"的候选里捞回几段。返回 [(doc, score, hits, ratio)]。"""
    grams = set(_bigrams(query))
    if not grams:
        return []
    found = []
    for doc, score in scored_docs:
        score = float(score)
        text = doc.page_content or ""
        hits = sum(1 for gram in grams if gram in text)
        if hits < LEXICAL_FALLBACK_HITS:
            continue
        ratio = hits / len(grams)
        strong = hits >= LEXICAL_MIN_HITS and ratio >= LEXICAL_MIN_RATIO
        weak = hits >= LEXICAL_FALLBACK_HITS and score >= LEXICAL_FALLBACK_SCORE
        if strong or weak:
            found.append((doc, score, hits, ratio))
    # 先看字面重合得多不多，再看重排分
    found.sort(key=lambda item: (-item[3], -item[2], -item[1]))
    return found[:limit]


def search_knowledge(query: str, top_k: int = 5) -> str:
    """检索知识库，先向量检索获得候选，再使用 Rerank 重排序，返回最相关文档内容

    注意：top_k 默认 5。若查询含聚合类关键词（最高/最大/求和/平均…），
    会自动切换为全量返回，避免"求最大值"时漏掉真正的最大值。
    """
    if not os.path.exists(VECTOR_DB_PATH):
        return (f"[知识库异常] 向量库目录不存在：{VECTOR_DB_PATH}。"
                f"请先调用 build_knowledge_base() 构建知识库。")

    vector_db = _get_vector_db()

    # 0. 空库体检：避免"库是空的"被误报成"没找到相关内容"
    try:
        total = vector_db._collection.count()
    except Exception:
        total = None
    if total == 0:
        return (f"[知识库异常] 向量库 {VECTOR_DB_PATH} 中一条数据都没有（0 条向量）。"
                f"这通常说明检索到了错误的库路径，或知识库还没构建。")

    is_aggregate = any(kw in query for kw in _AGGREGATE_KEYWORDS)

    if is_aggregate:
        # 全量取回（库小的时候最稳），再做 rerank 排序（仅影响顺序，不影响完整性）
        raw = vector_db.get()
        from langchain_core.documents import Document
        documents = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []
        if len(metadatas) < len(documents):
            metadatas = list(metadatas) + [{}] * (len(documents) - len(metadatas))
        docs = [
            Document(page_content=content or "", metadata=meta or {})
            for content, meta in zip(documents, metadatas)
        ]
        sys.stderr.write(f"[DEBUG] 聚合类问题，全量返回 {len(docs)} 个文档块\n")
    else:
        # 1. 获取更多候选（例如 top_k * 3），确保重排后有足够选择
        candidate_k = top_k * 3 if top_k * 3 < 20 else 20  # 限制最大20，避免性能问题
        docs = vector_db.similarity_search(query, k=candidate_k)

    if not docs:
        return "知识库中未找到相关内容。"

    # 2. 如果 Reranker 可用，进行重排序
    reranker = _get_reranker()
    if reranker is not None and len(docs) > 1:
        # 构建 (query, doc.page_content) 对
        pairs = [(query, doc.page_content) for doc in docs]
        scores = reranker.predict(pairs)  # sigmoid 之后的 0~1 相关度
        # 按分数降序排序
        scored_docs = sorted(zip(docs, scores), key=lambda item: float(item[1]), reverse=True)
        if is_aggregate:
            # 聚合类问题既不过滤也不截断，否则又会漏掉真正的最大值
            selected = [doc for doc, _score in scored_docs]
        else:
            # 低于阈值的一律丢掉，宁可回答"没找到"也不拿无关段落硬答
            kept = [(doc, float(score)) for doc, score in scored_docs if float(score) >= RERANK_MIN_SCORE]
            if not kept:
                best = float(scored_docs[0][1])
                rescued = _lexical_rescue(query, scored_docs)
                if rescued:
                    hits_desc = "、".join(f"{h} 个字对" for _d, _s, h, _r in rescued)
                    sys.stderr.write(
                        f"[DEBUG] {len(scored_docs)} 段全部低于相关性阈值 {RERANK_MIN_SCORE}"
                        f"（最高 {best:.3f}），字面兜底捞回 {len(rescued)} 段（{hits_desc}）\n"
                    )
                    note = (
                        f"[提示] 重排模型认为没有高相关段落（最高分 {best:.3f} < {RERANK_MIN_SCORE}），"
                        f"下面是按字面关键词（{hits_desc}）找回的低置信度结果："
                        f"只有确实能回答问题时才用，答不上来就如实告诉用户没找到。\n\n"
                    )
                    return note + "\n\n---\n\n".join(
                        _format_doc(doc) for doc, _s, _h, _r in rescued
                    )
                sys.stderr.write(
                    f"[DEBUG] {len(scored_docs)} 段全部低于相关性阈值 "
                    f"{RERANK_MIN_SCORE}（最高只有 {best:.3f}），字面兜底也没捞到\n"
                )
                return (f"知识库中没有与「{query}」相关的内容。"
                        f"（最相关的一段相关度只有 {best:.3f}，低于 {RERANK_MIN_SCORE} 的判定阈值）")
            dropped = len(scored_docs) - len(kept)
            if dropped:
                sys.stderr.write(
                    f"[DEBUG] 丢弃 {dropped} 段低于相关性阈值 {RERANK_MIN_SCORE} 的结果\n"
                )
            selected = [doc for doc, _score in kept[:top_k]]
    else:
        # 无 Reranker 时，直接取前 top_k
        selected = docs[:top_k]

    # 3. 拼接返回，每段带上来源
    return "\n\n---\n\n".join(_format_doc(doc) for doc in selected)


# ---------- 测试入口 ----------
if __name__ == "__main__":
    # 测试检索
    res = search_knowledge("科研类用印审批单的申请人是谁")
    sys.stderr.write("检索结果：\n" + res)
