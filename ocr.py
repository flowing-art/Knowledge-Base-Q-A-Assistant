"""扫描件 OCR。

用 RapidOCR（PP-OCRv4 + onnxruntime 推理）：模型随 wheel 一起安装，
运行时不需要联网下载，也不会往 C 盘写缓存。

对外两个函数：
  page_needs_ocr(text)   —— 判断 PDF 某页是不是「没有文字层」的扫描页
  ocr_image(image_bytes) —— 识别图片，返回按阅读顺序拼接的文本
"""

import os
import re
import sys
import threading

# 低于这个置信度的行直接丢掉，通常是噪点或印章边框
MIN_SCORE = 0.5
# 有效字符少于这个数，就认为这一页没有文字层
MIN_TEXT_CHARS = 50

_MEANINGFUL = re.compile(r"[0-9A-Za-z\u4e00-\u9fff]")

# 模型统一放工作区（models_cache/rapidocr），不留在 conda 环境里
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_cache", "rapidocr")
_MODEL_FILES = {
    "det_model_path": "ch_PP-OCRv4_det_infer.onnx",
    "rec_model_path": "ch_PP-OCRv4_rec_infer.onnx",
    "cls_model_path": "ch_ppocr_mobile_v2.0_cls_infer.onnx",
}

# 引擎是重对象（要加载 3 个 onnx 模型），按需初始化后复用
_engine = None
_engine_lock = threading.RLock()
_engine_failed = False


def page_needs_ocr(text: str) -> bool:
    """文字层里几乎没有有效字符，就判定为扫描页。

    扫描页经 PyPDFLoader 抽出来通常是空串，或只有零星几个页眉字符。
    """
    return len(_MEANINGFUL.findall(text or "")) < MIN_TEXT_CHARS


def _get_engine():
    global _engine, _engine_failed
    with _engine_lock:
        if _engine is None and not _engine_failed:
            try:
                from rapidocr_onnxruntime import RapidOCR

                paths = {
                    key: os.path.join(MODEL_DIR, name)
                    for key, name in _MODEL_FILES.items()
                }
                missing = [p for p in paths.values() if not os.path.isfile(p)]
                if missing:
                    sys.stderr.write(
                        f"[WARNING] 工作区缺少 OCR 模型 {missing}，改用包内默认路径。\n"
                    )
                    paths = {}
                _engine = RapidOCR(**paths)
            except Exception as exc:
                _engine_failed = True
                sys.stderr.write(f"[WARNING] OCR 引擎加载失败，扫描页将跳过: {exc}\n")
        return _engine


def ocr_image(image: bytes) -> str:
    """识别一张图片，返回按行拼接的文本；引擎不可用时返回空串。"""
    engine = _get_engine()
    if engine is None:
        return ""

    try:
        result, _elapsed = engine(image)
    except Exception as exc:
        sys.stderr.write(f"[WARNING] OCR 识别失败，跳过这一页: {exc}\n")
        return ""

    lines = []
    for item in result or []:
        try:
            text = (item[1] or "").strip()
            score = float(item[2])
        except (IndexError, TypeError, ValueError):
            continue
        if text and score >= MIN_SCORE:
            lines.append(text)
    return "\n".join(lines)
