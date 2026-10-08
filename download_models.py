import os

# 模型只下载到工作区，不使用用户主目录（通常在 C 盘）下的 HuggingFace 缓存。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CACHE_DIR = os.path.join(BASE_DIR, "models_cache")
HUB_CACHE_DIR = os.path.join(MODEL_CACHE_DIR, "hub")
os.makedirs(HUB_CACHE_DIR, exist_ok=True)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = MODEL_CACHE_DIR
os.environ["HF_HUB_CACHE"] = HUB_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = HUB_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HUB_CACHE_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = MODEL_CACHE_DIR
os.environ["TORCH_HOME"] = os.path.join(MODEL_CACHE_DIR, "torch")
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
# Windows 上建符号链接经常要管理员权限，直接存真实文件。
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
# 镜像不支持 Xet。不开这个开关时，大文件会停在 0 字节。
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import snapshot_download

# 两份权重内容相同。只下 safetensors，大约 1.13 GB，不下 pytorch_model.bin 和 onnx。
NEEDED = [
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
]

print("开始下载 Reranker 模型到", HUB_CACHE_DIR, flush=True)

try:
    local_path = snapshot_download(
        repo_id="BAAI/bge-reranker-base",
        cache_dir=HUB_CACHE_DIR,
        allow_patterns=NEEDED,
    )
    print("下载完成！", local_path, flush=True)
except Exception as exc:
    print(f"从 HuggingFace 镜像下载失败: {exc}", flush=True)
    raise
