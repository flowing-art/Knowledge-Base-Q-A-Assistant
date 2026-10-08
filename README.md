# 企业知识库智能助手

本地网页问答。上传 PDF / TXT / MD / 图片后，提问时会先检索知识库再回答，并带来源。扫描件没有文字层时会自动 OCR。

## 启动

用 **Agent** 这个 conda 环境。系统自带的 `python`（Anaconda base）没有装 FastAPI，不要用它。

```powershell
cd D:\AIkaifa\rumen
& D:\anaconda3\envs\Agent\python.exe app.py
```

看到 uvicorn 在 `8000` 端口起来后，浏览器打开：

`http://127.0.0.1:8000`

服务默认只监听本机（`127.0.0.1`）。要给局域网访问再设 `KB_HOST=0.0.0.0`，并先把 `.env` 里的 `KB_TOKEN` 改掉。

改过页面后用 **Ctrl+F5** 强刷，否则浏览器还是旧的 `index.html`。

停服务：在启动它的那个终端按 **Ctrl+C**。不要另开一个服务占着 8000。

如果已经执行过 `conda init powershell`，也可以新开一个终端：

```powershell
conda activate Agent
cd D:\AIkaifa\rumen
python app.py
```

同一个窗口里刚执行完 `conda init` 往往还不生效，要新开终端。

## 第一次使用前

在项目根目录放 `.env`（已有就不用改）：

```
DEEPSEEK_API_KEY=你的密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
KB_TOKEN=rumen-local
```

`KB_TOKEN` 是上传文档、删除知识库文件、删除会话时要填的口令。页面上方有「管理口令」输入框，填和 `.env` 一样的值。提问不需要口令。默认值只适合本机，不要原样暴露到公网。

程序会用 `.env` 覆盖同名的系统环境变量。如果系统里还留着旧密钥，回答会报 `401 api key is invalid`，以 `.env` 为准即可。

模型权重已经在 `models_cache` 里，启动时不需要联网下载：

- 向量模型：`BAAI/bge-small-zh-v1.5`
- 重排模型：`BAAI/bge-reranker-base`
- OCR：`models_cache\rapidocr` 下三个 onnx

缺重排权重时，在同一环境下运行 `download_models.py`，它只会把文件下到本项目的 `models_cache`，不会写到 C 盘。

## 网页怎么用

打开后是左右两栏：左边「会话历史 / 知识库文件」，右边对话。

1. 在「管理口令」里填 `.env` 的 `KB_TOKEN`，再点「选择文件」和「上传文档」。支持 `.pdf .txt .md .png .jpg .jpeg .bmp .tif .tiff .webp`。单个文件默认不超过 50 MB，改环境变量 `MAX_UPLOAD_MB` 可以放宽。口令会留在当前标签页，关掉就没了。
2. 扫描版 PDF、图片会走 OCR；本来就有文字层的 PDF 不会重复识别。
3. 在输入框提问，回车或点「发送」。回答是边生成边出来的。气泡上方有一行过程条，点开能看到检索了什么。
4. 问「最高 / 最低 / 求和 / 平均 / 所有」这类汇总时，一次检索会把相关条目都取回来再计算，回答里应保留 `[来源: 文件名 第N页]`。
5. **刷新页面会开一条新会话**，输入框是空的。旧对话还在左侧「会话历史」里，点进去可以接着问。点右上角「新会话」效果相同。
6. 「知识库文件」里能看到已入库文件和块数，点 `✕` 会删掉该文件的全部内容（会再确认一次，同样要口令）。删除左侧某条会话也要这句口令。

第一次提问会拉起检索子进程并加载模型，大约 10～20 秒，之后会快一些。上传和提问共用一把锁，大扫描件 OCR 时提问会等它结束。

## 命令行（可选）

不打开网页、只在终端里问：

```powershell
cd D:\AIkaifa\rumen
& D:\anaconda3\envs\Agent\python.exe agent.py
```

常用命令：`/help`、`/trace`（打印检索和计算过程）、`/steps`、`/graph`、`exit`。启动时加 `--trace` 或 `-t` 等于一开始就打开轨迹。

## 测试和评测

不启动网页服务，在项目目录执行：

```powershell
& D:\anaconda3\envs\Agent\python.exe -m unittest discover -s tests -v
```

覆盖三件事：会话的记录和删除、重排过低时的字面兜底、上传/删除的口令和文件大小上限。出错信息里不会带本机路径。

对当前知识库重跑检索分数（会加载本地模型，不访问外网）：

```powershell
& D:\anaconda3\envs\Agent\python.exe eval_retrieval.py
```

结果在 `eval/retrieval.md`。同一份软著表，「开发者都有谁」的重排分远低于 `0.1`，「全体开发者都是谁」远高于 `0.1`；前者靠字面兜底捞回，无关问题仍然拒绝。

换一台机器时，先装依赖再准备模型：

```powershell
pip install -r requirements.txt
```

`.env`、`history.db`、`chroma_db`、`models_cache` 已写进 `.gitignore`，不要提交。

## 目录

| 路径 | 作用 |
|---|---|
| `app.py` | 网页服务，默认只监听本机 8000 |
| `requirements.txt` | 依赖版本 |
| `tests/` | 不启动服务就能跑的检查 |
| `eval_retrieval.py` | 重跑检索评测，写出 `eval/retrieval.md` |
| `static/index.html` | 页面 |
| `agent.py` | 对话和工具调用 |
| `mcp_server.py` | 检索、计算用的子进程 |
| `rag.py` | 入库、检索、重排 |
| `ocr.py` | 扫描件 / 图片识别 |
| `history.py` | 会话记录，库文件是 `history.db`（第一次提问后才出现） |
| `download_models.py` | 缺权重时把重排模型下到 `models_cache` |
| `sample_docs/` | 虚构的演示文档，可直接上传试问（试试「LOLv1 上最高的 PSNR 是多少」） |
| `chroma_db` | 向量库，删掉等于清空知识库 |
| `models_cache` | 模型权重，不要挪到 C 盘 |
| `.env.example` | 配置模板，复制成 `.env` 再填 |
| `.env` | API 密钥，不要提交出去 |

## 常见问题

- **打不开页面**：确认是用 Agent 环境的 `python.exe` 启动的，并且终端还停在运行状态。`http://127.0.0.1:8000` 连不上就是服务没起来。
- **401 密钥无效**：检查 `.env` 里的 `DEEPSEEK_API_KEY`，改完要重启 `app.py`。上传或删除提示「口令不正确」是另一件事，填的是 `KB_TOKEN`，不是 API 密钥。
- **接口报错但页面只说看终端**：完整原因在运行 `app.py` 的那个窗口里，不会再把本机路径回给浏览器。
- **刚上传的文件搜不到**：看上传状态是不是绿色成功。同一文件再传一次会按文件名替换旧内容，不会叠成两份。
- **换个说法就说「没有相关内容」**：重排分数太低时，只要文档里确实出现了问题里的词，仍会把原文交回模型，并标成低置信度。完全无关的问题仍会说没找到。
- **Ctrl+C 关不掉**：在启动服务的那个窗口按。关的是 `app.py` 那个进程；不要对已经退出的窗口再按。
