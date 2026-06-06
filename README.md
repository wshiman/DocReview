# DocReview

DocReview 是一个基于 DeepSeek + Gradio 的文档审校工具，支持 `.doc` / `.docx` / `.pdf` 输入。

## 1. 目录说明（发布最小运行版）

本仓库仅包含最小可运行文件：

- `app.py`
- `pipeline.py`
- `models.py`
- `__init__.py`
- `services/`
- `requirements.txt`
- `README.md`
- `.gitignore`

## 2. 安装与启动

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py --host 0.0.0.0 --port 17930
```

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py --host 127.0.0.1 --port 17930
```

## 3. API Key 与环境变量

### 必需（文本审校）

- `DEEPSEEK_API_KEY`
  - 文本审校必须。
  - DeepSeek 官方 API 文档：`https://api-docs.deepseek.com/`
  - API Key 获取：`https://platform.deepseek.com/api_keys`
  - 默认 base URL：`https://api.deepseek.com`

### PDF 默认解析（PaddleOCR）

- `DOC_REVIEW_PADDLE_API_TOKEN`
  - 当 PDF backend 使用默认 Paddle 时必须。
  - PaddleOCR 官方 API 文档（AI Studio access token）：
    `https://www.paddleocr.ai/latest/en/version3.x/inference_deployment/serving/paddleocr_official_api/overview.html`

### PDF 视觉结构审查（默认开启）

- `DASHSCOPE_API_KEY` 或 `DOC_REVIEW_VISUAL_API_KEY`
  - 当视觉结构审查开启时需要。
  - DashScope/OpenAI 兼容模式 base URL：`https://dashscope.aliyuncs.com/compatible-mode/v1`
  - DashScope Key 获取：`https://help.aliyun.com/zh/model-studio/get-api-key`

### PDF 使用 Doc2X 后端时

- `DOC_REVIEW_DOC2X_API_TOKEN` 或 `DOC2X_API_KEY`
  - 仅当你将前端 PDF backend 切换为 Doc2X 时需要。
  - Doc2X 官方站点/API Key：`https://open.noedgeai.com/`

### PDF 版面解析切换为 MinerU 时

- `MINERU_API_TOKEN`
  - 仅当 `DOC_REVIEW_LAYOUT_BACKEND=mineru` 且启用 MinerU 版面解析时需要。
  - MinerU 官方文档：`https://mineru.net/apiManage/docs?openApplyModal=true`

## 4. 输入类型与依赖差异

### DOCX

- 需要：`DEEPSEEK_API_KEY`
- 不需要：PaddleOCR / Doc2X / MinerU / Qwen 视觉 Key

### DOC

- 需要：`DEEPSEEK_API_KEY`
- 额外需要本机安装 LibreOffice（`soffice`）用于 `.doc -> .docx` 转换

### PDF（默认）

- 需要：`DEEPSEEK_API_KEY` + `DOC_REVIEW_PADDLE_API_TOKEN`
- 若视觉结构审查开启（默认开启）：还需要 `DASHSCOPE_API_KEY` 或 `DOC_REVIEW_VISUAL_API_KEY`

### PDF + Doc2X

- 在前端把 PDF backend 选为 `Doc2X`
- 配置 `DOC_REVIEW_DOC2X_API_TOKEN`（或 `DOC2X_API_KEY`）

## 5. 常用可选变量

- `DEEPSEEK_BASE_URL`：覆盖 DeepSeek base URL
- `DOC_REVIEW_VISUAL_BASE_URL`：覆盖视觉模型 API base URL
- `DOC_REVIEW_LAYOUT_BACKEND`：版面解析后端（`mineru` / `paddle` / `pymupdf`）
- `DOC_REVIEW_ENABLE_VISUAL_STRUCTURE`：是否开启视觉结构审查（默认开启）

