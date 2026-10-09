# IPED embedding service (EmbeddingGemma 2)

Local HTTP service that turns item content into [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2)
vectors. Text, images, videos and audio share one 768-dimensional space, so a
natural language query (in any language) can find documents, chats, photos,
videos and spoken audio by meaning.

It is used by:

- `EmbeddingTask` during processing (`enableEmbedding = true` in `IPEDConfig.txt`,
  options in `conf/EmbeddingTaskConfig.txt`);
- the analysis UI "Semantic Search" button and the "Semantic Search" context menu
  (by text, by external file, or similar to the highlighted item). Searching
  similar to an item does not need the service; text and external file queries do.

Vectors are stored in the case index (field `embedding`, modality in
`embedding:modality`), so processed cases stay portable and work in multicases.

## Install

Python 3.12 is recommended. With [uv](https://docs.astral.sh/uv/), from this folder:

```bat
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe --index-url https://download.pytorch.org/whl/cu130 torch==2.14.1 torchvision==0.29.1
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

Use the PyTorch index matching your NVIDIA driver (`cu130`, `cu128`...), or plain
`torch torchvision` for CPU only. Install torch first: installing
`sentence-transformers` before may pull a CPU build.

The model (~1.5 GB) is downloaded from Hugging Face on first start (no token
needed) and cached in `%USERPROFILE%\.cache\huggingface`. For offline machines,
copy that cache or point `--model` to a local folder.

## Run

```bat
start_embedding_server.bat
```

It listens on `127.0.0.1:8691` (loopback: item content is sent to it). Check with
`curl http://127.0.0.1:8691/info`.

| Option | Environment variable | Default | |
|---|---|---|---|
| `--model` | `IPED_EMBEDDING_MODEL` | `google/embeddinggemma-2` | HF id or local folder |
| `--host` / `--port` | `IPED_EMBEDDING_HOST` / `IPED_EMBEDDING_PORT` | `127.0.0.1` / `8691` | must match `serviceUrl` |
| `--dim` | `IPED_EMBEDDING_DIM` | `768` | 768, 512, 256 or 128 (Matryoshka); must match `dimensions` |
| `--modalities` | `IPED_EMBEDDING_MODALITIES` | `text,image,video,audio` | unused encoders are not loaded |
| `--device` | `IPED_EMBEDDING_DEVICE` | `auto` | `cuda`, `cuda:1`, `cpu` |
| `--batch-size` | `IPED_EMBEDDING_BATCH_SIZE` | `16` | video/audio use smaller batches |
| `--max-text-tokens` | `IPED_EMBEDDING_MAX_TEXT_TOKENS` | `2048` | model supports up to 8192 |
| `--image-size` | `IPED_EMBEDDING_IMAGE_SIZE` | `768` | largest side of images/frames |
| `--max-video-frames` | `IPED_EMBEDDING_MAX_VIDEO_FRAMES` | `16` | |
| `--max-audio-seconds` | `IPED_EMBEDDING_MAX_AUDIO_SECONDS` | `300` | |

On CUDA GPUs with bfloat16 support the model runs in bfloat16, otherwise in
float32. It never uses float16 (the model overflows it and returns NaN).

## Contract `iped-embedding/v1`

`GET /info` returns `contract`, `model`, `dim`, `modalities`, `device`, `dtype`.

`POST /embed`:

```json
{"inputs": [
  {"id": "1", "type": "document", "title": "contrato.docx", "text": "..."},
  {"id": "2", "type": "query", "text": "comprovante de aluguel"},
  {"id": "3", "type": "image", "data": "<base64 image file>"},
  {"id": "4", "type": "video", "frames": ["<base64 jpeg>", "..."]},
  {"id": "5", "type": "video", "data": "<base64 video file>"},
  {"id": "6", "type": "audio", "data": "<base64 audio file, any FFmpeg format>"}
]}
```

returns `{"model", "dim", "vectors": {"1": [...]}, "errors": {"6": "message"}}`.
Vectors are L2-normalized (dot product = cosine). Prompts follow the model card:
documents as `title: {name} | text: {content}`, queries as
`task: search result | query: {query}`, media without prefix. A bad input only
fails itself, never the whole batch.

## Tests

```bat
.venv\Scripts\python -m pytest tests -q
set IPED_EMBEDDING_IT=1 && .venv\Scripts\python -m pytest tests -q
```

The second form also runs integration tests with the real model.

## Windows notes

- Smart App Control may block DLLs of recent wheels ("Uma política de Controle de
  Aplicativo bloqueou este arquivo"). `scikit-learn` is pinned to 1.6.1 for that
  reason; if another package is blocked, try a previous release of it.
- Behind TLS-inspecting proxies the first download may log SSL retries; once the
  model is cached the service loads it offline.
