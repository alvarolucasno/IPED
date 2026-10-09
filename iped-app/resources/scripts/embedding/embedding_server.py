"""
IPED multimodal embedding server (EmbeddingGemma 2).

Local HTTP service used by iped.engine.task.EmbeddingTask (processing) and by the
analysis GUI semantic search. It maps text, images, video and audio into a single
L2-normalized vector space, so a text query can retrieve any modality.

Contract "iped-embedding/v1":

  GET  /info  -> {"contract", "model", "dim", "modalities", "device", "dtype", ...}
  POST /embed -> request  {"inputs": [{"id": "1", "type": "document", "title": "...", "text": "..."},
                                      {"id": "2", "type": "query", "text": "..."},
                                      {"id": "3", "type": "image", "data": "<base64 image file>"},
                                      {"id": "4", "type": "video", "frames": ["<base64 jpeg>", ...]},
                                      {"id": "5", "type": "video", "data": "<base64 video file>"},
                                      {"id": "6", "type": "audio", "data": "<base64 audio file>"}]}
                 response {"model", "dim", "vectors": {"1": [...], ...}, "errors": {"6": "msg"}}

Run:  python embedding_server.py [--port 8691] [--dim 768] [--modalities text,image,video,audio]
All options can also be set by environment variables (see README.md).
"""

import argparse
import base64
import io
import logging
import os
import threading
import time
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageFile

CONTRACT = "iped-embedding/v1"
ALL_MODALITIES = ("text", "image", "video", "audio")
AUDIO_SAMPLE_RATE = 16000
QUERY_PREFIX = "task: search result | query: "

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 300_000_000

logger = logging.getLogger("iped-embedding")


def _env(name: str, default):
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return type(default)(value) if not isinstance(default, bool) else value.lower() in ("1", "true", "yes")


class Settings:
    def __init__(self, **overrides):
        self.model = _env("IPED_EMBEDDING_MODEL", "google/embeddinggemma-2")
        self.host = _env("IPED_EMBEDDING_HOST", "127.0.0.1")
        self.port = _env("IPED_EMBEDDING_PORT", 8691)
        self.dim = _env("IPED_EMBEDDING_DIM", 768)
        self.device = _env("IPED_EMBEDDING_DEVICE", "auto")
        self.modalities = _env("IPED_EMBEDDING_MODALITIES", ",".join(ALL_MODALITIES))
        self.batch_size = _env("IPED_EMBEDDING_BATCH_SIZE", 16)
        self.max_text_tokens = _env("IPED_EMBEDDING_MAX_TEXT_TOKENS", 2048)
        self.image_size = _env("IPED_EMBEDDING_IMAGE_SIZE", 768)
        self.max_video_frames = _env("IPED_EMBEDDING_MAX_VIDEO_FRAMES", 16)
        self.max_audio_seconds = _env("IPED_EMBEDDING_MAX_AUDIO_SECONDS", 300)
        for key, value in overrides.items():
            if value is not None:
                setattr(self, key, value)
        mods = [m.strip().lower() for m in str(self.modalities).split(",") if m.strip()]
        unknown = set(mods) - set(ALL_MODALITIES)
        if unknown:
            raise ValueError(f"Unknown modalities: {sorted(unknown)}")
        if "text" not in mods:
            mods.insert(0, "text")  # queries are always text
        self.modalities = mods
        if self.dim not in (128, 256, 512, 768):
            raise ValueError("dim must be one of 128, 256, 512, 768 (Matryoshka sizes)")


# ---------------------------------------------------------------------------
# Input formatting and decoding (no torch needed, unit-testable)
# ---------------------------------------------------------------------------

def format_document(title: Optional[str], text: str) -> str:
    title = (title or "").strip().replace("\n", " ") or "none"
    return f"title: {title} | text: {text}"


def format_query(text: str) -> str:
    return QUERY_PREFIX + text


def decode_image(data: bytes, max_side: int) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img.draft("RGB", (max_side, max_side))  # fast JPEG downscale on decode
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side), Image.BICUBIC)
    return img


def decode_audio(data: bytes, max_seconds: float) -> np.ndarray:
    """Decode any container/codec supported by FFmpeg (PyAV) to mono float32 16 kHz."""
    max_samples = int(max_seconds * AUDIO_SAMPLE_RATE)
    try:
        import av

        chunks, total = [], 0
        with av.open(io.BytesIO(data)) as container:
            stream = next(s for s in container.streams if s.type == "audio")
            resampler = av.AudioResampler(format="flt", layout="mono", rate=AUDIO_SAMPLE_RATE)
            for frame in container.decode(stream):
                for out in resampler.resample(frame):
                    arr = out.to_ndarray().reshape(-1)
                    chunks.append(arr)
                    total += arr.size
                if total >= max_samples:
                    break
            for out in resampler.resample(None):
                chunks.append(out.to_ndarray().reshape(-1))
        if not chunks:
            raise ValueError("no audio samples decoded")
        wav = np.concatenate(chunks).astype(np.float32)
    except StopIteration:
        raise ValueError("no audio stream found")
    except ImportError:
        import soundfile as sf

        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        if sr != AUDIO_SAMPLE_RATE:
            idx = np.linspace(0, len(wav) - 1, int(len(wav) * AUDIO_SAMPLE_RATE / sr))
            wav = np.interp(idx, np.arange(len(wav)), wav).astype(np.float32)
    wav = wav[:max_samples]
    if wav.size < AUDIO_SAMPLE_RATE // 10:
        raise ValueError("audio too short")
    return wav


def decode_video_file(data: bytes, max_frames: int, max_side: int) -> List[Image.Image]:
    """Sample up to max_frames frames uniformly from a video file."""
    import av

    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        total = stream.frames or 0
        if total <= 0 and stream.duration and stream.average_rate:
            total = int(float(stream.duration * stream.time_base) * float(stream.average_rate))
        frames = []
        if total > 0:
            wanted = set(np.linspace(0, total - 1, min(max_frames, total)).astype(int).tolist())
            for i, frame in enumerate(container.decode(stream)):
                if i in wanted:
                    frames.append(frame.to_image())
                if i >= max(wanted):
                    break
        else:
            for frame in container.decode(stream):
                frames.append(frame.to_image())
            if len(frames) > max_frames:
                idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
                frames = [frames[i] for i in idx]
    if not frames:
        raise ValueError("no video frames decoded")
    out = []
    for f in frames:
        f = f.convert("RGB")
        f.thumbnail((max_side, max_side), Image.BICUBIC)
        out.append(f)
    return out


def subsample(items: list, max_items: int) -> list:
    if len(items) <= max_items:
        return items
    idx = np.linspace(0, len(items) - 1, max_items).astype(int)
    return [items[i] for i in idx]


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class Embedder:
    def __init__(self, settings: Settings):
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers.utils import logging as hf_logging

        # IPED sends already sampled frames without fps metadata; that warning is expected noise
        hf_logging.set_verbosity_error()

        self.settings = settings
        device = settings.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # Never float16: EmbeddingGemma 2 activations overflow it (see model card).
        if device.startswith("cuda") and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        config_kwargs = {}
        if "image" not in settings.modalities and "video" not in settings.modalities:
            config_kwargs["vision_config"] = None
        if "audio" not in settings.modalities:
            config_kwargs["audio_config"] = None
        kwargs = dict(device=device, model_kwargs={"torch_dtype": dtype}, config_kwargs=config_kwargs)
        t = time.time()
        try:
            self.model = SentenceTransformer(settings.model, local_files_only=True, **kwargs)
        except Exception:
            logger.info("Model not in local cache, downloading %s", settings.model)
            self.model = SentenceTransformer(settings.model, **kwargs)
        self.model.eval()
        logger.info("Loaded %s on %s (%s) in %.1fs, modalities=%s, dim=%d", settings.model, device, dtype,
                    time.time() - t, settings.modalities, settings.dim)
        self.device = device
        self.dtype = str(dtype).replace("torch.", "")
        self.lock = threading.Lock()
        self.tokenizer = self._find_tokenizer()

    def _find_tokenizer(self):
        tok = getattr(self.model, "tokenizer", None)
        tok = getattr(tok, "tokenizer", tok)  # multimodal processors wrap the text tokenizer
        return tok if callable(getattr(tok, "encode", None)) else None

    def truncate_text(self, text: str) -> str:
        max_tokens = self.settings.max_text_tokens
        if self.tokenizer is None or len(text) <= max_tokens * 2:
            return text
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text
        return self.tokenizer.decode(ids[:max_tokens])

    def info(self) -> dict:
        return {
            "contract": CONTRACT,
            "model": self.settings.model,
            "dim": self.settings.dim,
            "modalities": self.settings.modalities,
            "device": self.device,
            "dtype": self.dtype,
            "max_text_tokens": self.settings.max_text_tokens,
            "image_size": self.settings.image_size,
            "max_video_frames": self.settings.max_video_frames,
            "max_audio_seconds": self.settings.max_audio_seconds,
            "query_prefix": QUERY_PREFIX,
        }

    def prepare(self, item: dict):
        """Turns one request input into (group, model_input). Raises ValueError on bad input."""
        kind = (item.get("type") or "").lower()
        s = self.settings
        if kind in ("document", "text"):
            text = item.get("text") or ""
            if not text.strip():
                raise ValueError("empty text")
            return "text", format_document(item.get("title"), self.truncate_text(text))
        if kind == "query":
            text = (item.get("text") or "").strip()
            if not text:
                raise ValueError("empty query")
            return "text", format_query(self.truncate_text(text))
        if kind == "image":
            self._require("image")
            return "image", decode_image(_b64(item, "data"), s.image_size)
        if kind == "video":
            self._require("video")
            if item.get("frames"):
                frames = [decode_image(base64.b64decode(f), s.image_size) for f in item["frames"]]
                frames = subsample(frames, s.max_video_frames)
            else:
                frames = decode_video_file(_b64(item, "data"), s.max_video_frames, s.image_size)
            if len(frames) == 1:
                return "image", frames[0]
            return "video", {"video": frames}
        if kind == "audio":
            self._require("audio")
            wav = decode_audio(_b64(item, "data"), s.max_audio_seconds)
            return "audio", {"audio": {"array": wav, "sampling_rate": AUDIO_SAMPLE_RATE}}
        raise ValueError(f"unknown input type '{kind}'")

    def _require(self, modality: str):
        if modality not in self.settings.modalities:
            raise ValueError(f"modality '{modality}' not enabled on this server")

    def _encode(self, inputs: list, batch_size: int) -> np.ndarray:
        import torch

        with self.lock, torch.inference_mode():
            vecs = self.model.encode(inputs, batch_size=batch_size, truncate_dim=self.settings.dim,
                                     normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        vecs = np.asarray(vecs, dtype=np.float32)
        # re-normalize in float32: bf16 normalization leaves norms ~1e-3 off, and clients use dot == cosine
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms > 0, norms, 1)

    def embed(self, items: List[dict]) -> dict:
        vectors: Dict[str, list] = {}
        errors: Dict[str, str] = {}
        groups: Dict[str, list] = {}
        for n, item in enumerate(items):
            item_id = str(item.get("id", n))
            try:
                group, model_input = self.prepare(item)
                groups.setdefault(group, []).append((item_id, model_input))
            except Exception as e:  # bad/corrupted input never fails the whole batch
                errors[item_id] = f"{type(e).__name__}: {e}"
        # heavier modalities get smaller batches to bound GPU memory
        batch_by_group = {"text": self.settings.batch_size, "image": self.settings.batch_size,
                          "video": max(1, self.settings.batch_size // 8), "audio": max(1, self.settings.batch_size // 4)}
        for group, entries in groups.items():
            ids = [e[0] for e in entries]
            inputs = [e[1] for e in entries]
            try:
                vecs = self._encode(inputs, batch_by_group[group])
                self._collect(ids, vecs, vectors, errors)
            except Exception as e:
                logger.warning("Batch of %d %s inputs failed (%s), retrying one by one", len(ids), group, e)
                _free_cuda()
                for item_id, model_input in entries:
                    try:
                        self._collect([item_id], self._encode([model_input], 1), vectors, errors)
                    except Exception as e1:
                        errors[item_id] = f"{type(e1).__name__}: {e1}"
                        _free_cuda()
        return {"model": self.settings.model, "dim": self.settings.dim, "vectors": vectors, "errors": errors}

    @staticmethod
    def _collect(ids, vecs, vectors, errors):
        for item_id, vec in zip(ids, vecs):
            if not np.all(np.isfinite(vec)):
                errors[item_id] = "non-finite embedding"
            else:
                vectors[item_id] = vec.tolist()


def _b64(item: dict, key: str) -> bytes:
    value = item.get(key)
    if not value:
        raise ValueError(f"missing '{key}'")
    return base64.b64decode(value)


def _free_cuda():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# HTTP app
# ---------------------------------------------------------------------------

def create_app(embedder) -> "FastAPI":
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    class EmbedRequest(BaseModel):
        inputs: List[dict]

    app = FastAPI(title="IPED embedding server", version=CONTRACT)

    @app.get("/info")
    def info():
        return embedder.info()

    @app.post("/embed")
    def embed(req: EmbedRequest):  # sync handler: runs in threadpool, GPU access serialized by lock
        if not req.inputs:
            raise HTTPException(status_code=400, detail="no inputs")
        t = time.time()
        result = embedder.embed(req.inputs)
        logger.info("embedded %d inputs (%d errors) in %.2fs", len(result["vectors"]), len(result["errors"]),
                    time.time() - t)
        return result

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--dim", type=int)
    parser.add_argument("--device")
    parser.add_argument("--modalities", help="comma separated subset of text,image,video,audio")
    parser.add_argument("--batch-size", dest="batch_size", type=int)
    parser.add_argument("--max-text-tokens", dest="max_text_tokens", type=int)
    parser.add_argument("--image-size", dest="image_size", type=int)
    parser.add_argument("--max-video-frames", dest="max_video_frames", type=int)
    parser.add_argument("--max-audio-seconds", dest="max_audio_seconds", type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings(**vars(args))
    embedder = Embedder(settings)

    import uvicorn

    uvicorn.run(create_app(embedder), host=settings.host, port=settings.port, log_level="warning")


if __name__ == "__main__":
    main()
