"""
Tests for embedding_server.py.

Unit tests use a fake model and run anywhere. Integration tests load the real
EmbeddingGemma 2 model and only run when IPED_EMBEDDING_IT=1.

    .venv\\Scripts\\python -m pytest tests -q
    set IPED_EMBEDDING_IT=1 && .venv\\Scripts\\python -m pytest tests -q
"""

import base64
import io
import os
import sys
import threading

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import embedding_server as srv  # noqa: E402

RUN_IT = os.environ.get("IPED_EMBEDDING_IT") == "1"


def png_bytes(color=(220, 20, 20), size=(320, 240)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def wav_bytes(seconds=1.0, sr=44100, freq=440.0, stereo=True) -> bytes:
    import soundfile as sf

    t = np.linspace(0, seconds, int(sr * seconds), dtype=np.float32)
    wav = 0.3 * np.sin(2 * np.pi * freq * t)
    if stereo:
        wav = np.stack([wav, wav], axis=1)
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    return buf.getvalue()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


# ---------------------------------------------------------------------------
# formatting / decoding
# ---------------------------------------------------------------------------

def test_prompts_follow_model_card():
    assert srv.format_document("Contrato.docx", "aluguel") == "title: Contrato.docx | text: aluguel"
    assert srv.format_document(None, "x") == "title: none | text: x"
    assert srv.format_document("a\nb", "x") == "title: a b | text: x"
    assert srv.format_query("carro vermelho") == "task: search result | query: carro vermelho"


def test_decode_image_converts_and_limits_size():
    img = srv.decode_image(png_bytes(size=(2000, 1000)), 512)
    assert img.mode == "RGB"
    assert max(img.size) == 512


def test_decode_audio_resamples_to_mono_16k():
    wav = srv.decode_audio(wav_bytes(seconds=2.0, sr=44100), max_seconds=300)
    assert wav.dtype == np.float32 and wav.ndim == 1
    assert abs(wav.size - 2 * srv.AUDIO_SAMPLE_RATE) < 400


def test_decode_audio_respects_max_seconds():
    wav = srv.decode_audio(wav_bytes(seconds=3.0, sr=16000), max_seconds=1)
    assert wav.size == srv.AUDIO_SAMPLE_RATE


def test_decode_audio_rejects_garbage():
    with pytest.raises(Exception):
        srv.decode_audio(b"not an audio file at all" * 10, max_seconds=10)


def test_subsample_keeps_first_and_last():
    assert srv.subsample(list(range(100)), 5) == [0, 24, 49, 74, 99]
    assert srv.subsample([1, 2], 5) == [1, 2]


def test_settings_validation(monkeypatch):
    s = srv.Settings(modalities="image,audio", dim=256)
    assert s.modalities == ["text", "image", "audio"]
    with pytest.raises(ValueError):
        srv.Settings(dim=300)
    with pytest.raises(ValueError):
        srv.Settings(modalities="text,smell")
    monkeypatch.setenv("IPED_EMBEDDING_PORT", "9000")
    assert srv.Settings().port == 9000


# ---------------------------------------------------------------------------
# Embedder with a fake model: grouping, error isolation, HTTP layer
# ---------------------------------------------------------------------------

class FakeModel:
    """Deterministic stand-in: the vector depends on the input group, so tests can check routing."""

    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def encode(self, inputs, batch_size, truncate_dim, normalize_embeddings, **kw):
        self.calls.append((len(inputs), batch_size))
        out = []
        for x in inputs:
            if self.fail_on is not None and self.fail_on(x):
                raise RuntimeError("boom")
            v = np.zeros(truncate_dim, dtype=np.float32)
            if isinstance(x, str):
                v[0 if x.startswith("task:") else 1] = 1
            elif isinstance(x, Image.Image):
                v[2] = 1
            elif "video" in x:
                v[3] = 1
            elif "audio" in x:
                v[4] = 1
            out.append(v)
        return np.stack(out)


def fake_embedder(model=None, **settings):
    e = srv.Embedder.__new__(srv.Embedder)
    e.settings = srv.Settings(**settings)
    e.model = model or FakeModel()
    e.device, e.dtype = "cpu", "float32"
    e.lock = threading.Lock()
    e.tokenizer = None
    return e


def test_embed_routes_every_modality_and_isolates_errors():
    frame = jpeg_bytes(Image.new("RGB", (64, 64), (0, 0, 255)))
    e = fake_embedder()
    res = e.embed([
        {"id": "q", "type": "query", "text": "carro vermelho"},
        {"id": "d", "type": "document", "title": "a.txt", "text": "conteúdo"},
        {"id": "i", "type": "image", "data": b64(png_bytes())},
        {"id": "v", "type": "video", "frames": [b64(frame)] * 3},
        {"id": "v1", "type": "video", "frames": [b64(frame)]},
        {"id": "a", "type": "audio", "data": b64(wav_bytes())},
        {"id": "bad", "type": "image", "data": b64(b"garbage")},
        {"id": "empty", "type": "document", "text": "   "},
        {"id": "unk", "type": "smell"},
    ])
    vec = {k: int(np.argmax(v)) for k, v in res["vectors"].items()}
    assert vec == {"q": 0, "d": 1, "i": 2, "v": 3, "v1": 2, "a": 4}
    assert set(res["errors"]) == {"bad", "empty", "unk"}
    assert res["dim"] == 768 and len(res["vectors"]["q"]) == 768


def test_embed_retries_one_by_one_when_batch_fails():
    model = FakeModel(fail_on=lambda x: isinstance(x, str) and "poison" in x)
    e = fake_embedder(model)
    res = e.embed([
        {"id": "1", "type": "document", "text": "ok one"},
        {"id": "2", "type": "document", "text": "poison"},
        {"id": "3", "type": "document", "text": "ok two"},
    ])
    assert set(res["vectors"]) == {"1", "3"}
    assert "boom" in res["errors"]["2"]


def test_disabled_modality_is_reported_per_item():
    e = fake_embedder(modalities="text")
    res = e.embed([{"id": "a", "type": "audio", "data": b64(wav_bytes())},
                   {"id": "t", "type": "query", "text": "x"}])
    assert "not enabled" in res["errors"]["a"]
    assert "t" in res["vectors"]


def test_truncate_dim_is_honored():
    e = fake_embedder(dim=256)
    res = e.embed([{"id": "q", "type": "query", "text": "x"}])
    assert len(res["vectors"]["q"]) == 256


def test_http_endpoints():
    from fastapi.testclient import TestClient

    client = TestClient(srv.create_app(fake_embedder()))
    info = client.get("/info").json()
    assert info["contract"] == srv.CONTRACT and info["dim"] == 768
    assert info["modalities"] == list(srv.ALL_MODALITIES)
    r = client.post("/embed", json={"inputs": [{"id": "x", "type": "query", "text": "oi"}]})
    assert r.status_code == 200 and "x" in r.json()["vectors"]
    assert client.post("/embed", json={"inputs": []}).status_code == 400


# ---------------------------------------------------------------------------
# Integration with the real model (GPU recommended)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real():
    if not RUN_IT:
        pytest.skip("set IPED_EMBEDDING_IT=1 to run integration tests with the real model")
    return srv.Embedder(srv.Settings())


def draw_car(color):
    img = Image.new("RGB", (512, 384), (200, 220, 240))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 300, 512, 384], fill=(90, 90, 90))
    d.rounded_rectangle([80, 200, 440, 300], radius=20, fill=color)
    d.polygon([(150, 200), (200, 140), (340, 140), (390, 200)], fill=color)
    for x in (150, 370):
        d.ellipse([x - 35, 265, x + 35, 335], fill=(20, 20, 20))
    return img


def cos_rank(res, query_id):
    q = np.array(res["vectors"][query_id])
    sims = {k: float(np.dot(q, np.array(v))) for k, v in res["vectors"].items() if k != query_id}
    return sorted(sims, key=sims.get, reverse=True)


def test_real_text_retrieval_portuguese(real):
    res = real.embed([
        {"id": "q", "type": "query", "text": "contrato de locação de imóvel residencial"},
        {"id": "aluguel", "type": "document", "title": "contrato.docx",
         "text": "O locador cede ao locatário o apartamento situado na Rua das Flores pelo prazo de 30 meses, "
                 "mediante aluguel mensal de R$ 2.000,00."},
        {"id": "futebol", "type": "document", "title": "jogo.txt",
         "text": "O time venceu a partida por 3 a 1 com dois gols no segundo tempo."},
        {"id": "receita", "type": "document", "title": "bolo.txt",
         "text": "Misture a farinha, os ovos e o açúcar e leve ao forno por 40 minutos."},
    ])
    assert not res["errors"]
    assert cos_rank(res, "q")[0] == "aluguel"


def test_real_cross_modal_text_to_image_video_audio(real):
    red_car = draw_car((210, 20, 20))
    frames = [b64(jpeg_bytes(draw_car((210, 20, 20)).rotate(a))) for a in (0, 2, 4, 6)]
    cat = Image.new("RGB", (512, 384), (240, 240, 230))
    ImageDraw.Draw(cat).ellipse([150, 100, 360, 330], fill=(30, 30, 30))
    res = real.embed([
        {"id": "q", "type": "query", "text": "a red car"},
        {"id": "car", "type": "image", "data": b64(jpeg_bytes(red_car))},
        {"id": "blob", "type": "image", "data": b64(jpeg_bytes(cat))},
        {"id": "video", "type": "video", "frames": frames},
        {"id": "audio", "type": "audio", "data": b64(wav_bytes(seconds=2))},
    ])
    assert not res["errors"], res["errors"]
    ranking = cos_rank(res, "q")
    assert ranking.index("car") < ranking.index("blob")
    assert ranking.index("video") < ranking.index("audio")
    for v in res["vectors"].values():
        assert abs(np.linalg.norm(v) - 1) < 1e-3
