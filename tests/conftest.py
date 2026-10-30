"""Shared test setup.

docuquery/constants.py reads the environment when it is first imported, so the
settings below must be in place before any test imports a docuquery module.
pytest imports this file first, which guarantees that.
"""

import copy
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# Force test values, so a developer's real settings are never used.
# Port 1 on localhost: nothing listens there, so no real database is touched.
os.environ["DOCUQUERY_POSTGRES_URL"] = os.environ.get(
    "DOCUQUERY_TEST_POSTGRES_URL", "postgresql://docuquery:test@127.0.0.1:1/test"
)
os.environ["OPENAI_API_KEY"] = "sk-test"
os.environ["DOCUQUERY_OPENAI_MODEL"] = "test-model"
os.environ["SENTENCE_TRANSFORMER_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"
os.environ["DOCUQUERY_CHUNK_SIZE"] = "512"
os.environ["DOCUQUERY_OVERLAP"] = "50"

EMBEDDING_DIMENSION = 384


@pytest.fixture(autouse=True)
def isolated_config_file(tmp_path, monkeypatch):
    """Point the config file at a temporary path, never the user's real one."""
    import docuquery.utils  # noqa: PLC0415 - must import after the env setup above

    config_file = tmp_path / "config" / "config.env"
    monkeypatch.setattr(docuquery.utils, "CONFIG_FILE", config_file)
    return config_file


class FakeEmbedder:
    """Stands in for SentenceTransformer: deterministic vectors, no download."""

    def __init__(self, *_args, **_kwargs):
        self.calls = []

    def encode(self, texts, **_kwargs):
        self.calls.append(texts)
        if isinstance(texts, str):
            return self._vector(texts)
        return np.array([self._vector(t) for t in texts])

    @staticmethod
    def _vector(text):
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")
        return np.random.default_rng(seed).random(EMBEDDING_DIMENSION)


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


def _chunk(content=None, tool_calls=None, finish_reason=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)]
    )


def text_reply(text):
    """A streamed answer, one chunk per word, like the real API."""
    words = text.split(" ")
    chunks = [_chunk(content=w if i == 0 else f" {w}") for i, w in enumerate(words)]
    return [*chunks, _chunk(finish_reason="stop")]


def tool_reply(*calls):
    """A streamed tool-call turn. Each call is (call_id, name, arguments_json).

    The name and arguments arrive split over two chunks, as the real API sends them.
    """
    chunks = []
    for index, (call_id, name, arguments) in enumerate(calls):
        middle = len(arguments) // 2
        first = SimpleNamespace(
            index=index,
            id=call_id,
            function=SimpleNamespace(name=name, arguments=arguments[:middle]),
        )
        second = SimpleNamespace(
            index=index,
            id=None,
            function=SimpleNamespace(name=None, arguments=arguments[middle:]),
        )
        chunks += [_chunk(tool_calls=[first]), _chunk(tool_calls=[second])]
    return [*chunks, _chunk(finish_reason="tool_calls")]


class FakeOpenAI:
    """Replays scripted streamed responses and records each request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return iter(response)


@pytest.fixture
def llm():
    return SimpleNamespace(
        client=FakeOpenAI,
        text=text_reply,
        tool=tool_reply,
        empty_chunk=lambda: SimpleNamespace(choices=[]),
    )


def make_pdf(path: Path, text: str) -> Path:
    """Write a one-page PDF. An empty text gives a page with no text layer."""
    content = f"BT /F1 24 Tf 72 720 Td ({text}) Tj ET".encode() if text else b""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R"
        b" /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    path.write_bytes(out)
    return path


@pytest.fixture
def pdf_factory(tmp_path):
    return lambda name, text: make_pdf(tmp_path / name, text)
