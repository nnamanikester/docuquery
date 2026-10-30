from unittest.mock import MagicMock

import click
import pytest

import docuquery.ingest
from docuquery.ingest import DocumentIngestor


@pytest.fixture
def ingestor(monkeypatch, fake_embedder):
    monkeypatch.setattr(
        docuquery.ingest, "SentenceTransformer", lambda *_a, **_k: fake_embedder
    )
    monkeypatch.setattr(docuquery.ingest, "Database", MagicMock)
    return DocumentIngestor()


def test_reads_a_text_file(ingestor, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Alpha beta gamma.", encoding="utf-8")
    assert ingestor.load_text(str(path)) == "Alpha beta gamma."


@pytest.mark.parametrize("name", ["readme.md", "data.csv", "script.py", "page.html"])
def test_reads_any_utf8_text_file(ingestor, tmp_path, name):
    path = tmp_path / name
    path.write_text("Text with ünïcode ✓", encoding="utf-8")
    assert ingestor.load_text(str(path)) == "Text with ünïcode ✓"


def test_reads_the_text_of_a_pdf(ingestor, pdf_factory):
    path = pdf_factory("report.pdf", "Hello PDF world")
    assert "Hello PDF world" in ingestor.load_text(str(path))


def test_pdf_extension_is_not_case_sensitive(ingestor, pdf_factory):
    path = pdf_factory("REPORT.PDF", "Upper case name")
    assert "Upper case name" in ingestor.load_text(str(path))


def test_rejects_a_binary_file(ingestor, tmp_path):
    path = tmp_path / "file.docx"
    path.write_bytes(b"\xff\xfe\x00binary")
    with pytest.raises(click.ClickException, match="is not a PDF or a UTF-8 text file"):
        ingestor.load_text(str(path))


def test_stores_the_chunks_under_the_plain_file_name(ingestor, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Alpha beta gamma.\n\nThe secret code is 4417.", encoding="utf-8")

    ingestor.process_and_store_document(str(path))

    ingestor.db.replace_document.assert_called_once()
    file_name, chunks, embeddings = ingestor.db.replace_document.call_args.args
    assert file_name == "notes.txt"  # no "document_" prefix
    assert chunks == ["Alpha beta gamma.\n\nThe secret code is 4417."]
    assert len(embeddings) == len(chunks)


def test_embeds_all_chunks_in_one_batch(ingestor, tmp_path, fake_embedder):
    path = tmp_path / "long.txt"
    path.write_text("x" * 2000, encoding="utf-8")

    ingestor.process_and_store_document(str(path))

    assert len(fake_embedder.calls) == 1
    _, chunks, _ = ingestor.db.replace_document.call_args.args
    assert max(len(c) for c in chunks) <= 512


@pytest.mark.parametrize("content", ["", "   \n\n\t  "])
def test_rejects_a_file_with_no_text(ingestor, tmp_path, content):
    path = tmp_path / "empty.txt"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(click.ClickException, match="has no text"):
        ingestor.process_and_store_document(str(path))
    ingestor.db.replace_document.assert_not_called()


def test_rejects_a_scanned_pdf(ingestor, pdf_factory):
    path = pdf_factory("scan.pdf", "")  # a page with no text layer
    with pytest.raises(click.ClickException, match="use OCR first"):
        ingestor.process_and_store_document(str(path))
    ingestor.db.replace_document.assert_not_called()
