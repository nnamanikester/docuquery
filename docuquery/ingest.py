from pathlib import Path

import click
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from docuquery.chunker import recursive_semantic_chunking
from docuquery.constants import SENTENCE_TRANSFORMER_MODEL
from docuquery.database import Database


class DocumentIngestor:
    def __init__(self) -> None:
        self.embedder = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
        self.db = Database()

    def load_text(self, path: str) -> str:
        if Path(path).suffix.lower() == ".pdf":
            reader = PdfReader(path)
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)
        try:
            return Path(path).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise click.ClickException(
                f"'{path}' is not a PDF or a UTF-8 text file"
            ) from None

    def process_and_store_document(self, document_path: str) -> None:
        """Process a document, chunk it, and store in database."""
        click.echo(f"[*] Processing document: {document_path}")

        content = self.load_text(document_path)

        file_name = Path(document_path).name

        if not content.strip():
            raise click.ClickException(
                f"'{document_path}' has no text. If it is a scanned PDF, use OCR first."
            )

        click.echo("[*] Chunking content...")

        # Chunk the content
        chunks = recursive_semantic_chunking(content)

        click.echo(f"[*] Ingesting '{file_name}': Split into {len(chunks)} chunks.")
        click.echo(f"[*] Generated {len(chunks)} chunks.")

        embeddings = self.embedder.encode(chunks, show_progress_bar=True)
        self.db.replace_document(file_name, chunks, embeddings)

        click.secho(f"[+] Successfully indexed {file_name}!", fg="green")
