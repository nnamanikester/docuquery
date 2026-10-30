import logging
import os
import uuid

import click


@click.group()
@click.version_option(package_name="docuquery")
def cli() -> None:
    from docuquery.utils import ensure_env_variables

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    # PyTorch warns on macOS that a distributed-training feature is missing.
    # DocuQuery does not use it, so hide the warning.
    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(
        logging.ERROR
    )
    ensure_env_variables()
    pass


@click.command(name="init")
def init() -> None:
    """Initializes the local environment: creates tables in the Postgres database."""
    # Imported here: constants.py reads the environment on import, so it must
    # run after ensure_env_variables() has loaded the config file.
    from docuquery.database import Database

    try:
        click.echo("Initializing database...", color=True)
        db = Database()
        db.initialize_database()
        click.secho("✔ Database initialized successfully.", fg="green")
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Could not initialize database: {e}") from e


@click.command(name="ingest")
@click.argument(
    "file_path",
    type=click.Path(exists=True),
    required=True,
    help="The path to the document file to be ingested. You can drag and drop the file here.",
)
def ingest(file_path: str) -> None:
    """Ingest a document from the specified file path."""
    from docuquery.ingest import DocumentIngestor

    try:
        ingestor = DocumentIngestor()
        ingestor.process_and_store_document(file_path)
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Could not ingest the document: {e}") from e


@click.command(name="ask")
@click.argument(
    "question", required=False, help="Start the conversation with this question."
)
@click.option(
    "--session-id", required=False, help="Resume a saved conversation by session ID."
)
def ask(question: str, session_id: str) -> None:
    """Chat with your documents. Type 'exit' or press Ctrl-D to quit."""
    from docuquery.rag import RAGBase

    rag = RAGBase()

    if session_id:
        try:
            uuid.UUID(session_id)
        except ValueError:
            raise click.ClickException(
                "The session could not be loaded. Invalid session ID."
            ) from None
        try:
            session = rag.load_conversation(session_id)
            click.echo(f'Resumed session: {session_id} - Title: "{session["title"]}"')
        except click.ClickException:
            raise
        except ValueError:
            raise click.ClickException(
                "The session could not be loaded. Session does not exist."
            ) from None

    title = question
    try:
        while True:
            if not question:
                question = click.prompt(
                    click.style("You", fg="cyan"), prompt_suffix=": "
                ).strip()
            if question.lower() in ("exit", "quit"):
                break
            if question:
                title = title or question
                click.secho("Answer: ", nl=False, fg="green")
                try:
                    for token in rag.query_rag_stream(question):
                        click.secho(token, nl=False)
                    # Save conversation
                    session_id = rag.save_conversation(title=title[:50])
                except Exception as e:
                    click.secho(
                        f"❌ Error: Could not generate an answer: {e}",
                        err=True,
                        fg="red",
                    )
                click.echo("\n")
            question = None
    except (KeyboardInterrupt, EOFError, click.Abort):
        pass
    click.echo("\nGoodbye!")


@click.command(name="sessions")
def list_sessions() -> None:
    """List all saved conversation sessions."""
    from docuquery.rag import RAGBase

    rag = RAGBase()
    try:
        sessions = rag.list_sessions()
        if not sessions:
            click.echo("No sessions found.")
        for session in sessions:
            click.echo(
                f'Session ID: {session["id"]} - Title: "{session.get("title", "Untitled")}"'
            )
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Could not list sessions: {e}") from e


@click.command(name="resume")
@click.argument("session_id", required=True, help="The session ID to resume.")
@click.pass_context
def resume_session(ctx: click.Context, session_id: str) -> None:
    """Resume a saved conversation by session ID."""
    ctx.invoke(ask, question=None, session_id=session_id)


cli.add_command(init)
cli.add_command(ingest)
cli.add_command(ask)
cli.add_command(list_sessions)
cli.add_command(resume_session)

if __name__ == "__main__":
    cli()
