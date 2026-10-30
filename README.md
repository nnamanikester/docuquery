# DocuQuery

**Ask questions about your documents from the terminal.**

[![PyPI](https://img.shields.io/pypi/v/docuquery)](https://pypi.org/project/docuquery/)
[![Python](https://img.shields.io/pypi/pyversions/docuquery)](https://pypi.org/project/docuquery/)
[![License: MIT](https://img.shields.io/pypi/l/docuquery)](https://opensource.org/licenses/MIT)

DocuQuery reads your PDF and text files and keeps their text in a PostgreSQL database.
You ask a question in plain language. An OpenAI model finds the related text and writes the answer.

```console
$ docuquery ingest lease.pdf
[+] Successfully indexed lease.pdf!

$ docuquery ask "When does my lease end?"
Answer: Your lease ends on 31 August 2027.
```

## Contents

1. [Quick start](#1-quick-start)
2. [Install](#2-install)
3. [Set up the database](#3-set-up-the-database)
4. [Configure](#4-configure)
5. [Add documents](#5-add-documents)
6. [Ask questions](#6-ask-questions)
7. [Sessions](#7-sessions)
8. [Commands](#8-commands)
9. [Problems and solutions](#9-problems-and-solutions)
10. [Privacy](#10-privacy)
11. [Development](#11-development)
12. [License](#12-license)

---

## 1. Quick start

You need Docker and an [OpenAI API key](https://platform.openai.com/api-keys).

1. Install DocuQuery:

   ```bash
   uv tool install docuquery
   ```

2. Start a PostgreSQL database with pgvector:

   ```bash
   docker run -d --name docuquery-db --restart unless-stopped \
     -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=docuquery_db \
     -p 5432:5432 -v docuquery-data:/var/lib/postgresql/data \
     pgvector/pgvector:pg17
   ```

3. Set your settings:

   ```bash
   export DOCUQUERY_POSTGRES_URL="postgresql://postgres:postgres@localhost:5432/docuquery_db"
   export OPENAI_API_KEY="sk-your-key"
   ```

4. Prepare the database:

   ```bash
   docuquery init
   ```

5. Add a document:

   ```bash
   docuquery ingest ~/Documents/report.pdf
   ```

6. Ask a question:

   ```bash
   docuquery ask "What are the main points in the report?"
   ```

> **NOTE:** The first time DocuQuery adds a document or searches your documents, it downloads a small embedding model (approximately 90 MB). This occurs one time only.

---

## 2. Install

DocuQuery needs Python 3.12 or later. Use one of these commands:

```bash
uv tool install docuquery      # recommended
pipx install docuquery
pip install docuquery
```

If you do not have uv, read the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

To make sure that the installation is correct, run:

```bash
docuquery --help
```

If the shell cannot find `docuquery`, run `uv tool update-shell` (or `pipx ensurepath`). Then open a new terminal.

> **NOTE:** If you install with `pip` on Linux and you do not have an NVIDIA GPU, install the CPU version of PyTorch first. This makes the download much smaller:
>
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install docuquery
> ```

---

## 3. Set up the database

DocuQuery needs PostgreSQL 13 or later with the **pgvector** (0.5.0 or later) and **pg_trgm** extensions. Use one of these options.

### 3.1 Docker (on your computer)

1. Make a file with the name `docker-compose.yml`. Put this text in the file:

   ```yaml
   services:
     postgres:
       image: pgvector/pgvector:pg17
       restart: unless-stopped
       environment:
         POSTGRES_USER: postgres
         POSTGRES_PASSWORD: postgres
         POSTGRES_DB: docuquery_db
       ports:
         - "5432:5432"
       volumes:
         - pgdata:/var/lib/postgresql/data

   volumes:
     pgdata:
   ```

2. In the same folder, start the database:

   ```bash
   docker compose up -d
   ```

Your connection URL is:

```text
postgresql://postgres:postgres@localhost:5432/docuquery_db
```

> **NOTE:** If port 5432 is already in use, change `"5432:5432"` to `"5433:5432"`. Then use port `5433` in the URL.

### 3.2 Hosted database (online)

These providers supply PostgreSQL with pgvector. Many have a free plan.

| Provider | Where to find the connection URL |
|---|---|
| [Neon](https://neon.tech) | Project dashboard > **Connect** |
| [Supabase](https://supabase.com) | Project > **Connect** > **Session pooler** |
| AWS RDS, Google Cloud SQL, Azure | Use PostgreSQL 16 or later. On Azure, allow `VECTOR` and `PG_TRGM` in `azure.extensions`. |

If the URL does not contain `sslmode`, add `?sslmode=require` to the end.

> **NOTE:** `docuquery init` creates the `vector` and `pg_trgm` extensions. The database user must have permission to create extensions. On hosted providers, the default admin user usually has this permission.

### 3.3 Connection URL format

```text
postgresql://USER:PASSWORD@HOST:PORT/DATABASE
```

If the password has special characters, encode them. For example, `@` becomes `%40`.

---

## 4. Configure

### 4.1 Settings

DocuQuery reads its settings from a config file and from environment variables.

| Variable | Required | Default |
|---|---|---|
| `DOCUQUERY_POSTGRES_URL` | Yes | `postgresql://postgres:postgres@localhost:5432/docuquery_db` |
| `OPENAI_API_KEY` | Yes | — |
| `DOCUQUERY_OPENAI_MODEL` | No | `gpt-5.4-mini` |
| `SENTENCE_TRANSFORMER_MODEL` | No | `sentence-transformers/all-MiniLM-L6-v2` |
| `DOCUQUERY_CHUNK_SIZE` | No | `512` (characters) |
| `DOCUQUERY_OVERLAP` | No | `50` (characters) |

If a setting is not set, DocuQuery asks for it one time. For an optional setting, push **Enter** to use the default.

DocuQuery saves your answers in a config file:

| System | Config file |
|---|---|
| macOS | `~/Library/Application Support/docuquery/config.env` |
| Linux | `~/.config/docuquery/config.env` |
| Windows | `%APPDATA%\docuquery\config.env` |

To change a saved setting, edit this file. The file contains your API key. On macOS and Linux, DocuQuery makes the file readable only by you.

### 4.2 Keep the settings

You can also set the settings as environment variables. Add them to your shell file. For zsh, the file is `~/.zshrc`. For bash, the file is `~/.bashrc`.

```bash
export DOCUQUERY_POSTGRES_URL="postgresql://postgres:postgres@localhost:5432/docuquery_db"
export OPENAI_API_KEY="sk-your-key"
export DOCUQUERY_OPENAI_MODEL="gpt-5.4-mini"
export SENTENCE_TRANSFORMER_MODEL="sentence-transformers/all-MiniLM-L6-v2"
export DOCUQUERY_CHUNK_SIZE="512"
export DOCUQUERY_OVERLAP="50"
```

Then open a new terminal. On Windows, use `setx NAME "value"` for each setting.

An environment variable has priority over the config file.

> **CAUTION:** Do not put your API key in a Git repository.

> **CAUTION:** Use only an embedding model that makes 384 dimensions. If you change the embedding model, add your documents again.

### 4.3 Prepare the database

Do this one time for each new database:

```bash
docuquery init
```

You see `✔ Database initialized successfully.` It is safe to run this command again.

---

## 5. Add documents

```bash
docuquery ingest path/to/file.pdf
```

DocuQuery reads **PDF files** and **UTF-8 text files** (`.txt`, `.md`, `.csv`, and source code). For a Word file, save it as PDF first.

To add all the PDF files in a folder:

```bash
for f in ~/Documents/*.pdf; do docuquery ingest "$f"; done
```

If you add a file with the same name again, DocuQuery replaces the old copy. This also occurs for two different files that have the same name.

> **NOTE:** A scanned PDF contains only images. DocuQuery cannot read it. Use an OCR tool, for example [OCRmyPDF](https://ocrmypdf.readthedocs.io/), first.

---

## 6. Ask questions

Start a conversation with a question:

```bash
docuquery ask "What is the cancellation policy?"
```

After each answer, type your next question. The model remembers the conversation.

```text
$ docuquery ask "Which documents mention a deposit?"
Answer: lease.pdf and booking.pdf mention a deposit.

You: How much is the deposit in the lease?
Answer: The deposit is 1,200 EUR.

You: exit
```

To stop, type `exit` or push **Ctrl+D**.

### Examples of questions

- "What is the due date on the March invoice?"
- "Give me a summary of the project plan in 5 points."
- "Compare the salaries in the two job offers."
- "Which documents do I have?"
- "What did we talk about in my last session?"

### Get better answers

- Be specific. "What is the notice period in the lease?" is better than "Tell me about notice".
- If you know the document, say its name.
- Ask one thing at a time.

---

## 7. Sessions

DocuQuery saves each conversation after each answer.

Show your 20 most recent sessions:

```bash
docuquery sessions
```

Continue a session:

```bash
docuquery resume 3f6c2a1e-8d4b-4c1a-9e2f-7b5d0a9c1e33
```

---

## 8. Commands

| Command | What it does |
|---|---|
| `docuquery init` | Prepares the database. |
| `docuquery ingest FILE_PATH` | Adds a document. |
| `docuquery ask [QUESTION]` | Starts a conversation. |
| `docuquery ask --session-id ID` | Continues a session. |
| `docuquery sessions` | Shows your sessions. |
| `docuquery resume SESSION_ID` | Continues a session. |

Run `docuquery COMMAND --help` for more information.

---

## 9. Problems and solutions

| Problem | Solution |
|---|---|
| `docuquery: command not found` | Run `uv tool update-shell`. Then open a new terminal. |
| A setting is incorrect. | Edit the config file (read [4.1](#41-settings)), or set the environment variable (read [4.2](#42-keep-the-settings)). |
| `connection refused` | Start the database. Make sure that the host and port in the URL are correct. |
| `password authentication failed` | Correct the user and password in the URL. |
| `SSL connection is required` | Add `?sslmode=require` to the end of the URL. |
| `extension "vector" is not available` | Use the `pgvector/pgvector` Docker image, or a provider that supports pgvector. |
| `relation "document_chunks" does not exist` | Run `docuquery init`. |
| `port is already allocated` | Use port 5433. Read [3.1](#31-docker-on-your-computer). |
| `401` or `Incorrect API key` | Make a new key at <https://platform.openai.com/api-keys>. |
| `429` or `insufficient_quota` | Add credit to your OpenAI account. |
| `is not a PDF or a UTF-8 text file` | The file type is not supported. Save it as PDF. |
| `has no text` | The file is empty, or it is a scanned PDF. Use OCR first. |
| `[Stopped: reached max tool-call turns]` | The model needed more than 3 rounds of searches. Ask a more specific question. |

---

## 10. Privacy

Your documents stay in your database. The embeddings are made on your computer.

When you ask a question, DocuQuery sends these items to OpenAI:

- Your question and the previous messages in the conversation.
- The parts of your documents that are related to the question.
- The names of your documents, and messages from your past sessions, if the model asks for them.

> **CAUTION:** Do not add documents that you cannot share with OpenAI.

---

## 11. Development

1. Get the source code:

   ```bash
   git clone https://github.com/nnamanikester/docuquery.git docuquery
   cd docuquery
   ```

2. Install the dependencies:

   ```bash
   uv sync
   ```

3. Activate the virtual environment:

   ```bash
   source .venv/bin/activate        # macOS and Linux
   .venv\Scripts\activate           # Windows
   ```

4. Run DocuQuery:

   ```bash
   docuquery --help
   ```

5. Before you commit, run the tests, then check and format the code:

   ```bash
   pytest
   ruff check --fix
   ruff format
   ```

The database tests run only when `DOCUQUERY_TEST_POSTGRES_URL` is set. To run them, start a separate test database:

```bash
docker run -d --rm --name docuquery-test -e POSTGRES_PASSWORD=test \
  -p 55432:5432 pgvector/pgvector:pg17
DOCUQUERY_TEST_POSTGRES_URL=postgresql://postgres:test@localhost:55432/postgres pytest
```

> **CAUTION:** The database tests delete all DocuQuery tables in the test database. Do not use a database that contains data that you want to keep.

---

## 12. License

DocuQuery is available under the MIT License. Read the `LICENSE` file.
