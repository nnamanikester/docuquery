import typing
from typing import ClassVar

import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from docuquery.constants import DOCUQUERY_POSTGRES_URL, EMBEDDING_DIMENSION

DB_SQL_QUERY = f"""
-- 1. Enable the vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create the documents table
CREATE TABLE IF NOT EXISTS document_chunks (
    id BIGSERIAL PRIMARY KEY,
    file_name TEXT NOT NULL,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    embedding vector({EMBEDDING_DIMENSION})
);

-- 3. Create an HNSW index for lightning-fast vector searches
CREATE INDEX IF NOT EXISTS idx_chunks_hnsw
ON document_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

-- 4. Enable trigram matching and index file names for fuzzy name search
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE INDEX IF NOT EXISTS idx_chunks_file_name_trgm
ON document_chunks
USING gin (file_name gin_trgm_ops);

-- 5. Conversations and their messages (stored in OpenAI message format)
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS conversation_messages (
    id BIGSERIAL PRIMARY KEY,
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    position INT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
    content TEXT,
    tool_calls JSONB,
    tool_call_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (conversation_id, position)
);
"""


class Database:
    MAX_PAGE_SIZE: ClassVar[int] = 100

    def __get_connection(self) -> psycopg.Connection:
        # Connect using standard environment variable DOCUQUERY_POSTGRES_URL
        conn = psycopg.connect(DOCUQUERY_POSTGRES_URL)
        # register_vector needs the extension to exist, so it can't run before init
        register_vector(conn)
        return conn

    def initialize_database(self) -> None:
        with psycopg.connect(DOCUQUERY_POSTGRES_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(DB_SQL_QUERY)
            conn.commit()

    def replace_document(
        self, file_name: str, chunks: list[str], embeddings: list[list[float]]
    ) -> None:
        with self.__get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM document_chunks WHERE file_name = %s", (file_name,)
                )
                cur.executemany(
                    "INSERT INTO document_chunks (file_name, chunk_index, content, embedding) VALUES (%s, %s, %s, %s)",
                    [
                        (file_name, i, c, e)
                        for i, (c, e) in enumerate(zip(chunks, embeddings, strict=True))
                    ],
                )
            conn.commit()

    def search_similar_chunks(
        self, query_embedding: list[float], limit: int = 3
    ) -> list[dict]:
        with self.__get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT file_name, content, 1 - (embedding <=> %s) AS similarity
                FROM document_chunks
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (query_embedding, query_embedding, limit),
            )
            rows = cur.fetchall()
            return [
                {"file_name": r[0], "content": r[1], "similarity": float(r[2])}
                for r in rows
            ]

    # Allowed sort keys mapped to SQL, so user input is never put into the query directly
    DOCUMENT_SORT_COLUMNS: ClassVar[dict[str, str]] = {
        "file_name": "file_name",
        "chunk_count": "chunk_count",
        "total_characters": "total_characters",
        "relevance": "relevance",
    }

    def __build_documents_query(
        self,
        file_name: str | None = None,
        name_contains: str | None = None,
        content_contains: str | None = None,
        query_embedding: list[float] | None = None,
        min_similarity: float = 0.3,
    ) -> tuple[str, list]:
        """Build the grouped documents query shared by list_documents and count_documents.

        With `query_embedding`, a document matches when any of its chunks has a cosine
        similarity of at least `min_similarity`, and `relevance` is its best chunk's similarity.
        """
        select_params = []
        where, where_params = [], []
        having, having_params = [], []

        if query_embedding is not None:
            relevance = "MAX(1 - (embedding <=> %s::vector))"
            select_params.append(query_embedding)
            # Distance is 1 - similarity, so similarity >= min means distance <= 1 - min
            where.append(
                "file_name IN (SELECT file_name FROM document_chunks WHERE embedding <=> %s::vector <= %s)"
            )
            where_params += [query_embedding, 1 - min_similarity]
        else:
            relevance = "NULL::float"

        if file_name is not None:
            where.append("file_name = %s")
            where_params.append(file_name)
        if name_contains is not None:
            where.append("file_name ILIKE %s")
            where_params.append(f"%{name_contains}%")
        if content_contains is not None:
            # Match documents with any chunk containing the text, but still count all their chunks
            where.append(
                "file_name IN (SELECT file_name FROM document_chunks WHERE content ILIKE %s)"
            )
            where_params.append(f"%{content_contains}%")

        query = f"""
        SELECT
            file_name,
            COUNT(*) AS chunk_count,
            SUM(LENGTH(content)) AS total_characters,
            {relevance} AS relevance
        FROM document_chunks
        """  # noqa: S608
        if where:
            query += " WHERE " + " AND ".join(where)
        query += " GROUP BY file_name"
        if having:
            query += " HAVING " + " AND ".join(having)

        return query, select_params + where_params + having_params

    def list_documents(
        self,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "file_name",
        descending: bool = False,
        file_name: str | None = None,
        name_contains: str | None = None,
        content_contains: str | None = None,
        query_embedding: list[float] | None = None,
        min_similarity: float = 0.3,
    ) -> list[dict]:
        if page < 1:
            raise ValueError("page must be 1 or greater")
        if not 1 <= page_size <= self.MAX_PAGE_SIZE:
            raise ValueError(f"page_size must be between 1 and {self.MAX_PAGE_SIZE}")
        if sort_by not in self.DOCUMENT_SORT_COLUMNS:
            raise ValueError(
                f"sort_by must be one of {list(self.DOCUMENT_SORT_COLUMNS)}"
            )
        if sort_by == "relevance" and query_embedding is None:
            raise ValueError("sort_by='relevance' requires query_embedding")

        query, params = self.__build_documents_query(
            file_name,
            name_contains,
            content_contains,
            query_embedding,
            min_similarity,
        )
        direction = "DESC" if descending else "ASC"
        # file_name as a tie-breaker keeps page boundaries stable
        query += f" ORDER BY {self.DOCUMENT_SORT_COLUMNS[sort_by]} {direction}, file_name ASC"
        query += " LIMIT %s OFFSET %s"
        params += [page_size, (page - 1) * page_size]

        with self.__get_connection() as conn, conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
            return [
                {
                    "file_name": r[0],
                    "chunk_count": r[1],
                    "total_characters": r[2],
                    "relevance": None if r[3] is None else float(r[3]),
                }
                for r in rows
            ]

    @typing.override
    def count_documents(
        self,
        file_name: str | None = None,
        name_contains: str | None = None,
        content_contains: str | None = None,
        query_embedding: list[float] | None = None,
        min_similarity: float = 0.3,
    ) -> int:
        query, params = self.__build_documents_query(
            file_name,
            name_contains,
            content_contains,
            query_embedding,
            min_similarity,
        )
        with self.__get_connection() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM ({query}) AS documents", params)  # noqa: S608
            return cur.fetchone()[0]

        with self.__get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    file_name,
                    COUNT(*) AS chunk_count,
                    SUM(LENGTH(content)) AS total_characters,
                    json_agg(
                    json_build_object('chunk_index', chunk_index, 'content', content)
                    ORDER BY chunk_index
                    ) AS chunks
                FROM document_chunks
                WHERE file_name ILIKE %(contains)s OR %(name)s <%% file_name
                GROUP BY file_name
                ORDER BY
                    LOWER(file_name) = LOWER(%(name)s) DESC,
                    file_name ILIKE %(prefix)s DESC,
                    file_name ASC
                LIMIT %(limit)s
                """,
                params,
            )
            rows = cur.fetchall()
            return [
                {
                    "file_name": r[0],
                    "chunk_count": r[1],
                    "total_characters": r[2],
                    "chunks": r[3],
                }
                for r in rows
            ]

    def create_conversation(self, title: str | None = None) -> str:
        with self.__get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO conversations (title) VALUES (%s) RETURNING id",
                    (title,),
                )
                conversation_id = cur.fetchone()[0]
            conn.commit()
        return str(conversation_id)

    def add_messages(self, conversation_id: str, messages: list[dict]) -> None:
        """Append OpenAI-format messages to a conversation, after any already saved."""
        with self.__get_connection() as conn:
            with conn.cursor() as cur:
                # Lock the conversation so concurrent appends can't take the same positions
                cur.execute(
                    "UPDATE conversations SET updated_at = now() WHERE id = %s RETURNING id",
                    (conversation_id,),
                )
                if cur.fetchone() is None:
                    raise ValueError(f"Conversation {conversation_id} not found")

                cur.execute(
                    "SELECT COALESCE(MAX(position), -1) FROM conversation_messages WHERE conversation_id = %s",
                    (conversation_id,),
                )
                next_position = cur.fetchone()[0] + 1

                cur.executemany(
                    """
                    INSERT INTO conversation_messages
                        (conversation_id, position, role, content, tool_calls, tool_call_id)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            conversation_id,
                            next_position + i,
                            m["role"],
                            m.get("content"),
                            Jsonb(m["tool_calls"]) if m.get("tool_calls") else None,
                            m.get("tool_call_id"),
                        )
                        for i, m in enumerate(messages)
                    ],
                )
            conn.commit()

    def get_conversation_by_id(self, conversation_id: str) -> dict | None:
        """Return the conversation with its messages in order, ready to pass back to OpenAI."""
        with self.__get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    c.id,
                    c.title,
                    c.created_at,
                    c.updated_at,
                    COALESCE(
                    json_agg(
                        json_strip_nulls(json_build_object(
                        'role', m.role,
                        'content', m.content,
                        'tool_calls', m.tool_calls,
                        'tool_call_id', m.tool_call_id
                        ))
                        ORDER BY m.position
                    ) FILTER (WHERE m.id IS NOT NULL),
                    '[]'
                    ) AS messages
                FROM conversations c
                LEFT JOIN conversation_messages m ON m.conversation_id = c.id
                WHERE c.id = %s
                GROUP BY c.id
                """,
                (conversation_id,),
            )
            row = cur.fetchone()

        if row is None:
            return None
        return {
            "id": str(row[0]),
            "title": row[1],
            "created_at": row[2],
            "updated_at": row[3],
            "messages": row[4],
        }

    def list_conversations(self, page: int = 1, page_size: int = 20) -> list[dict]:
        """List conversations, most recently active first."""
        if page < 1:
            raise ValueError("page must be 1 or greater")
        if not 1 <= page_size <= self.MAX_PAGE_SIZE:
            raise ValueError(f"page_size must be between 1 and {self.MAX_PAGE_SIZE}")
        with self.__get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, title
                    FROM conversations
                    ORDER BY updated_at DESC, id
                    LIMIT %s OFFSET %s
                    """,
                    (page_size, (page - 1) * page_size),
                )
                rows = cur.fetchall()
            return [{"id": str(r[0]), "title": r[1]} for r in rows]
