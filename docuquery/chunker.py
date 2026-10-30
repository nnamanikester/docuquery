from docuquery.constants import CHUNK_SIZE, OVERLAP


def split_text(text_block: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Splits a text block into chunks based on the specified separators and chunk size.

    Args:
        text_block (str): The text block to be split.
        chunk_size (int): The maximum size of each chunk.
        separators (list[str]): A list of separators to use for splitting.

    Returns:
        list[str]: A list of text chunks.
    """
    if len(text_block) <= chunk_size:
        return [text_block]

    for i, separator in enumerate(separators):
        if separator not in text_block:
            continue
        chunks, current = [], ""

        for part in text_block.split(separator):
            if len(part) > chunk_size:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(split_text(part, chunk_size, separators[i + 1 :]))
            elif not current:
                current = part
            elif len(current) + len(separator) + len(part) <= chunk_size:
                current += separator + part
            else:
                chunks.append(current)
                current = part
        if current:
            chunks.append(current)
        return chunks

    # No separator left: cut at chunk_size, so no text is lost
    return [
        text_block[i : i + chunk_size] for i in range(0, len(text_block), chunk_size)
    ]


def overlap_chunked_texts(
    text_chunks: list[str], overlap: int, chunk_size: int
) -> list[str]:
    """Applies overlap to a list of text chunks.

    Args:
        text_chunks (list[str]): The list of text chunks.
        overlap (int): The number of overlapping characters between consecutive chunks.
        chunk_size (int): The maximum size of each chunk.

    Returns:
        list[str]: A list of text chunks with applied overlap.
    """
    overlapped_chunks = []
    for i, text_chunk in enumerate(text_chunks):
        chunk = text_chunk
        if overlap and i > 0 and len(text_chunk) + overlap <= chunk_size:
            overlap_text_chunk = text_chunks[i - 1][-overlap:]
            chunk = overlap_text_chunk + text_chunk
        overlapped_chunks.append(chunk)
    return overlapped_chunks


def recursive_semantic_chunking(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP
) -> list[str]:
    """Recursively chunks the input text into smaller segments based on semantic separators.
    If a chunk exceeds the specified chunk size, it will be further split using the next available separator.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int, optional): The maximum size of each chunk. Defaults to CHUNK_SIZE.
        overlap (int, optional): The number of overlapping characters between consecutive chunks. Defaults to OVERLAP.

    Returns:
        list[str]: A list of text chunks.
    """
    separators = ["\n\n", "\n", ".", "!", "?", ",", " "]

    text_chunks = split_text(text, chunk_size, separators)

    return overlap_chunked_texts(text_chunks, overlap, chunk_size)
