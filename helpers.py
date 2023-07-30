
from random import randint
import tiktoken
from typing import NamedTuple
import logging
import openai
from pathlib import Path
from itertools import islice
from typing import Iterable, Union
import numpy as np
import asyncio
import streamlit as st


EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = "cl100k_base"


class EmbeddingResponse(NamedTuple):
    embedding: list[float]
    total_tokens: int


class DocumentsEntry(NamedTuple):
    content: str
    embedding: list[float]
    metadata: dict


@st.cache_data
def cached_embed(text: str) -> EmbeddingResponse:
    return asyncio.run(get_embedding(text))



# Attempt to call openai Embeddings endpoint with input text or tokens
async def get_embedding(text_or_tokens: Union[str, tuple[int]], retry_attempt=1) -> EmbeddingResponse:
    try:
        response = await openai.Embedding.acreate(input=text_or_tokens, model=EMBEDDING_MODEL)
        embedding = response["data"][0]["embedding"]
        # Allow this to fail gracefully
        total_tokens = response.get("usage", {}).get("total_tokens", 0)
        return EmbeddingResponse(embedding=embedding, total_tokens=total_tokens)
    except openai.APIError as e:
        if retry_attempt >= 5:
            raise e
        asyncio.sleep((randint(50, 100) / 100) + (2**retry_attempt))
        return get_embedding(text_or_tokens, retry_attempt + 1)
    except Exception as e:
        logging.exception(e, stack_info=True)
        raise e


def batched(iterable: Iterable, n: int) -> Iterable[tuple]:
    """
    Batch data into tuples of length n. The last batch may be shorter.
    `batched('ABCDEFG', 3)` --> ABC DEF G
    """
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def chunked_tokens(text: str) -> Iterable[tuple[int]]:
    """Using tokenizer encoder, loops over input text to limit data sent per request"""
    encoding = tiktoken.get_encoding(EMBEDDING_ENCODING)
    tokens = encoding.encode(text)
    chunks_iterator = batched(tokens, EMBEDDING_CTX_LENGTH)
    yield from chunks_iterator


# Gets embeddings for input text. Chunks and averages chunks if necessary
async def embed_long_text(text: str) -> EmbeddingResponse:
    chunk_embeddings = []
    chunk_lens = []
    total_tokens = 0
    for tokens in chunked_tokens(text):
        embedding_response = await get_embedding(tokens)
        chunk_embeddings.append(embedding_response.embedding)
        chunk_lens.append(len(tokens))
        total_tokens = total_tokens + embedding_response.total_tokens

    chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lens)
    chunk_embeddings = chunk_embeddings / np.linalg.norm(
        chunk_embeddings
    )  # normalizes length to 1
    return EmbeddingResponse(embedding=chunk_embeddings, total_tokens=total_tokens)

async def embed_document(file_path: Path) -> DocumentsEntry:
    file_text = file_path.read_text()
    embedding_response = await embed_long_text(file_text)
    return DocumentsEntry(
        content=file_text,
        embedding=embedding_response.embedding,
        metadata={
            "file_name": file_path.name,
            "file_path": str(file_path),
            "total_tokens": embedding_response.total_tokens,
            "file_length": len(file_text),
        },
    )

async def gather_with_concurrency(n, *coros):
    semaphore = asyncio.Semaphore(n)
    async def sem_coro(coro):
        async with semaphore:
            return await coro
    return await asyncio.gather(*(sem_coro(c) for c in coros))

async def embed_directory(directory_path: str, num_workers=3, return_batch_size=10) -> Iterable[tuple[DocumentsEntry]]:
    tasks = [
        asyncio.ensure_future(embed_document(file_path))
        for file_path in Path(directory_path).rglob("*.md")
    ]
    results = await gather_with_concurrency(num_workers, *tasks)
    return batched(results, return_batch_size)

def count_directory_tokens(directory_path: str) -> int:
    total = 0
    for file_path in Path(directory_path).rglob("*.md"):
        encoding = tiktoken.get_encoding(EMBEDDING_ENCODING)
        tokens = encoding.encode(file_path.read_text())
        total = total + len(tokens)
    return total
