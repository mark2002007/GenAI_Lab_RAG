#!/usr/bin/env python3
"""
retrieve.py

Production-ready retrieval CLI for Vector-RAG system. Embeds a query, searches FAISS index,
optionally de-duplicates by URL, and outputs top-K results.
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import faiss
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Read OpenAI API key from environment
def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set.")
        raise RuntimeError("Missing OpenAI API key.")
    return OpenAI(api_key=api_key)

# Embed the query text

def embed_query(client: OpenAI, text: str, model: str) -> np.ndarray:
    resp = client.embeddings.create(
        model=model,
        input=[text]
    )
    return np.array(resp.data[0].embedding, dtype='float32')

# Load FAISS index

def load_index(path: Path) -> faiss.Index:
    if not path.is_file():
        logger.error("FAISS index file not found: %s", path)
        raise FileNotFoundError(f"Index not found: {path}")
    return faiss.read_index(str(path))

# Load metadata JSON list

def load_metadata(path: Path) -> List[dict]:
    if not path.is_file():
        logger.error("Metadata file not found: %s", path)
        raise FileNotFoundError(f"Metadata not found: {path}")
    return json.loads(path.read_text(encoding='utf-8'))

# Perform search and optional deduplication

def retrieve(
    client: OpenAI,
    index: faiss.Index,
    metadata: List[dict],
    query: str,
    topk: int,
    embed_model: str,
    dedupe: bool = True
) -> List[Tuple[dict, float]]:
    q_emb = embed_query(client, query, embed_model).reshape(1, -1)
    D, I = index.search(q_emb, topk * 2 if dedupe else topk)

    results = []
    seen_urls = set()
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        m = metadata[idx]
        url = m.get('url', '')
        if dedupe and url in seen_urls:
            continue
        seen_urls.add(url)
        results.append((m, float(score)))
        if len(results) >= topk:
            break
    return results

# Output results to console or JSON

def print_results(results: List[Tuple[dict, float]]):
    for rank, (m, score) in enumerate(results, start=1):
        print(f"[{rank}] {m.get('title','No Title')} ({m.get('date','Unknown')}): similarity={1-score:.4f}")
        print(m.get('url',''))
        snippet = m.get('text','')[:200].replace('\n',' ')
        print(f"  {snippet}...\n")


def save_json(results: List[Tuple[dict, float]], path: Path):
    out = []
    for m, score in results:
        entry = {**m, 'score': score}
        out.append(entry)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    logger.info("Saved JSON results to %s", path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Search FAISS index for a query and print top-K results."
    )
    parser.add_argument('--index',  required=True, type=Path, help='Path to FAISS index file')
    parser.add_argument('--meta',   required=True, type=Path, help='Path to metadata JSON')
    parser.add_argument('--query',  required=True, type=str,  help='Search query text')
    parser.add_argument('--topk',   type=int, default=5,        help='Number of results to return')
    parser.add_argument('--model',  type=str, default='text-embedding-ada-002', help='Embedding model')
    parser.add_argument('--no-dedupe', action='store_true', help='Disable URL deduplication')
    parser.add_argument('--output', type=Path, help='Optional path to save JSON results')
    args = parser.parse_args()

    client = get_openai_client()
    index = load_index(args.index)
    metadata = load_metadata(args.meta)

    results = retrieve(
        client=client,
        index=index,
        metadata=metadata,
        query=args.query,
        topk=args.topk,
        embed_model=args.model,
        dedupe=not args.no_dedupe
    )

    if args.output:
        save_json(results, args.output)
    else:
        print_results(results)
