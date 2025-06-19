#!/usr/bin/env python3
"""
embed_and_index.py

Production-ready embedding and indexing script:
- Reads cleaned chunks JSON
- Embeds text in batches via OpenAI
- Builds a FAISS index (FlatL2 or IVFFlat)
- Saves index and metadata to disk
"""
import os
import json
import logging
from pathlib import Path
from typing import List

import numpy as np
import faiss
from tqdm import tqdm
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable is not set.")
        raise RuntimeError("Missing OpenAI API key.")
    return OpenAI(api_key=api_key)


def embed_texts(
    client: OpenAI,
    texts: List[str],
    model: str,
    batch_size: int
) -> np.ndarray:
    """
    Embed a list of texts in batches using the specified OpenAI model.
    Returns a numpy array of shape (len(texts), embedding_dim).
    """
    embeddings: List[List[float]] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts", unit="batch"):
        batch = texts[i : i + batch_size]
        try:
            resp = client.embeddings.create(model=model, input=batch)
            embeddings.extend([d.embedding for d in resp.data])
        except Exception as e:
            logger.error("Embedding batch %d-%d failed: %s", i, i+len(batch), e)
            raise
    emb_array = np.array(embeddings, dtype="float32")
    logger.info("Embedded %d texts into array of shape %s", emb_array.shape[0], emb_array.shape)
    return emb_array


def build_faiss_index(
    embeddings: np.ndarray,
    use_ivf: bool = False,
    n_list: int = 100
) -> faiss.Index:
    """
    Builds and returns a FAISS index. If use_ivf is True, creates an IVF index.
    """
    dim = embeddings.shape[1]
    if use_ivf:
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, n_list, faiss.METRIC_L2)
        index.train(embeddings)
        index.add(embeddings)
        logger.info("Built IVFFlat index with %d lists", n_list)
    else:
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        logger.info("Built FlatL2 index of dimension %d", dim)
    return index


def save_index(index: faiss.Index, path: Path) -> None:
    """
    Writes FAISS index to disk.
    """
    faiss.write_index(index, str(path))
    logger.info("Saved FAISS index to %s", path)


def save_metadata(data: List[dict], path: Path) -> None:
    """
    Writes metadata list to JSON file.
    """
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    logger.info("Saved metadata to %s", path)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Embed text chunks and build a FAISS index."
    )
    parser.add_argument(
        "--chunks", required=True, type=Path,
        help="Path to chunks.json (list of dicts with 'text' field)"
    )
    parser.add_argument(
        "--outdir", required=True, type=Path,
        help="Directory where index and metadata will be saved"
    )
    parser.add_argument(
        "--model", default="text-embedding-ada-002",
        help="OpenAI embedding model"
    )
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="Number of texts to embed per API call"
    )
    parser.add_argument(
        "--use-ivf", action="store_true",
        help="Use an IVF index instead of FlatL2"
    )
    parser.add_argument(
        "--n-list", type=int, default=100,
        help="Number of Voronoi cells for IVF index"
    )
    args = parser.parse_args()

    client = get_openai_client()

    # Load chunks
    if not args.chunks.is_file():
        logger.error("Chunks file not found: %s", args.chunks)
        return
    data = json.loads(args.chunks.read_text(encoding="utf-8"))
    texts = [d.get("text", "") for d in data]

    args.outdir.mkdir(parents=True, exist_ok=True)
    index_path = args.outdir / "thebatch.faiss"
    meta_path = args.outdir / "meta.json"

    # Embed
    embeddings = embed_texts(
        client=client,
        texts=texts,
        model=args.model,
        batch_size=args.batch_size
    )

    # Build
    index = build_faiss_index(
        embeddings,
        use_ivf=args.use_ivf,
        n_list=args.n_list
    )

    # Save
    save_index(index, index_path)
    save_metadata(data, meta_path)


if __name__ == '__main__':
    main()