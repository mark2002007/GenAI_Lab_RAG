#!/usr/bin/env python3
"""
preprocess.py

Production-ready text preprocessing and chunking for The Batch articles.
"""
import json
import logging
import re
from pathlib import Path
from typing import List, Optional

import tiktoken
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize tiktoken encoder
try:
    ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception as e:
    logger.error("Failed to initialize tiktoken encoder: %s", e)
    raise


def clean_text(text: Optional[str]) -> str:
    """
    Basic cleanup: collapse whitespace and normalize newlines.
    """
    if not text:
        return ""
    # Remove spaces at end of lines
    t = re.sub(r"[ \t]+\n", "\n", text)
    # Collapse multiple blank lines
    t = re.sub(r"\n{2,}", "\n\n", t)
    return t.strip()


def split_sentences(text: str) -> List[str]:
    """
    Splits text on periods, exclamation, or question marks as naive sentence boundaries.
    Retains delimiter at end.
    """
    # Use regex to keep delimiters
    parts = re.split(r'(?<=[\.\?\!])\s+', text)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences


def count_tokens(text: str) -> int:
    """
    Returns the number of tokens in text using tiktoken.
    """
    return len(ENCODER.encode(text))


def chunk_text(
    text: str,
    max_tokens: int = 500,
    min_tokens: int = 20,
    overlap_sentences: int = 1
) -> List[str]:
    """
    Splits text into coherent chunks under max_tokens tokens,
    with optional overlap of last few sentences.
    """
    sentences = split_sentences(text)
    chunks: List[str] = []
    cur_chunk: List[str] = []
    cur_count = 0

    i = 0
    n = len(sentences)
    while i < n:
        sent = sentences[i]
        tok = count_tokens(sent)
        if cur_chunk and cur_count + tok > max_tokens:
            # flush current chunk if above threshold
            chunk_str = " ".join(cur_chunk).strip()
            if count_tokens(chunk_str) >= min_tokens:
                chunks.append(chunk_str)
            # prepare overlap
            overlap = cur_chunk[-overlap_sentences:] if overlap_sentences <= len(cur_chunk) else cur_chunk
            cur_chunk = overlap.copy()
            cur_count = sum(count_tokens(s) for s in cur_chunk)
        else:
            cur_chunk.append(sent)
            cur_count += tok
            i += 1
    # append last chunk
    if cur_chunk:
        chunk_str = " ".join(cur_chunk).strip()
        if count_tokens(chunk_str) >= min_tokens:
            chunks.append(chunk_str)

    return chunks


def preprocess(
    input_path: Path,
    output_dir: Path,
    max_tokens: int = 500,
    min_tokens: int = 20,
    overlap_sentences: int = 1
) -> None:
    """
    Reads full-article JSON, cleans and chunks text, writes chunks to output_dir/chunks.json
    """
    logger.info("Loading articles from %s", input_path)
    try:
        articles = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("Failed to read input JSON: %s", e)
        raise

    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_out = []

    for art in tqdm(articles, desc="Preprocessing articles", unit="art"):
        raw = art.get("text") or ""
        cleaned = clean_text(raw)
        if not cleaned:
            logger.warning("Empty text for URL %s", art.get("url"))
            continue
        try:
            chunks = chunk_text(
                cleaned,
                max_tokens=max_tokens,
                min_tokens=min_tokens,
                overlap_sentences=overlap_sentences
            )
        except Exception as e:
            logger.error("Chunking failed for URL %s: %s", art.get("url"), e)
            continue

        for idx, txt in enumerate(chunks):
            chunks_out.append({
                "url": art.get("url", ""),
                "date": art.get("date_str", ""),
                "title": art.get("title", ""),
                "summary": art.get("summary"),
                "chunk_id": idx,
                "text": txt,
                "img_path": art.get("img_path")
            })

    out_path = output_dir / "chunks.json"
    logger.info("Writing %d chunks to %s", len(chunks_out), out_path)
    out_path.write_text(json.dumps(chunks_out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Production-ready preprocessing: clean, sentence-split, and chunk articles."
    )
    parser.add_argument(
        "--input", required=True,
        type=Path,
        help="Path to thebatch_full.json"
    )
    parser.add_argument(
        "--outdir", required=True,
        type=Path,
        help="Directory where chunks.json will be saved"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=500,
        help="Max tokens per chunk"
    )
    parser.add_argument(
        "--min-tokens", type=int, default=20,
        help="Min tokens per chunk"
    )
    parser.add_argument(
        "--overlap-sentences", type=int, default=1,
        help="Number of sentences to overlap between chunks"
    )
    args = parser.parse_args()
    preprocess(
        input_path=args.input,
        output_dir=args.outdir,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        overlap_sentences=args.overlap_sentences
    )