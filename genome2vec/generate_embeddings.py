'''
This script attempts to generate embeddings for various genome sequences using
a Transformer architecture

Usage:
    python generate_embeddings.py /path/to/config.yaml
'''
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel
import yaml

from genome2vec.transformer_tools import (
    parse_fasta,
    get_cls_token_embedding,
    split_sequence_for_tokenizer,
    TRANSFORMER_MODEL_ARGS
)


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main(config_path: Path) -> None:
    # --- Load config ---
    config = load_config(config_path)

    transformer_model = config["transformer_model"]
    sequence_dir = Path(config["sequence_dir"])

    if not sequence_dir.exists():
        raise FileNotFoundError(f"FASTA directory not found: {sequence_dir}")

    max_seq_length = TRANSFORMER_MODEL_ARGS[transformer_model]["max_seq_length"]

    # --- Load tokenizer and model ---
    tokenizer = AutoTokenizer.from_pretrained(
        TRANSFORMER_MODEL_ARGS[transformer_model]["remote_path"],
        trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        TRANSFORMER_MODEL_ARGS[transformer_model]["remote_path"],
        trust_remote_code=True
    )

    # --- Process FASTA files ---
    fasta_files = sorted(
        f for f in sequence_dir.iterdir()
        if f.suffix in {".fna", ".fasta", ".fa"}
    )

    if not fasta_files:
        raise RuntimeError(f"No FASTA files found in {sequence_dir}")

    for i, fasta_file in enumerate(fasta_files):
        start_time = time.perf_counter()

        # parse entire assembly into single string
        genome_sequence = parse_fasta(str(fasta_file))

        # split into tokenizer-friendly chunks
        chunks = split_sequence_for_tokenizer(genome_sequence, max_seq_length)

        # generate CLS token embedding for each chunk
        chunk_embeddings = [
            get_cls_token_embedding(tokenizer, model, [chunk])
            for chunk in chunks
        ]

        # stack and average
        all_chunk_embeddings = torch.vstack(chunk_embeddings)
        genome_embedding = all_chunk_embeddings.mean(dim=0)

        # Save output next to input
        output_path = Path(config["output_dir"]) / fasta_file.with_suffix(".pt").name
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(genome_embedding, output_path)

        elapsed = time.perf_counter() - start_time
        print(
            f"[{i+1}/{len(fasta_files)}] "
            f"{fasta_file.name} embedded in {elapsed:.2f}s"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python generate_embeddings.py /path/to/config.yaml"
        )

    main(Path(sys.argv[1]))
