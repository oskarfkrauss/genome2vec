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
    get_chunk_embedding,
    split_sequence_for_tokenizer,
    TRANSFORMER_MODEL_ARGS
)
from genome2vec.genome_annotation_tools import annotate_genomes


def main(config_path: Path) -> None:
    # --- Load config ---
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    transformer_model = config["transformer_model"]
    sequence_dir = Path(config["sequence_dir"])
    # for genome annotation
    bakta_db_path = config["annotation_dir"]

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

        # run genome annotation using bakta, thread count is hardcoded for now
        annotations = annotate_genomes(fasta_file, bakta_db_path, 12)

        # use annotations to get coding and non coding chunks
        chunks = split_sequence_for_tokenizer(annotations, max_seq_length)

        # generate token embedding for each chunk
        chunk_embeddings = [
            get_chunk_embedding(tokenizer, model, [chunk])
            for chunk in chunks
        ]

        # stack and mean
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
