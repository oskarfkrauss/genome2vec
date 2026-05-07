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

from utils.transformer_tools import (
    parse_fasta,
    get_chunk_embeddings,
    split_sequence_for_tokenizer,
    TRANSFORMER_MODEL_ARGS
)


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main(config_path: Path) -> None:
    '''
    1 chunk = ~6,000 nucleotides. By incrasing batch size, it increases the number of chunks processed
    1 chunk = 1[cls]
    Increase batch size to utilise GPU strengths, but be mindful of VRAM limits. If you get OOM errors, reduce batch size.
    '''
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

    # Define device and move model onto gpu vram.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for i, fasta_file in enumerate(fasta_files):
        start_time = time.perf_counter()

        # parse entire assembly into single string
        genome_sequence = parse_fasta(str(fasta_file))

        # split into tokenizer-friendly chunks 
        # bug currently loading the 2.5b model for every loop.
        chunks = split_sequence_for_tokenizer(genome_sequence, max_seq_length)

        # 2. Pass the entire list of chunks to the function
        chunk_embeddings = get_chunk_embeddings(
            tokenizer=tokenizer, 
            model=model, 
            chunks=chunks, 
            batch_size=16, # Adjust this based on your GPU VRAM (e.g., 8, 16, 32, 64)
            device=device
        )


        # Save everything to do cross attention output next to input
        output_path = Path(config["output_dir"]) / fasta_file.with_suffix(".pt").name
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Shape: [n_chunks, 2560] not [2560]
        torch.save(chunk_embeddings, output_path)

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
