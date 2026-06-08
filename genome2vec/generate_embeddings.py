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
from transformers import utils as hf_utils
from transformers import AutoTokenizer, AutoModel
import yaml

from genome2vec.transformer_tools import (
    get_chunk_embedding,
    split_sequence_for_tokenizer,
    TRANSFORMER_MODEL_ARGS
)
from genome2vec.genome_annotation_tools import annotate_genomes
from genome2vec.logger import Logger


def main(config_path: Path) -> None:
    # --- Load config ---
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger = Logger(config=config)

    # get the Hugging Face logger
    hf_logger = hf_utils.logging.get_logger("transformers")
    hf_logger.handlers = []

    # direct Hugging Face straight to custom filer handler
    # pull the handler from the custom logger and add the HF one
    if logger._logger.handlers:
        custom_file_handler = logger._logger.handlers[0]
        hf_logger.addHandler(custom_file_handler)

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

        logger.info(f'annotating fasta file {fasta_file}')

        # run genome annotation using bakta, thread count is hardcoded for now
        annotations_dict = annotate_genomes(fasta_file, bakta_db_path, 24)

        logger.info('annotation_complete!')

        # use annotations to get coding and non coding chunks
        annotation_segments = split_sequence_for_tokenizer(annotations_dict, max_seq_length)

        output_path = Path(config["output_dir"]) / fasta_file.with_suffix(".pt").name
        # exit if embedding has already been completed
        if os.path.exists(output_path):
            logger.info('embedding already complete, continuing to next sample')
            continue

        # generate token embedding for each chunk
        annotation_embeddings = [
            get_chunk_embedding(tokenizer, model, segment)
            for segment in annotation_segments
        ]

        # stack and mean
        all_chunk_embeddings = torch.vstack(annotation_embeddings)
        genome_embedding = all_chunk_embeddings.mean(dim=0)

        # Save output next to input
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(genome_embedding, output_path)

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"[{i+1}/{len(fasta_files)}] "
            f"{fasta_file.name} embedded in {elapsed:.2f}s"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python generate_embeddings.py /path/to/config.yaml"
        )
    main(Path(sys.argv[1]))
