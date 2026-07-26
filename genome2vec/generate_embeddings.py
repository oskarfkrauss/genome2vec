'''
This script attempts to generate embeddings for various genome sequences using
a Transformer architecture

Usage:
    python generate_embeddings.py /path/to/config.yaml
'''
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import time
from pathlib import Path

import torch
from transformers import utils as hf_utils
from transformers import AutoTokenizer, AutoModel
import yaml

from genome2vec.transformer_tools import (
    get_annotation_embeddings,
    TRANSFORMER_MODEL_ARGS
)
from genome2vec.genome_annotation_tools import annotate_genome, split_sequence_for_tokenizer
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
    annotation_dir = Path(config["annotation_dir"])
    # for genome annotation
    bakta_db_path = config["annotation_db"]
    annotation_threads = config["annotation_threads"]

    if not sequence_dir.exists():
        raise FileNotFoundError(f"FASTA directory not found: {sequence_dir}")

    max_seq_length = TRANSFORMER_MODEL_ARGS[transformer_model]["max_seq_length"]

    # --- Load tokenizer, create model for each available device ---
    devices = config["available_devices"]
    models = {}
    for device in devices:
        model_for_device = AutoModel.from_pretrained(
            TRANSFORMER_MODEL_ARGS[transformer_model]["remote_path"],
            trust_remote_code=True)
        model_for_device.to(device)
        model_for_device.eval()
        models[device] = model_for_device

    tokenizer = AutoTokenizer.from_pretrained(
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

    for i, fasta_path in enumerate(fasta_files):
        start_time = time.perf_counter()

        # sometimes downloaded fastas are empty due to some poor ftp management from the database
        # skip these files and continue
        if os.path.getsize(fasta_path) < 50:
            continue

        logger.info(f'Loading annotations for genome {fasta_path.name}')

        annotation_table_path = os.path.join(annotation_dir, f"{fasta_path.stem}.PATRIC.gff")
        # we either fetch annotations we've downloaded from the database or annotate ourselves
        # using Bakta. Sometimes downloaded gffs are empty, skip these files and annotate
        if not os.path.exists(annotation_table_path) or \
            os.path.getsize(annotation_table_path) < 100:
            # run genome annotation using bakta, thread count is in config for now
            annotation_table_path = annotate_genome(fasta_path, bakta_db_path, annotation_threads)

        logger.info('Annotations loaded! Preparing sequence for tokeniser')

        # use annotations to get coding and non coding chunks
        annotation_segments = split_sequence_for_tokenizer(
            annotation_table_path, fasta_path, max_seq_length)

        num_segments = len(annotation_segments)
        # count number of chunks
        total_chunks = sum(len(chunk) for chunk in annotation_segments)

        logger.info('Sequence split into %s segments formed of %s chunks',
                    num_segments, total_chunks)

        # exit if embedding has already been completed
        output_path = Path(config["output_dir"]) / fasta_path.with_suffix(".pt").name
        if os.path.exists(output_path):
            logger.info('Embedding already complete, continuing to next sample')
            continue

        logger.info('Generating embedding for sequence')

        # Parellise across available devices, first split annotations into three roughly equal
        # batches
        batch_size = (num_segments + len(devices) - 1) // len(devices)
        batches = [
            annotation_segments[i:i + batch_size]
            for i in range(0, num_segments, batch_size)
        ]

        def run(batch, device):
            return get_annotation_embeddings(
                tokenizer,
                models[device],
                batch,
                batch_size=4,
                device=device,
            )

        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            results = list(executor.map(run, batches, devices))

        # Flatten if each result is a list
        annotation_embeddings = [
            embedding
            for result in results
            for embedding in result]

        # stack all CLS token embeddings
        genome_embedding = torch.vstack(annotation_embeddings)

        # Save output next to input
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(genome_embedding, output_path)

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"[{i+1}/{len(fasta_files)}] "
            f"{fasta_path.name} embedded in {elapsed:.2f}s"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python generate_embeddings.py /path/to/config.yaml"
        )
    main(Path(sys.argv[1]))
