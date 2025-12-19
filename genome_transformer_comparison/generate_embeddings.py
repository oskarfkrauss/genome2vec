'''
This script attempts to generate embeddings for various genome sequences using
a Transformer architecture
'''
import os
import time

import torch
from transformers import AutoTokenizer, AutoModel

from genome_transformer_comparison.tools import (
    parse_fasta, get_chunk_embedding, split_sequence_for_tokenizer)
from genome_transformer_comparison.generate_embeddings_config import (
    EMBEDDING_CONFIG, MODEL_MAX_SEQ_LENGTH_DICT)
from genome_transformer_comparison.configuration import ROOT_DIR, PRETRAINED_MODELS_DIR

# load pretrained models and tokenizer
tokenizer = AutoTokenizer.from_pretrained(os.path.join(
    PRETRAINED_MODELS_DIR, EMBEDDING_CONFIG['transformer_model'], 'tokenizer'),
    trust_remote_code=True)
model = AutoModel.from_pretrained(os.path.join(
    PRETRAINED_MODELS_DIR, EMBEDDING_CONFIG['transformer_model'], 'model'),
    trust_remote_code=True)

# max sequence length for tokenizer
max_seq_length = MODEL_MAX_SEQ_LENGTH_DICT[EMBEDDING_CONFIG['transformer_model']]

# get paths to folders in input
bacteria_names = os.listdir(
    os.path.join(ROOT_DIR, 'inputs', EMBEDDING_CONFIG['whole_genomes_or_plasmids'],
                 EMBEDDING_CONFIG['cpes_or_imps']))

for bacteria_name in bacteria_names:
    fasta_file_names = os.listdir(
        os.path.join(ROOT_DIR, 'inputs', EMBEDDING_CONFIG['whole_genomes_or_plasmids'],
                     EMBEDDING_CONFIG['cpes_or_imps'], bacteria_name))

    fasta_file_paths = [
        os.path.join(ROOT_DIR, 'inputs', EMBEDDING_CONFIG['whole_genomes_or_plasmids'],
                     EMBEDDING_CONFIG['cpes_or_imps'], bacteria_name, file_name)
        for file_name in fasta_file_names
    ]

    for i, fasta_file in enumerate(fasta_file_paths):
        start_time = time.perf_counter()

        # parse entire assembly into single string
        cpe_genome = parse_fasta(fasta_file)

        # split into list that the tokenizer can handle
        tokenizer_friendly_inputs = split_sequence_for_tokenizer(cpe_genome, max_seq_length)

        # generate embeddings for each chunk
        all_chunk_embeddings = []
        for chunk in tokenizer_friendly_inputs:
            chunk_embedding = get_chunk_embedding(tokenizer, model, [chunk])
            all_chunk_embeddings.append(chunk_embedding)

        # stack all chunk embeddings into a single tensor
        all_chunk_embeddings_tensor = torch.vstack(all_chunk_embeddings)

        # compute mean embedding across all chunks
        # NOTE: very important step, assumes order is not important
        genome_embedding = all_chunk_embeddings_tensor.mean(dim=0)

        # now save into outputs folder for specified transformer model
        embeddings_output_path = fasta_file.replace(
            "inputs", f"outputs/{EMBEDDING_CONFIG['transformer_model']}").replace("fasta", "pt")
        os.makedirs(os.path.dirname(embeddings_output_path), exist_ok=True)
        torch.save(genome_embedding, embeddings_output_path)

        elapsed_time = time.perf_counter() - start_time
        print(
            f"Isolate {i}/{len(fasta_file_paths) - 1} embedding took {elapsed_time:.4f} seconds")
