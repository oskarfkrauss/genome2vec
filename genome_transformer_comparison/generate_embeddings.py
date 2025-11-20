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
from genome_transformer_comparison.generate_embeddings_config import EMBEDDING_CONFIG
from genome_transformer_comparison.configuration import ROOT_DIR, PRETRAINED_MODELS_DIR

# load pretrained models and tokenizer
tokenizer = AutoTokenizer.from_pretrained(os.path.join(
    PRETRAINED_MODELS_DIR, EMBEDDING_CONFIG['transformer_model'], 'tokenizer'))
model = AutoModel.from_pretrained(os.path.join(
    PRETRAINED_MODELS_DIR, EMBEDDING_CONFIG['transformer_model'], 'model'))

# get k of k-mer from the tokenizer
special_tokens = set(tokenizer.all_special_tokens)
vocab_keys = [k for k in tokenizer.get_vocab().keys() if k not in special_tokens]
first_kmer = vocab_keys[0]
kmer_length = len(first_kmer)

# max sequence length for tokenizer
max_seq_length = kmer_length * (tokenizer.model_max_length - 1)

# get paths to folders in input
bacteria_names = os.listdir(
    os.path.join(ROOT_DIR, 'inputs', EMBEDDING_CONFIG['whole_genomes_or_plasmids'],
                 EMBEDDING_CONFIG['cpes_or_imps']))

for bacteria_name in bacteria_names:
    # TODO: setup logging
    if EMBEDDING_CONFIG['whole_genomes_or_plasmids'] == 'whole_genomes':
        # then the folder structure is broken down by bacteria
        fasta_file_names = os.listdir(
            os.path.join(ROOT_DIR, 'inputs', EMBEDDING_CONFIG['whole_genomes_or_plasmids'],
                         EMBEDDING_CONFIG['cpes_or_imps'], bacteria_name))

        fasta_file_paths = [
            os.path.join(ROOT_DIR, 'inputs', EMBEDDING_CONFIG['whole_genomes_or_plasmids'],
                         EMBEDDING_CONFIG['cpes_or_imps'], bacteria_name, file_name)
            for file_name in fasta_file_names
        ]
    else:
        # then the folder structure is not broken down by bacteria
        fasta_file_names = os.listdir(
            os.path.join(ROOT_DIR, 'inputs', EMBEDDING_CONFIG['whole_genomes_or_plasmids'],
                         EMBEDDING_CONFIG['cpes_or_imps']))

        fasta_file_paths = [
            os.path.join(ROOT_DIR, 'inputs', EMBEDDING_CONFIG['whole_genomes_or_plasmids'],
                         EMBEDDING_CONFIG['cpes_or_imps'], file_name)
            for file_name in fasta_file_names
        ]
        # NOTE: this is a not a great solution but can't really be avoided atm, sorry

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

        # now save into outputs folder
        embeddings_output_path = fasta_file.replace('inputs', 'outputs').replace('fasta', 'pt')
        os.makedirs(os.path.dirname(embeddings_output_path), exist_ok=True)
        torch.save(genome_embedding, embeddings_output_path)

        elapsed_time = time.perf_counter() - start_time
        print(
            f"Isolate {i}/{len(fasta_file_paths) - 1} embedding took {elapsed_time:.4f} seconds")
