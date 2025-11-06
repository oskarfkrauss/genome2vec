'''
This script attempts to generate embeddings for various genome sequences using
a Transformer architecture
'''
import os
import time

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from genome_transformer_comparison.tools import (
    parse_fasta, get_chunk_embedding, split_sequence_for_tokenizer)
from genome_transformer_comparison.configuration import ROOT_DIR, PRETRAINED_MODELS_DIR

# parse the genome file
path_to_cpe_genome = os.path.join(
    ROOT_DIR, 'data', 'cpes', 'C_freundii', '234098wA8_CPE0002631_assembly_filtered.fasta')

# load pretrained models and tokenizer
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODELS_DIR)
model = AutoModelForMaskedLM.from_pretrained(PRETRAINED_MODELS_DIR)
model.eval()  # set model to evaluation mode

# get k of k-mer from the tokenizer
special_tokens = set(tokenizer.all_special_tokens)
vocab_keys = [k for k in tokenizer.get_vocab().keys() if k not in special_tokens]
first_kmer = vocab_keys[0]
kmer_length = len(first_kmer)

# max sequence length for tokenizer
max_seq_length = kmer_length * (tokenizer.model_max_length - 1)

# parse entire assembly file into single string
cpe_genome = parse_fasta(path_to_cpe_genome)

# split into list that the tokenizer can handle
tokenizer_friendly_inputs = split_sequence_for_tokenizer(cpe_genome, max_seq_length)

# generate embeddings for each chunk
all_chunk_embeddings = []
for i, chunk in enumerate(tokenizer_friendly_inputs, 1):
    start_time = time.perf_counter()
    chunk_embedding = get_chunk_embedding(tokenizer, model, [chunk])
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print(f"Chunk {i}/{len(tokenizer_friendly_inputs)} embedding took {elapsed_time:.4f} seconds")

    all_chunk_embeddings.append(chunk_embedding)

# stack all chunk embeddings into a single tensor
all_chunk_embeddings_tensor = torch.vstack(all_chunk_embeddings)

# compute mean embedding across all chunks
genome_embedding = all_chunk_embeddings_tensor.mean(dim=0)
