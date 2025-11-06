'''
This script attempts to generate embeddings for various CPE genome sequences using
different Transformer architectures
'''
import os

from transformers import AutoTokenizer, AutoModelForMaskedLM

from genome_transformer_comparison.tools import parse_fasta, get_mean_embedding
from genome_transformer_comparison.configuration import ROOT_DIR, PRETRAINED_MODELS_DIR


# parse the genome file
path_to_cpe_genome_1 = os.path.join(
    ROOT_DIR, 'data', 'cpes', 'C_freundii', '234098wA8_CPE0002631_assembly_filtered.fasta')

# parse first line of assembly file
cpe_genome_1 = parse_fasta(path_to_cpe_genome_1)[0]

# load pretrained models and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    PRETRAINED_MODELS_DIR
    )
model = AutoModelForMaskedLM.from_pretrained(
    PRETRAINED_MODELS_DIR,
    )

# attempt embedding of genome (1x2560 tensor)
genome_embedding = get_mean_embedding(tokenizer, model, cpe_genome_1)
