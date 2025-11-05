'''
This script attempts to generate embeddings for various CPE genome sequences using
different Transformer architectures
'''
import os

from transformers import AutoTokenizer, AutoModelForMaskedLM

from genome_transformer_comparison.tools import parse_fasta, get_masked_embedding
from genome_transformer_comparison.configuration import ROOT_DIR

# parse the genome file
path_to_cpe_genome_1 = os.path.join(
    ROOT_DIR, 'data', 'cpes', 'C_freundii', '234098wA8_CPE0002631_assembly_filtered.fasta')

cpe_genome_1 = parse_fasta(path_to_cpe_genome_1)[0][0:100]
print(cpe_genome_1)

# load pretrained models and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
model = AutoModelForMaskedLM.from_pretrained(
    "InstaDeepAI/nucleotide-transformer-2.5b-multi-species")

# attempt embedding of genome
genome_embedding = get_masked_embedding(tokenizer, model, cpe_genome_1)

print(genome_embedding)
