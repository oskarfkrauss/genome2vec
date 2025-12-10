'''this should be a yaml file'''
EMBEDDING_CONFIG = {
    # wgs or plasmids
    'whole_genomes_or_plasmids': 'whole_genomes',
    # whether we are trying to embed the CPE sequences or IMPs
    'cpes_or_imps': 'cpes',
    # which pre trained transformer model to use, can be NucleotideTransformer_2.5B, DNABERT_S
    'transformer_model': 'NucleotideTransformer_2.5B'
}

MODEL_MAX_SEQ_LENGTH_DICT = {
    # nucleotide transformer allows for 1000 6-mers but we make slightly smaller to allow for
    # start and end of sequence
    'NucleotideTransformer_2.5B': 5994,
    # DNABERT_S allows for a sequence of 2000 which are tokenised using Byte Pair Encoding,
    # cannot find any robust proof that this is the limit. When running the model, there seems,
    # to be a 512 token limit ?
    'DNABERT_S': 2000
}
