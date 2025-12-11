'''this should be a yaml file'''
EMBEDDING_CONFIG = {
    # wgs or plasmids
    'whole_genomes_or_plasmids': 'plasmids',
    # whether we are trying to embed the CPE sequences or IMPs
    'cpes_or_imps': 'imps',
    # which pre trained transformer model to use, can be NucleotideTransformer_2.5B, DNABERT_S
    'transformer_model': 'DNABERT_S'
}

MODEL_MAX_SEQ_LENGTH_DICT = {
    # nucleotide transformer allows for 1000 6-mers but we make slightly smaller to allow for
    # start and end of sequence
    'NucleotideTransformer_2.5B': 5994,
    # DNABERT_S allows for a sequence of 2000 which are tokenised using Byte Pair Encoding,
    # given in the tokenizer_config.json. When running the model, the maximum number of tokens is
    # 512
    'DNABERT_S': 2000
}
