'''this should be a yaml file'''
EMBEDDING_CONFIG = {
    # whole_genomes or plasmids
    'whole_genomes_or_plasmids': 'whole_genomes',
    # whether we are trying to embed the CPE sequences or IMPs
    'cpes_or_imps': 'imps',
    # which pre trained transformer model to use, can be NucleotideTransformer_2.5B, DNABERT_S,
    # HyenaDNA_medium_160k, ModernBert_DNA_37M_Virus
    'transformer_model': 'ModernBert_DNA_37M_Virus'
}

MODEL_MAX_SEQ_LENGTH_DICT = {
    # nucleotide transformer allows for 1000 6-mers but we make slightly smaller to allow for
    # start and end of sequence. Embedding dimension is 2560
    'NucleotideTransformer_2.5B': 5994,
    # DNABERT_S allows for a sequence of 2000 which are tokenised using Byte Pair Encoding,
    # given in the tokenizer_config.json. When running the model, the maximum number of tokens is
    # 512. Embedding dimension is 768
    'DNABERT_S': 2000,
    # Hyena models vary in max sequence length, the largest one compatible for our compute is the
    # 160k model which allows for 160k single nucleotide tokens. Embedding dimension is 256
    'HyenaDNA_medium_160k': 160000,
    # ModernBert_DNA_37M_Virus allows for 8192 tokens but is tokenised using byte pair encoding.
    # Was trained on ~1kb virus sequences so we use that?
    'ModernBert_DNA_37M_Virus': 1000
}
