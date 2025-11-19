'''this should be a yaml file'''
EMBEDDING_CONFIG = {
    # whether we are trying to embed the CPE sequences or IMPs
    'cpes_or_imps': 'cpes',
    # which pre trained transformer model to use, can be
    # NucleotideTransformer_2.5B, DNABERT_S, HyenaDNA_medium_160k, or GENA_LM_base, currently
    # only workd for NucleotideTransformer_2.5B
    'transformer_model': 'NucleotideTransformer_2.5B'
}
