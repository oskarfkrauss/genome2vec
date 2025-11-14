'''this should be a yaml file'''
EMBEDDING_CONFIG = {
    # whether we are trying to embed the CPE sequences or IMPs
    'cpes_or_imps': 'cpes',
    # which bacteria fasta files to generate embedding for, required to break up the process since I
    # think the GPU runs out of memory at some stage ??
    # can be any permutation of ['C_freundii', 'E_coli', 'E_hormaechei', 'K_pneumoniae']
    'bacteria_list': ['C_freundii', 'E_coli', 'E_hormaechei', 'K_pneumoniae'],
    # which pre trained transformer model to use, can be
    # NucleotideTransformer_2.5B, DNABERT_S, HyenaDNA_medium_160k, or GENA_LM_base
    'transformer_model': 'NucleotideTransformer_2.5B'
}
