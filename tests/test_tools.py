import os

from genome_transformer_comparison.tools import parse_fasta


def test_parse_fasta(test_inputs_dir):
    mock_fasta_file = os.path.join(test_inputs_dir, 'mock_sequence.fna')
    parse_fasta(mock_fasta_file)
    assert len(parse_fasta(mock_fasta_file)) == 51

