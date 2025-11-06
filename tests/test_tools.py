import os

from genome_transformer_comparison.tools import parse_fasta, split_sequence_for_tokenizer


def test_parse_fasta(test_inputs_dir):
    mock_fasta_file = os.path.join(test_inputs_dir, 'mock_sequence.fna')
    parse_fasta(mock_fasta_file)
    assert len(parse_fasta(mock_fasta_file)) == 51


def test_split_sequence_for_tokenizer(mock_sequence):
    chunked_sequence = split_sequence_for_tokenizer(mock_sequence, 25)
    # the sequence has a length of 51, so splitting into chunks of size 25 should give 3 list
    # with length 25, 25, 1
    assert len(chunked_sequence) == 3
    assert len(chunked_sequence[-1]) == 1
