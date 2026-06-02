import torch

import os

from genome2vec.transformer_tools import (
    split_sequence_for_tokenizer, get_chunk_embedding)


def test_split_sequence_for_tokenizer(mock_annotations_dict):
    chunked_annotations = split_sequence_for_tokenizer(mock_annotations_dict, 25)
    expected_chunks = [
        "AGTG", "GACGCATCAC", "TGGTGTTCGG", "ATTGGCATGACAAC", "GGCATTGCCCGGT",
        "AG", "TGGACGCA", "TCACTGGTGGTCGCGTTGTCATGCCAATGGCATTGCACAAA"
        ]
    assert chunked_annotations == expected_chunks


def test_get_chunk_embedding(mock_tokenizer, mock_model, mock_sequence):
    chunk_embedding = get_chunk_embedding(mock_tokenizer, mock_model, mock_sequence)
    torch.testing.assert_close(chunk_embedding, torch.tensor([0.1, 0.2, 0.3]))
