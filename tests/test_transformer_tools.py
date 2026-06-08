from genome2vec.transformer_tools import (
    split_sequence_for_tokenizer, get_annotation_embedding)


def test_split_sequence_for_tokenizer(mock_annotations_dict):
    chunked_annotations = split_sequence_for_tokenizer(mock_annotations_dict, 25)
    expected_chunks = [
        ["AGTG"], ["GACGCATCAC"], ["TGGTGTTCGG"], ["ATTGGCATGACAAC"], ["GGCATTGCCCGGT"],
        ["AG"], ["TGGACGCA"], ["TCACTGGTGGTCGCGTTGTCATGCC", "AATGGCATTGCACAAA"]
        ]
    assert chunked_annotations == expected_chunks


# not a great unit test since we run multiple times in a comprehension
def test_get_annotation_embedding(mock_tokenizer, mock_model, mock_segments):
    segment_embeddings = [
        get_annotation_embedding(mock_tokenizer, mock_model, segment) for segment in mock_segments
        ]
    # we have a sequence made of 4 annotations, one of which makes up two chunks, the result
    # should be 4 segment embeddings
    assert len(segment_embeddings) == 4
