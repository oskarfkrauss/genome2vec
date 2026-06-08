'''
Integration test for genome2vec.
'''
import os
from pathlib import Path
import yaml

from genome2vec.generate_embeddings import main


def test_generate_embeddings(test_inputs_dir, tmp_path):
    """
    End-to-end integration test for generate_embeddings.py.
    """
    # -- Generate mock config.yaml --
    # Mock input FASTA directory
    sequences_dir = test_inputs_dir / "mock_sequences"

    # mock output directory
    output_dir = tmp_path / "outputs"

    # mock the annotations directory and skip any call to bakta since annotation already exists
    annotation_dir = test_inputs_dir / 'mock_annotations' / 'db'

    # mock a logging directory
    logging_dir = tmp_path / "logs"

    config = {
        "sequence_dir": str(sequences_dir),
        "output_dir": str(output_dir),
        "annotation_dir": str(annotation_dir),
        "logging_dir": str(logging_dir),
        "transformer_model": "test_transformer"
    }

    config_path = tmp_path / "mock_config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

    main(config_path)

    expected_tensors = [
        Path(output_dir) / Path(f).with_suffix(".pt")
        for f in os.listdir(sequences_dir)
    ]

    for tensor_file in expected_tensors:
        assert tensor_file.exists(), f"Expected tensor file not found: {tensor_file}"
