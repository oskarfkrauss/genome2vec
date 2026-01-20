'''
Integration test for genome2vec.

Input is path to folder with assembly files
Output is tensors mathcing the name of the assemblies
'''
import os
import subprocess
import sys
from pathlib import Path
import yaml


def test_generate_embeddings(test_inputs_dir, tmp_path):
    """
    End-to-end integration test for generate_embeddings.py.
    """
    # -- Generate mock config.yaml --
    # Mock input FASTA directory
    sequences_dir = test_inputs_dir / "mock_sequences"

    # mock output directory
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    config = {
        "sequence_dir": str(sequences_dir),
        "output_dir": str(output_dir),
        "transformer_model": "ModernBert_DNA_37M_Virus",
    }

    config_path = tmp_path / "mock_config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

    # path to script under test
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "genome2vec" / "generate_embeddings.py"

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            str(config_path),
        ],
        capture_output=True
    )

    expected_tensors = [
        Path(output_dir) / Path(f).with_suffix(".pt")
        for f in os.listdir(sequences_dir)
    ]

    for tensor_file in expected_tensors:
        assert tensor_file.exists(), f"Expected tensor file not found: {tensor_file}"
