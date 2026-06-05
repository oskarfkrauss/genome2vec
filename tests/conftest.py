import json
import os
import pytest
import torch
from unittest.mock import MagicMock

from pathlib import Path


@pytest.fixture()
def test_inputs_dir() -> Path:
    """
    Returns the path to the test_inputs directory as a Path object.
    """
    return Path(__file__).resolve().parent / "test_inputs"


@pytest.fixture()
def mock_sequence() -> str:
    return 'AGTGGACGCATCACTGGTGTTCGGGTTGTCATGCCAATGGCATTGCCCGGT'


@pytest.fixture()
def mock_annotations_dict(test_inputs_dir) -> dict:
    with open(os.path.join(test_inputs_dir, 'mock_annotations', 'annotation_results',
                           'mock_sequence_1', 'mock_sequence_1.json'), "r") as f:
        annotation_dict = json.load(f)
    return annotation_dict


@pytest.fixture()
def mock_tokenizer() -> MagicMock:
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]])
    }
    return mock_tokenizer


@pytest.fixture()
def mock_model() -> MagicMock:
    mock_model = MagicMock()
    # ignore .to and .eval methods
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = mock_model

    # mock outputs from model
    mock_outputs = MagicMock()
    mock_outputs.hidden_states = torch.tensor([[0.1, 0.2, 0.3]])

    mock_model.return_value = mock_outputs
    return mock_model
