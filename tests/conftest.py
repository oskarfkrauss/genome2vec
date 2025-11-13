import pytest
import torch
from unittest.mock import MagicMock

import os


@pytest.fixture()
def test_inputs_dir() -> str:
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_inputs')


@pytest.fixture()
def mock_sequence() -> str:
    return 'AGTGGACGCATCACTGGTGTTCGGGTTGTCATGCCAATGGCATTGCCCGGT'


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
