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

    #  create a specific 3D tensor: shape (1, 3, 4)
    # Token 0 (The CLS token) will be exactly [0.1, 0.2, 0.3, 0.4]
    fixed_tensor = torch.tensor([[[0.1, 0.2, 0.3, 0.4],  # Token 0 (CLS token)
                                  [0.5, 0.6, 0.7, 0.8],  # Token 1
                                  [0.9, 1.0, 1.1, 1.2]]]) # Token 2    
    

    # mock outputs from model
    mock_outputs = MagicMock()
    # mock the hidden states to be a list of tensors, where the last one has shape [1, seq_len, hidden_dim]
    mock_outputs.hidden_states = [fixed_tensor]

    mock_model.return_value = mock_outputs
    return mock_model
