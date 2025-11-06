import pytest

import os


@pytest.fixture()
def test_inputs_dir() -> str:
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_inputs')


@pytest.fixture()
def mock_sequence() -> str:
    return 'AGTGGACGCATCACTGGTGTTCGGGTTGTCATGCCAATGGCATTGCCCGGT'
