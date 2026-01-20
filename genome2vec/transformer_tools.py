
import torch
from transformers import AutoTokenizer, AutoModel


def parse_fasta(file_path: str):
    '''
    Parse fasta file (.fna or .fasta) file into a single string

    Parameters
    ----------
    file_path : str
        Path to the fasta assembly file

    Returns
    -------
    seq : str
        The sequence parsed into a single string
    '''
    with open(file_path) as f:
        seq = ''
        for line in f:
            line = line.rstrip()
            # ignore lines containing read headers
            if line.startswith('>'):
                continue
            else:
                seq = seq + line
    return seq


def split_sequence_for_tokenizer(sequence: str, max_length: int) -> list:
    """
    Split a long genome sequence string into a list of substrings each no longer than
    max_length

    Parameters
    ----------
    sequence : str
        Raw sequence. This will be normalized to uppercase.
    max_length : int
        Maximum length (in characters) of each chunk. Choose this to match the tokenizer's
        maximum input size (or slightly smaller).

    Returns
    -------
    List[str]
        List of sequence chunks suitable for passing individually to the tokenizer.
    """
    if max_length <= 0:
        raise ValueError("max_length must be > 0")

    chunks = []
    step = max_length
    start = 0
    seq_len = len(sequence)
    while start < seq_len:
        end = start + max_length
        chunks.append(sequence[start:end])
        start += step
    return chunks


def get_chunk_embedding(
        tokenizer: AutoTokenizer, model: AutoModel, sequence: str, device=None):
    """
    Create an embedding of a 'chunk' of a genome sequence (on GPU if available).

    Parameters
    ----------
    sequence : str
        A 'chunk' of the genome sequence.
    device : torch.device or None
        Device to run the model on (CPU or GPU). Defaults to CPU.

    Returns
    -------
    torch.Tensor
        The embedding for the tokenized chunk (shape [seq_len, hidden_dim])
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenize and move to device, truncate the sequnece so that the model can handle the input,
    # can result in the last few nucleotides (of the whole seq) not being included in the embedding
    tokens = tokenizer(sequence, return_tensors="pt", truncation=True)
    input_ids = tokens["input_ids"].to(device)

    # this is some torch logic to put some variables (?) onto GPU accessible memory
    model = model.to(device)
    model.eval()  # ensure evaluation mode

    with torch.no_grad():
        outputs = model(
            input_ids,
            output_hidden_states=True)

    # Get last hidden state, remove batch dimension, and move back to cpu
    embeddings = outputs.hidden_states[-1].squeeze(0).cpu()

    return embeddings
