
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
        The sequence parsed into a single string and capitalised
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
    return seq.upper()


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

    # this is some torch logic to put some variables onto GPU accessible memory
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(
            input_ids,
            output_hidden_states=True)

    # Get last hidden state, remove batch dimension, and move back to cpu
    embeddings = outputs.hidden_states[-1].squeeze(0).cpu()

    return embeddings


def get_cls_token_embedding(
        tokenizer: AutoTokenizer, model: AutoModel, sequence: str, device=None):
    """
    Create an embedding using only the CLS token (on GPU if available).

    Parameters
    ----------
    sequence : str
        A 'chunk' of the genome sequence.
    device : torch.device or None
        Device to run the model on (CPU or GPU). Defaults to CPU.

    Returns
    -------
    torch.Tensor
        CLS embedding (shape [hidden_dim])
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenize and move to device, truncate the sequnece so that the model can handle the input,
    # can result in the last few nucleotides (of the whole seq) not being included in the embedding
    tokens = tokenizer(sequence, return_tensors="pt", truncation=True)
    input_ids = tokens["input_ids"].to(device)

    # this is some torch logic to put some variables onto GPU accessible memory
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(
            input_ids,
            output_hidden_states=True)

    # Get CLS token embedding from final layer
    embeddings = outputs.hidden_states[-1][:, 0, :].squeeze(0).cpu()

    return embeddings


TRANSFORMER_MODEL_ARGS = {
    "NucleotideTransformer_2.5B": {
        "remote_path": "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
        # Nucleotide Transformer allows for 1000 6-mers but we make slightly smaller to allow for
        # start and end of sequence. Embedding dimension is 2560
        "max_seq_length": 5994},
    "ModernBert_DNA_37M_Virus": {
        'remote_path': "RaphaelMourad/ModernBert-DNA-v1-37M-virus",
        # ModernBert_DNA_37M_Virus allows for 8192 tokens but is tokenised using byte pair encoding.
        # Was trained on ~1kb virus sequences so we use that
        'max_seq_length': 160000},
    "DNABERT_S": {
        "remote_path": "zhihan1996/DNABERT-S",
        # DNABERT_S allows for a sequence of 2000 which are tokenised using Byte Pair Encoding,
        # given in the tokenizer_config.json. When running the model, the maximum number of tokens
        # is 512. Embedding dimension is 768
        'max_seq_length': 2000},
    "HyenaDNA_medium_160k": {
        "remote_path": "LongSafari/hyenadna-medium-160k-seqlen-hf",
        # ModernBert_DNA_37M_Virus allows for 8192 tokens but is tokenised using byte pair encoding.
        # Was trained on ~1kb virus sequences so we use that
        'max_seq_length': 160000}
}
