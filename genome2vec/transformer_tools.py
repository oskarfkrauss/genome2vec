
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
        tokenizer: AutoTokenizer, model: AutoModel, chunk: list, device=None):
    """
    Create an embedding of an annotated segment of a genome sequence (on GPU if available).

    Parameters
    ----------
    tokenizer : AutoTokenizer
        The tokenizer corresponding to the model we've chosen.
    model : AutoModel
        The Transformer embedding model
    segment : List[str]
        An annotated segment of the genome, either as a single item list or multiple
        items, all within the model's context limit
    device : torch.device or None
        Device to run the model on (CPU or GPU). Defaults to CPU.

    Returns
    -------
    torch.Tensor
        The embedding for the tokenized chunk (shape [seq_len, hidden_dim])
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # this is some torch logic to put some variables onto GPU accessible memory
    model = model.to(device)
    model.eval()

    tokens = tokenizer(chunk, return_tensors="pt")
    input_ids = tokens["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    # Get last hidden state, remove batch dimension, and move back to cpu
    token_embeddings = outputs.hidden_states[-1].squeeze(0).cpu()
    # return CLS token only as representation of the chunk
    return token_embeddings[0]


def split_to_max_length(annotation, max_length):
    """Splits an annotation into a list of chunks of up to max_length."""
    if len(annotation) <= max_length:
        return [annotation]
    return [annotation[i:i + max_length] for i in range(0, len(annotation), max_length)]


TRANSFORMER_MODEL_ARGS = {
    "NucleotideTransformer_2.5B": {
        "remote_path": "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
        # Nucleotide Transformer allows for 1000 6-mers but we make slightly smaller to allow for
        # start and end of sequence. Embedding dimension is 2560
        "max_seq_length": 5994},
    "ProkBERT": {
        "remote_path": "neuralbioinfo/prokbert-mini-long",
        # ProkBERT mini-long has a maximum context size of 4096 base pairs with an embedding
        # dimension of 384
        "max_seq_length": 4096},
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
        'max_seq_length': 160000},
    # **FOR TESTING*** set max length to 25
    "test_transformer": {
        "remote_path": "neuralbioinfo/prokbert-mini",
        "max_seq_length": 25}
}
