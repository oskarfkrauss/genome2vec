
import torch
from transformers import AutoTokenizer, AutoModel


def parse_fasta(file_path: str):
    '''
    Parse fasta file (.fna or .fasta) file into a single string
    Used to concetante all contigs into one big string

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

# Do we need it to overlap?
# What if the resistance gene sits exactly on the boundary.
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


def get_chunk_embeddings(
        tokenizer: AutoTokenizer, 
        model: AutoModel, 
        chunks: list[str], 
        batch_size: int = 16, 
        device: torch.device = None
) -> torch.Tensor:
    """
        Computes [CLS] embeddings for a list of genome chunks using mini-batching 
        and mixed precision.

        Returns a single tensor containing the [CLS] embeddings for all chunks, stacked along the first dimension.
    """

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


    model.eval()
    all_embeddings = []

    # Determine the best precision type for the hardware
    # bfloat16 is preferred for stability if the GPU supports it
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    with torch.no_grad():
        # Iterate over the chunks in mini-batches (looping through batch size)
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            
            # Tokenize the entire batch at once. 
            # Padding is required now because sequences in a batch must be the same length.
            tokens = tokenizer(
                batch_chunks, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(device)

            # Force the forward pass into mixed precision
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                outputs = model(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"], # Pass attention mask for padded tokens
                    output_hidden_states=True
                )

            # Get last hidden state, remove batch dimension, and move back to cpu
            last_hidden_state = outputs.hidden_states[-1]
            # Grab all sequences in the batch (:), the [CLS] token (0), and all hidden dims (:)
            cls_embeddings = last_hidden_state[:, 0, :]
            # Shape: [16, 2560] for 16 sequences in batch and 2560 hidden dimension for NucleotideTransformer_2.5B

            # Move back to CPU immediately to free up GPU VRAM for the next batch
            all_embeddings.append(cls_embeddings.cpu().to(torch.float32))
            
    # Stack all mini-batch tensors into one final tensor
    return torch.cat(all_embeddings, dim=0)



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
