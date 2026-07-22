
import torch
from transformers import AutoTokenizer, AutoModel


def split_sequence_for_tokenizer(annotations_dict: dict, max_length: int) -> list:
    """
    TODO: write a descriptive docstring

    Parameters
    ----------
    annotations : dict
        Annotations of the sequence produced by Baka.
    max_length : int
        Maximum length (in characters) of each chunk. Choose this to match the tokenizer's
        maximum input size (or slightly smaller).

    Returns
    -------
    List[str]
        List of sequence segments suitable for chunking and passing to the model.
    """
    if max_length <= 0:
        raise ValueError("max_length must be > 0")

    # get features from annotations
    features = annotations_dict.get("features", [])
    # some features are 'gaps' and not very helpful, ignore
    filtered_features = [x for x in features if x.get("type") != "gap"]
    ordered_segments = []
    for feat in filtered_features:
        chunk_list = split_to_max_length(feat["nt"], max_length)
        ordered_segments.append(chunk_list)
    return ordered_segments


def get_annotation_embeddings(
        tokenizer: AutoTokenizer, model: AutoModel, segments: list, batch_size: int, device=None):
    """
    Create embedding of all annotated segments of a genome sequence (on GPU if available).

    Parameters
    ----------
    tokenizer : AutoTokenizer
        The tokenizer corresponding to the model we've chosen.
    model : AutoModel
        The Transformer embedding model
    segments : List[str]
        Annotated segmenta of the genome, as a list of: a single item list
        (containing the whole annotation's sequqnce) or a multiple items list
        (when an annotation's length exceeds model context).
    batch_size : int
        Number of chunks to process in parallel
    device : torch.device or None
        Device to run the model on (CPU or GPU). Defaults to CPU.

    Returns
    -------
    torch.Tensor
        The embedding for the tokenized chunk (shape [seq_len, hidden_dim])
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get chunks into one flat list but keep index of which segment they're in
    all_chunks = []
    segment_ids = []

    for segment_idx, segment in enumerate(segments):
        for chunk in segment:
            all_chunks.append(chunk)
            segment_ids.append(segment_idx)

    # create empty list of lists to append the chunk embeddings for each segment
    segment_embeddings = [
        [] for _ in range(len(segments))
    ]

    # process chunks in batches
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i + batch_size]
        batch_segment_ids = segment_ids[i:i + batch_size]

        tokens = tokenizer(batch_chunks, return_tensors="pt", padding=True)
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True)

        # batch token embeddings
        token_embeddings = outputs.hidden_states[-1]
        # batch CLS token embeddings
        cls_token_embeddings = token_embeddings[:, 0, :].cpu()

        # put each chunk embeddings for batch back its segment
        for emb, segment_idx in zip(cls_token_embeddings, batch_segment_ids):
            segment_embeddings[segment_idx].append(emb)

    # average chunks for each segment to get single segment representation
    mean_segment_embeddings = []
    for embeddings in segment_embeddings:
        mean_segment_embeddings.append(
            torch.mean(torch.vstack(embeddings), dim=0)
            )

    return mean_segment_embeddings


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
        # ProkBERT mini-long has a maximum context size of 4096 base paris with an embedding
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
