# genome2vec

This repository provides a framework for generating (fixed-dimensional) genome embeddings using transformer-based foundation models. It loads pretrained sequence models, tokenizes genomes from FASTA files, chunks long sequences to respect tokenizer limits, computes per-chunk embeddings, and aggregates them into a single vector per genome.

![](figures/genome2vec.png)

The workflow is implemented primarily in two modules:

`transformer_tools.py` — utilities for parsing FASTA files, splitting sequences, and computing chunk embeddings

`generate_embeddings.py` — main script that orchestrates model loading, sequence processing, and embedding generation

## Method

![](figures/embedding_generation.png)

### Chunking

Sequences are parsed and chunked according to model context 

### Tokenisation

Each genome sequence is split into non-overlapping *k-mers* (e.g. 6-mers).  
The tokenizer converts these into numerical IDs that the Transformer model can process.

- **Token IDs:**  
  Example: `[2, 312, ..., 3671]` — each ID corresponds to a unique 6-mer in the tokenizer’s vocabulary.  

---

### Embedding Generation

The Nucleotide Transformer model produces **token-level hidden states for each layer**.

- The model consists of:
- **1 embedding layer** (layer 0)  
- **32 Transformer blocks** (layers 1–32)

This results in **33 hidden-state layers** returned per forward pass.

By default, the pipeline uses the **last hidden layer** (layer 32) as the representation of a chunk, but this could be modified to:

- use earlier layers  
- average multiple layers  
- concatenate layers  

Chunk embeddings are averaged across all chunks to produce one **genome-level embedding vector**. Again, this could be modified to a more intelligent aggregation e.g. with an RNN.

This final embedding is written as a PyTorch `.pt` tensor to the output directory.

## How It Works (Pipeline Summary)

1. **Load FASTA**  
   `parse_fasta()` reads sequences and concatenates them into one string.

2. **Split Sequence**  
   `split_sequence_for_tokenizer()` creates chunks compatible with model limits.

3. **Tokenise & Encode**  
   Each chunk is converted to k-mer or BPE tokens and passed through the transformer.

4. **Extract Hidden States**  
   The model returns hidden layers of token embeddings.

5. **Select Layer**  
   The pipeline selects the *last hidden layer*.

6. **Aggregate Across Chunks**  
   All chunk embeddings are stacked and averaged to yield a single genome embedding.

7. **Save Output**  
   A `.pt` tensor is written to the output directory.

## How to Use

Run
```
python generate_embeddings.py path/to/assembly/files
```

## Acknowledgements

The general framework of this repository was inspired by the teachings of [Tristan Goss](https://uk.linkedin.com/in/tristan-goss) while at [Earthwave](https://earthwave.co.uk)