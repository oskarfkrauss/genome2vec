# Genome Transformer Comparison

A repository for generating embeddings from genome sequences

## Method

### Tokenisation

Each genome sequence is split into non-overlapping *k-mers* (e.g. 6-mers).  
The tokenizer converts these into numerical IDs that the Transformer model can process.

- **Token IDs:**  
  Example: `[2, 312, ..., 3671]` — each ID corresponds to a unique 6-mer in the tokenizer’s vocabulary.  
- **Attention mask:**  
  Example: `[1, 1, 1, 0, 0]` — indicates which tokens should be attended to by the model.  
  In our case, this will typically be all `1`s, since every 6-mer in the chunk is used.

---

### Embedding Generation

The Transformer model produces an **embedding vector for each token at each layer**.  
Because the model has 32 layers *plus* an initial embedding layer, the output contains **33 layers in total** (`layer 0` = embedding layer, `layers 1–32` = Transformer blocks).

To obtain the *final* representation of a sequence, we typically take the **last hidden layer** (`layer 32`). However, depending on the analysis, earlier or averaged layers may also be used.


## Environment Setup

First, install [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). Then, run

```
conda create --name genome_transformer python=3.10.12
```
this creates an environment called `genome_transformer` in which we will install our packages. Now run
```
conda activate genome_transformer
```
to activate the environment, then, install packages with
```
pip install -r requirements.txt
```
finally, run
```
pip install -e .
```
to setup the repository for development.

n.b.: Conda is just a way of managing your working environment, running the above command with a
[Python Virtual Environment (venv)](https://docs.python.org/3/library/venv.html) will also work.

## Acknowledgements

The general framework of this repository was inspired by the teachings of [Tristan Goss](https://uk.linkedin.com/in/tristan-goss) while at [Earthwave](https://earthwave.co.uk)