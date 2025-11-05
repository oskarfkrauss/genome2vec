# Genome Transformer Comparison

A repository for generating embeddings from genome sequences

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