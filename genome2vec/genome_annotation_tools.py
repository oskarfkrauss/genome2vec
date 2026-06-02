import json
import os
import sys
import shutil

import subprocess
from pathlib import Path
import yaml


def annotate_genomes(fasta_path: Path, bakta_db_path: str, threads: int):
    """
    Run Bakta annotation on a single fasta file

    Parameters
    ----------
    fasta_path : Path
        Path to fasta assembly file to be annotated

    threads : int
        Number of threads to run the annotation on

    Returns
    -------
    dict
        A JSON-serializable dictionary containing the parsed Bakta 
        annotation results.
    """
    # save to temporary dirrectory within 'annotations' foler
    bakta_db = Path(bakta_db_path)
    annotation_output_dir = os.path.join(os.path.dirname(bakta_db), 'annotation_results_tmp')
    os.makedirs(annotation_output_dir, exist_ok=True)

    sample = fasta_path.stem

    cmd = [
        "bakta",
        "--db", str(bakta_db),
        "--output", str(os.path.join(annotation_output_dir, sample)),
        "--threads", str(threads),
        str(fasta_path),
        "--force"
    ]
    # logic for subprocess to catch errors
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    annotation_json_path = os.path.join(annotation_output_dir, sample, f"{sample}.json")

    # load the annotations into a dictionary
    with open(annotation_json_path, "r") as f:
        bakta_data = json.load(f)

    # delete the entire temporary directory and its contents
    shutil.rmtree(annotation_output_dir)

    return bakta_data