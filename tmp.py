from pathlib import Path
import shutil

import pandas as pd

# read csv of genome IDs to embed
df = pd.read_csv('ecoli_ampicillin_250_balanced_genome_ids.csv')
ids_to_embed = df['genome_id']

src_dir = Path("/home/oskar/data/bacterial_sequences/resistant_phenotypes")
dst_dir = Path("/home/oskar/data/bacterial_sequences/ampicillin_resistant_ecoli")
dst_dir.mkdir(parents=True, exist_ok=True)

for gid in ids_to_embed:
    file = src_dir / f"{gid}.fna"
    if file.exists():
        shutil.copy(file, dst_dir / file.name)
    else:
        print(f"Missing: {file.name}")
