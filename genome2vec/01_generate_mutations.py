"""
Generate artificial contigs by mutating a 'base' contig for embedding later
"""

import os
import sys
import csv
from pathlib import Path

import yaml

# From your own package
from utils.transformer_tools import TRANSFORMER_MODEL_ARGS
from utils.mutation_tools import (
    mutate_sequence, 
    hamming_distance, 
    get_affected_chunks, 
    write_fasta, 
    parse_fasta_contigs
)



def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
    

def main(config_path: Path) -> None:
    config = load_config(config_path)
 
    fasta_path    = Path(config["base_fasta"])
    output_dir    = Path(config["output_dir"])
    snp_levels    = config["snp_levels"]        # e.g. [0, 5, 10, 50, 100, 500]
    replicates    = config["replicates"]        # e.g. 3
    base_seed     = config.get("base_seed", 42)
    model_name    = config["transformer_model"]
    contig_index  = config.get("contig_index", 0)  # defaults to NODE_1
 
    chunk_size = TRANSFORMER_MODEL_ARGS[model_name]["max_seq_length"]
 
    output_dir.mkdir(parents=True, exist_ok=True)
 
    # --- Parse contigs individually and pick target ---
    print(f"Parsing FASTA: {fasta_path}")
    contigs = parse_fasta_contigs(str(fasta_path))
    print(f"Total contigs found: {len(contigs)}")

    target_header, genome = contigs[contig_index]
    print(f"Using contig {contig_index}: {target_header}")
    print(f"Contig length: {len(genome):,} bp")
    print(f"Expected chunks: {len(genome) // chunk_size + 1}")
    print()
 
    # Validate — only ACGT allowed
    invalid = set(genome) - set("ACGT")
    if invalid:
        print(f"WARNING: unexpected characters in genome: {invalid}")
        print("Filtering to ACGT only.")
        genome = "".join(b if b in "ACGT" else "N" for b in genome)
 
    # --- Write the original (0 SNP) FASTA once ---
    original_fasta = output_dir / f"{fasta_path.stem}_snp0_rep0.fasta"
    write_fasta(original_fasta, header=f"{fasta_path.stem}_snp0_rep0", sequence=genome)
 
    # --- Manifest ---
    manifest_path = output_dir / "manifest.csv"
    manifest_rows = []
 
    # Record the original
    manifest_rows.append({
        "fasta_file":       str(original_fasta),
        "base_genome":      str(fasta_path),
        "snp_count":        0,
        "replicate":        0,
        "mutated_positions": "",
        "affected_chunks":  "",
        "hamming_distance": 0,
        "genome_length":    len(genome),
        "chunk_size":       chunk_size,
    })
 
    # --- Generate mutated FASTAs ---
    for snp_count in snp_levels:
        if snp_count == 0:
            continue  # already written above
 
        for rep in range(replicates):
            seed = base_seed + snp_count * 1000 + rep
 
            mutated, positions = mutate_sequence(genome, snp_count, seed=seed)
 
            # Sanity check
            hd = hamming_distance(genome, mutated)
            assert hd == snp_count, (
                f"Mutation mismatch: expected {snp_count}, got {hd}"
            )
 
            affected = get_affected_chunks(positions, chunk_size)
 
            stem = f"{fasta_path.stem}_snp{snp_count}_rep{rep}"
            out_fasta = output_dir / f"{stem}.fasta"
            write_fasta(out_fasta, header=stem, sequence=mutated)
 
            manifest_rows.append({
                "fasta_file":        str(out_fasta),
                "base_genome":       str(fasta_path),
                "snp_count":         snp_count,
                "replicate":         rep,
                "mutated_positions": ",".join(map(str, positions)),
                "affected_chunks":   ",".join(map(str, affected)),
                "hamming_distance":  hd,
                "genome_length":     len(genome),
                "chunk_size":        chunk_size,
            })
 
            print(
                f"snp={snp_count:>5} | rep={rep} | "
                f"affected chunks: {affected} | "
                f"saved: {out_fasta.name}"
            )
 
    # --- Write manifest ---
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)
 
    print(f"\nManifest saved: {manifest_path}")
    print(f"Total FASTAs written: {len(manifest_rows)}")
 
 
if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python genome2vec/02_generate_mutations.py "
            "genome2vec/configs/mutation_config.yaml"
        )
    main(Path(sys.argv[1]))

