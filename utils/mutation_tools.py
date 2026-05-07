import random
from pathlib import Path


BASES = ["A", "C", "G", "T"]

def parse_fasta_contigs(file_path: str) -> list[tuple[str, str]]:
    """
    Parse FASTA file preserving individual contigs.
    Contigs are therefore kept seperate from one another.

    Parameters
    ----------
    file_path : str
        Path to the FASTA assembly file.

    Returns
    -------
    list of (header, sequence) tuples, one per contig.
    """
    contigs = []
    header = None
    seq = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    contigs.append((header, "".join(seq)))
                header = line[1:]
                seq = []
            else:
                seq.append(line.upper())
    if header is not None:
        contigs.append((header, "".join(seq)))
    return contigs



def mutate_sequence(seq: str, n_snps: int, seed: int = 42) -> tuple[str, list[int]]:
    """
    Introduce exactly n_snps point mutations at random positions.
 
    Parameters
    ----------
    seq : str
        Original DNA sequence (uppercase ACGT only).
    n_snps : int
        Number of SNPs to introduce.
    seed : int
        Random seed for reproducibility.
 
    Returns
    -------
    mutated : str
        Mutated sequence.
    positions : list[int]
        Sorted list of positions that were mutated.
    """
    if n_snps == 0:
        return seq, []
 
    if n_snps > len(seq):
        raise ValueError(f"n_snps ({n_snps}) exceeds sequence length ({len(seq)})")
 
    random.seed(seed)
    seq_list = list(seq)
    positions = random.sample(range(len(seq_list)), n_snps)
 
    for pos in positions:
        original = seq_list[pos]
        alternatives = [b for b in BASES if b != original]
        seq_list[pos] = random.choice(alternatives)
 
    mutated = "".join(seq_list)
    return mutated, sorted(positions)
 
 
def hamming_distance(s1: str, s2: str) -> int:
    """Count positions where two equal-length strings differ."""
    return sum(a != b for a, b in zip(s1, s2))
 
 
def get_affected_chunks(positions: list[int], chunk_size: int) -> list[int]:
    """
    Given mutation positions in the full genome string,
    return the sorted list of chunk indices that contain at least one mutation.
 
    This is ground truth — used later to validate whether the cosine
    similarity matrix correctly identifies the divergent chunks.
    """
    return sorted(set(pos // chunk_size for pos in positions))
 
 
# ---------------------------------------------------------------------------
# FASTA writing
# ---------------------------------------------------------------------------
 
def write_fasta(path: Path, header: str, sequence: str) -> None:
    """Write a single-sequence FASTA file with 80-char line wrapping."""
    with open(path, "w") as f:
        f.write(f">{header}\n")
        for i in range(0, len(sequence), 80):
            f.write(sequence[i:i + 80] + "\n")