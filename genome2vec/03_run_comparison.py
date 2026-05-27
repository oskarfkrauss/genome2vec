"""
03_run_comparison.py

Loads pre-computed CLS token embeddings for the original genome and all
mutated versions, then tests three comparison options:

    Option 1 — Cross attention directly on all CLS tokens (A vs B)
    Option 2 — Cosine filtering then cross attention on divergent chunks only
    Option 3 — Cosine filtering then token-level cross attention on divergent
                chunks (re-embeds raw chunks at full token resolution)

NOTE ON UNTRAINED WEIGHTS
--------------------------
The cross attention modules have random untrained weights at this stage.
What we are validating here is NOT whether the MLP outputs a meaningful
distance — it is whether the cosine similarity signal and chunk detection
are working correctly. Specifically:

    1. Does mean_best_match_cosine decrease as SNP count increases?
    2. Does filter_divergent_chunks find the correct chunks?
    3. Is opt2_recall high even at low SNP counts (5 SNPs)?

Training the cross attention heads against real SNP distances comes later.

Usage:
    python genome2vec/03_run_comparison.py genome2vec/configs/comparison_config.yaml
"""

import os
import sys
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from utils.transformer_tools import TRANSFORMER_MODEL_ARGS
from utils.pooling_tools import (
    cosine_similarity_matrix,
    filter_divergent_chunks,
    Option1CrossAttention,
    Option2FilteredCrossAttention,
    Option3TokenLevelCrossAttention,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Token-level re-embedding for Option 3
# ---------------------------------------------------------------------------

def get_token_embeddings(
    tokenizer: AutoTokenizer,
    model: AutoModel,
    sequence: str,
    device: torch.device,
) -> torch.Tensor:
    """
    Re-embed a single chunk and return ALL token embeddings (not just CLS).
    Used by Option 3 for token-level cross attention.

    Returns
    -------
    token_embeddings : [seq_len, hidden_dim]
    """
    tokens = tokenizer(
        sequence,
        return_tensors="pt",
        padding=False,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        out = model(**tokens, output_hidden_states=True)

    # all tokens from last hidden state, remove batch dim
    return out.hidden_states[-1].squeeze(0).cpu()


# ---------------------------------------------------------------------------
# Chunk detection accuracy
# ---------------------------------------------------------------------------

def chunk_detection_accuracy(
    predicted_divergent: list[int],
    ground_truth_affected: list[int],
) -> dict:
    """
    Compare which chunks the model flagged as divergent against ground truth.

    Returns precision, recall, and whether all affected chunks were found.
    This is the key validation metric — are we finding the right chunks?
    """
    predicted_set = set(predicted_divergent)
    truth_set = set(ground_truth_affected)

    if len(predicted_set) == 0:
        precision = 0.0
    else:
        precision = len(predicted_set & truth_set) / len(predicted_set)

    if len(truth_set) == 0:
        recall = 1.0
    else:
        recall = len(predicted_set & truth_set) / len(truth_set)

    return {
        "precision":         precision,
        "recall":            recall,
        "all_affected_found": truth_set.issubset(predicted_set),
        "n_predicted":       len(predicted_set),
        "n_ground_truth":    len(truth_set),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(config_path: Path) -> None:
    config = load_config(config_path)

    embeddings_dir  = Path(config["embeddings_dir"])
    manifest_path   = Path(config["manifest_path"])
    output_csv      = Path(config["output_csv"])
    model_name      = config["transformer_model"]
    low_threshold   = config.get("low_threshold", 0.3)
    high_threshold  = config.get("high_threshold", 0.99)
    run_option3     = config.get("run_option3", False)

    # hidden dim depends on which model was used to generate embeddings
    hidden_dim = 2560 if "2.5B" in model_name else 768
    num_heads  = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Thresholds: low={low_threshold}, high={high_threshold}")
    print()

    # --- Load NT model only if running Option 3 ---
    tokenizer = None
    nt_model  = None
    if run_option3:
        print("Loading tokenizer and model for Option 3 token-level re-embedding...")
        tokenizer = AutoTokenizer.from_pretrained(
            TRANSFORMER_MODEL_ARGS[model_name]["remote_path"],
            trust_remote_code=True,
        )
        nt_model = AutoModel.from_pretrained(
            TRANSFORMER_MODEL_ARGS[model_name]["remote_path"],
            trust_remote_code=True,
        ).to(device)
        nt_model.eval()
        print("Model loaded.\n")

    # --- Initialise comparators (untrained at this stage) ---
    opt1 = Option1CrossAttention(hidden_dim, num_heads).to(device)
    opt2 = Option2FilteredCrossAttention(hidden_dim, num_heads).to(device)
    opt3 = Option3TokenLevelCrossAttention(hidden_dim, num_heads).to(device) if run_option3 else None

    # --- Read manifest ---
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        manifest = list(reader)

    # --- Load original embedding ---
    original_row = next(r for r in manifest if int(r["snp_count"]) == 0)
    original_stem = Path(original_row["fasta_file"]).stem
    original_emb_path = embeddings_dir / f"{original_stem}.pt"

    if not original_emb_path.exists():
        raise FileNotFoundError(
            f"Original embedding not found: {original_emb_path}\n"
            f"Have you run 02_generate_embeddings.py first?"
        )

    print(f"Loading original embedding: {original_emb_path.name}")
    emb_original = torch.load(original_emb_path).to(device)
    print(f"Original embedding shape: {emb_original.shape}")
    print()

    results = []

    for row in manifest:
        snp_count = int(row["snp_count"])
        replicate = int(row["replicate"])

        if snp_count == 0:
            continue

        fasta_stem = Path(row["fasta_file"]).stem
        emb_path = embeddings_dir / f"{fasta_stem}.pt"

        if not emb_path.exists():
            print(f"WARNING: embedding not found for {fasta_stem}, skipping")
            continue

        emb_mutated = torch.load(emb_path).to(device)

        ground_truth_chunks = (
            [int(x) for x in row["affected_chunks"].split(",")]
            if row["affected_chunks"] else []
        )

        print(f"snp={snp_count:>5} | rep={replicate} | ground truth chunks: {ground_truth_chunks}")

        result = {
            "snp_count":           snp_count,
            "replicate":           replicate,
            "ground_truth_chunks": str(ground_truth_chunks),
        }

        # --- Baseline: cosine similarity between mean-pooled embeddings ---
        # This is the naive single-vector comparison — useful as a lower bound
        mean_a = F.normalize(emb_original.mean(dim=0), dim=0)
        mean_b = F.normalize(emb_mutated.mean(dim=0), dim=0)
        result["baseline_mean_pool_cosine"] = F.cosine_similarity(
            mean_a, mean_b, dim=0
        ).item()

        # --- Compute full cosine similarity matrix ---
        sim_matrix = cosine_similarity_matrix(emb_original, emb_mutated)

        # best match score for each chunk in A
        best_scores, _ = sim_matrix.max(dim=1)
        result["mean_best_match_cosine"] = best_scores.mean().item()
        result["min_best_match_cosine"]  = best_scores.min().item()

        # --- Option 1: cross attention on all chunks ---
        with torch.no_grad():
            opt1_distance, _ = opt1(emb_original, emb_mutated)
        result["opt1_distance"] = opt1_distance.item()

        # --- Option 2: cosine filter then cross attention ---
        div_idx_a, div_idx_b, _ = filter_divergent_chunks(
            sim_matrix, low_threshold, high_threshold
        )

        n_divergent = len(div_idx_a)
        result["opt2_n_divergent_chunks"]        = n_divergent
        result["opt2_divergent_chunk_indices_a"] = str(div_idx_a.tolist())

        if n_divergent > 0:
            div_emb_a = emb_original[div_idx_a]
            div_emb_b = emb_mutated[div_idx_b]

            with torch.no_grad():
                opt2_distance, _ = opt2(div_emb_a, div_emb_b)
            result["opt2_distance"] = opt2_distance.item()

            acc = chunk_detection_accuracy(div_idx_a.tolist(), ground_truth_chunks)
            result["opt2_precision"]          = round(acc["precision"], 4)
            result["opt2_recall"]             = round(acc["recall"], 4)
            result["opt2_all_affected_found"] = acc["all_affected_found"]
        else:
            print(f"  WARNING: no divergent chunks found — consider lowering high_threshold")
            result["opt2_distance"]           = None
            result["opt2_precision"]          = None
            result["opt2_recall"]             = None
            result["opt2_all_affected_found"] = None

        # --- Option 3: token-level cross attention on divergent chunks ---
        if run_option3 and n_divergent > 0:
            chunk_size    = int(row["chunk_size"])
            original_seq  = "".join(
                seq for _, seq in
                [line.strip() for line in open(original_row["fasta_file"])]
                if not seq.startswith(">")
            )
            mutated_seq = "".join(
                seq for _, seq in
                [line.strip() for line in open(row["fasta_file"])]
                if not seq.startswith(">")
            )

            opt3_distances = []
            for ca_idx, cb_idx in zip(div_idx_a.tolist(), div_idx_b.tolist()):
                seq_a = original_seq[ca_idx * chunk_size:(ca_idx + 1) * chunk_size]
                seq_b = mutated_seq[cb_idx * chunk_size:(cb_idx + 1) * chunk_size]

                tok_a = get_token_embeddings(tokenizer, nt_model, seq_a, device).to(device)
                tok_b = get_token_embeddings(tokenizer, nt_model, seq_b, device).to(device)

                with torch.no_grad():
                    dist, _ = opt3(tok_a, tok_b)
                opt3_distances.append(dist.item())

            result["opt3_mean_distance"]      = sum(opt3_distances) / len(opt3_distances)
            result["opt3_n_chunks_compared"]  = len(opt3_distances)
        else:
            result["opt3_mean_distance"]     = None
            result["opt3_n_chunks_compared"] = None

        results.append(result)

        # --- Print summary for this pair ---
        print(
            f"  baseline_cosine={result['baseline_mean_pool_cosine']:.6f} | "
            f"mean_best_match={result['mean_best_match_cosine']:.6f} | "
            f"min_best_match={result['min_best_match_cosine']:.6f} | "
            f"divergent={n_divergent}"
        )
        if n_divergent > 0:
            print(
                f"  opt2_precision={result.get('opt2_precision')} | "
                f"opt2_recall={result.get('opt2_recall')} | "
                f"all_found={result.get('opt2_all_affected_found')}"
            )
        print()

    # --- Save results ---
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        print("No results to save — check embeddings directory and manifest.")
        return

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved: {output_csv}")
    print(f"Total pairs compared: {len(results)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python genome2vec/03_run_comparison.py "
            "genome2vec/configs/comparison_config.yaml"
        )
    main(Path(sys.argv[1]))