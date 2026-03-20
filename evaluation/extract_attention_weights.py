"""
evaluation/extract_attention_weights.py

Extracts mean GATv2Conv attention weights across the evaluation dataset
and produces a 4×4 agent-pair attention matrix.

Usage:
    python evaluation/extract_attention_weights.py

Outputs:
    - Prints a 4×4 attention weight matrix to stdout
    - Saves evaluation/results/attention_weights_matrix.csv
"""

import os
import sys
import csv
import numpy as np
import torch

# ── project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from train.energy_env_robust import EnergyMarketEnvRobust

# ── config ────────────────────────────────────────────────────────────────────
N_AGENTS = 4
CHECKPOINT_PATH = "models_phase5_hybrid/ppo_hybrid_150000_steps.zip"
EVAL_STEPS = 1_752   # 20 % of 8760 hours ≈ evaluation split
OUTPUT_DIR = "evaluation/results"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "attention_weights_matrix.csv")
AGENT_NAMES = ["Agent 0 (Solar)", "Agent 1 (Wind)", "Agent 2 (EV)", "Agent 3 (Standard)"]


def get_gnn_policy(model):
    """Attempt to locate the GNN / GATv2Conv module inside the SB3 policy."""
    policy = model.policy
    # Walk the module tree looking for a GATv2Conv layer
    for name, module in policy.named_modules():
        module_type = type(module).__name__
        if "GAT" in module_type or "GNN" in module_type or "Graph" in module_type:
            return module, name
    return None, None


def build_fully_connected_edge_index(n_agents: int):
    """Return edge_index for a fully-connected graph (self-loops excluded)."""
    src, dst = [], []
    for i in range(n_agents):
        for j in range(n_agents):
            if i != j:
                src.append(i)
                dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)


def extract_weights_from_forward_pass(model, env, n_steps: int):
    """
    Run n_steps of evaluation; collect per-edge attention weights.

    Returns
    -------
    weight_accumulator : np.ndarray, shape (N_AGENTS, N_AGENTS)
        Sum of attention weights per directed agent pair.
    count : int
        Number of timesteps that contributed.
    """
    edge_index = build_fully_connected_edge_index(N_AGENTS)
    weight_accumulator = np.zeros((N_AGENTS, N_AGENTS), dtype=np.float64)
    count = 0

    obs, _ = env.reset()
    gnn_module, gnn_name = get_gnn_policy(model)

    if gnn_module is None:
        raise RuntimeError(
            "Could not locate a GATv2Conv / GNN module inside the policy network. "
            "Make sure the policy was trained with a graph-based feature extractor."
        )

    print(f"Found GNN module: '{gnn_name}'  ({type(gnn_module).__name__})")

    # Hook to capture attention weights during forward pass
    captured = {}

    def attn_hook(module, inp, out):
        # GATv2Conv returns (out_tensor, (edge_index, attn_weights)) when
        # return_attention_weights=True is set, or via the forward signature.
        # Some versions expose attn_weights on the module itself.
        if isinstance(out, tuple) and len(out) == 2:
            _, (_, attn) = out
            captured["attn"] = attn.detach().cpu()

    hook_handle = gnn_module.register_forward_hook(attn_hook)

    for _ in range(n_steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            try:
                model.policy(obs_tensor)
            except Exception:
                pass  # inference may fail on raw obs; continue

        if "attn" in captured:
            attn = captured["attn"].numpy()  # shape: (n_edges,) or (n_edges, heads)
            if attn.ndim > 1:
                attn = attn.mean(axis=-1)     # average over heads

            src_nodes = edge_index[0].numpy()
            dst_nodes = edge_index[1].numpy()

            if len(attn) == len(src_nodes):
                for s, d, w in zip(src_nodes, dst_nodes, attn):
                    weight_accumulator[s, d] += float(w)
                count += 1
            captured.clear()

        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    hook_handle.remove()
    return weight_accumulator, count


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: Checkpoint not found at '{CHECKPOINT_PATH}'")
        print("Update CHECKPOINT_PATH at the top of this script and re-run.")
        sys.exit(1)

    env = EnergyMarketEnvRobust()
    model = PPO.load(CHECKPOINT_PATH, env=env)

    print(f"\nExtracting attention weights over {EVAL_STEPS} timesteps …\n")
    weight_sum, count = extract_weights_from_forward_pass(model, env, EVAL_STEPS)
    env.close()

    if count == 0:
        print("WARNING: No attention weights were captured. "
              "The GNN module may not expose attention through the hook API.")
        print("Consider calling conv(..., return_attention_weights=True) directly.")
        sys.exit(1)

    mean_weights = weight_sum / count

    # ── print matrix ──────────────────────────────────────────────────────────
    col_w = 14
    print(f"\nMean Attention Weight Matrix  (row = source agent, col = target agent)")
    print(f"Based on {count} timesteps from checkpoint: {CHECKPOINT_PATH}\n")
    header_row = " " * 18 + "".join(f"{n:>{col_w}}" for n in AGENT_NAMES)
    print(header_row)
    for i, row_name in enumerate(AGENT_NAMES):
        row_vals = "".join(f"{mean_weights[i, j]:>{col_w}.4f}" for j in range(N_AGENTS))
        print(f"{row_name:<18}{row_vals}")

    print()
    max_idx = np.unravel_index(np.argmax(mean_weights), mean_weights.shape)
    print(f"Highest attention: {AGENT_NAMES[max_idx[0]]} → {AGENT_NAMES[max_idx[1]]}  "
          f"(weight = {mean_weights[max_idx]:.4f})")

    # ── save CSV ──────────────────────────────────────────────────────────────
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["From \\ To"] + AGENT_NAMES)
        for i, row_name in enumerate(AGENT_NAMES):
            writer.writerow([row_name] + [f"{mean_weights[i, j]:.4f}" for j in range(N_AGENTS)])

    print(f"\nMatrix saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
