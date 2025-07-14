import pandas as pd
import matplotlib.pyplot as plt
import re

# -------------------------------------------------------------
# CONFIG -------------------------------------------------------
# -------------------------------------------------------------
FILE = "decode-PP.xlsx"  
HEADER_ROW = 15                
GT_COL_CANDIDATES  = ["GT-speedup", "GT_speedup", "Ground truth speedup"]
SIM_COL_CANDIDATES = ["simulator-speedup", "simulation-speedup", "Sim_speedup"]

# -------------------------------------------------------------
# LOAD & CLEAN -------------------------------------------------
# -------------------------------------------------------------
# Read every sheet (in case user has multiple BS splits)
sheets = pd.read_excel(FILE, sheet_name=None, header=HEADER_ROW)
df = pd.concat(sheets.values(), ignore_index=True)

# Trim column names
df.columns = [str(c).strip() for c in df.columns]

# Identify key columns
def find_col(candidates, cols):
    for c in candidates:
        for real in cols:
            if real.strip().lower() == c.strip().lower():
                return real
    raise KeyError(f"None of {candidates} found in {cols}")

gt_col  = find_col(GT_COL_CANDIDATES,  df.columns)
sim_col = find_col(SIM_COL_CANDIDATES, df.columns)

# Figure out batch size column (BS / BatchSize)
bs_col_candidates = ["BS", "BatchSize", "batch_size"]
bs_col = find_col(bs_col_candidates, df.columns)

# Ensure pp_size column name
pp_col = find_col(["pp_size", "PP_degree", "pp"], df.columns)

# Filter: TP == 1 (some sheet may have that col)
if "TP" in df.columns:
    df = df[df["TP"] == 1]

# Keep essentials
df = df[[pp_col, bs_col, gt_col, sim_col]].dropna()

# Cast numeric
df[pp_col]  = pd.to_numeric(df[pp_col], errors="coerce").astype(int)
df[bs_col]  = pd.to_numeric(df[bs_col], errors="coerce").astype(int)
df[gt_col]  = pd.to_numeric(df[gt_col], errors="coerce")
df[sim_col] = pd.to_numeric(df[sim_col], errors="coerce")
df = df.dropna()

# -------------------------------------------------------------
# PLOT ---------------------------------------------------------
# -------------------------------------------------------------
def slug(s): return re.sub(r"[^\w\-]", "_", s)

for bs in [1, 8, 16]:
    subset = df[df[bs_col] == bs].sort_values(pp_col)
    if subset.empty:
        continue

    x = subset[pp_col].values
    gt = subset[gt_col].values
    sim = subset[sim_col].values

    idx = range(len(x))
    bw  = 0.35

    plt.figure(figsize=(5, 3))
    plt.bar([i - bw/2 for i in idx], gt,  width=bw, label="Ground Truth")
    plt.bar([i + bw/2 for i in idx], sim, width=bw, label="Simulation")

    plt.xticks(idx, x)
    plt.xlabel("PP degree")
    plt.ylabel("Speedup")
    plt.title(f"Speedup vs PP degree (BS={bs})")
    plt.legend(fontsize="small")
    plt.grid(axis="y", linestyle="--", linewidth=0.4)
    plt.tight_layout()
    plt.savefig(f"PP-speedup-BS{bs}.png", dpi=300)

plt.show()
