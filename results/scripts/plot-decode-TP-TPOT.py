import pandas as pd
import matplotlib.pyplot as plt
import re
# -------------------------------------------------------------
# CONFIG -------------------------------------------------------
# -------------------------------------------------------------
EXCEL_FILE = "decode-TP.xlsx"

# We expect the spreadsheet to contain at least:
#   - a column that tells which LLM model (e.g. "Model")
#   - "BatchSize" (values 1, 8, 16)
#   - "TP"          (tensor?parallel degree)
#   - ground?truth latency column (name contains "GT" or "Ground")
#   - simulator latency column   (name contains "Sim")
#
# The script tries to infer the ground?truth / simulation columns
# automatically.  If it fails, you can hard?code them below:
GT_COL_CANDIDATES  = ["Ground truth(ms)"] # Ground truth(ms)
SIM_COL_CANDIDATES = ["simulator(ms)"]

# -------------------------------------------------------------
# Load workbook ------------------------------------------------
# -------------------------------------------------------------
sheets = pd.read_excel(EXCEL_FILE, sheet_name=None)
df = pd.concat(sheets.values(), ignore_index=True)

# If the sheet *repeats* a header row inside the data (common
# when people export), drop those rows.
if isinstance(df.iloc[0, 0], str) and (df.iloc[0, 0].strip().lower() == "model"):
    df = df[df.iloc[:, 0] != "Model"]

# Make sure column names are trimmed
df.columns = [str(c).strip() for c in df.columns]

# Detect the key columns --------------------------------------------------
def _find_col(candidates, all_cols):
    for c in candidates:
        for real in all_cols:
            if real.strip().lower() == c.strip().lower():
                return real
    raise KeyError(f"Cannot find any of {candidates} in spreadsheet")

gt_col  = _find_col(GT_COL_CANDIDATES, df.columns)
sim_col = _find_col(SIM_COL_CANDIDATES, df.columns)

# Give them uniform names
df = df.rename(columns={gt_col: "GT_ms", sim_col: "SIM_ms"})

# Ensure numeric dtype
df["TP"]        = pd.to_numeric(df["TP"], errors="coerce")
df["BS"] = pd.to_numeric(df["BS"], errors="coerce")
df["GT_ms"]     = pd.to_numeric(df["GT_ms"], errors="coerce")
df["SIM_ms"]    = pd.to_numeric(df["SIM_ms"], errors="coerce")

df = df.dropna(subset=["TP", "BS", "GT_ms", "SIM_ms"])

# -------------------------------------------------------------
# PLOT ---------------------------------------------------------
# -------------------------------------------------------------
def _slugify(s: str) -> str:
    """Turn model name into safe file name."""
    return re.sub(r"[^\w\-]", "_", s)


# for model_name, g_model in df.groupby("Model"):
#     for bs in [1, 8, 16]:
#         g_bs = g_model[g_model["BS"] == bs].sort_values("TP")
#         if g_bs.empty:
#             continue  # Skip if this batch size doesn't exist

#         plt.figure(figsize=(5, 3))  # each plot is independent
#         plt.plot(
#             g_bs["TP"],
#             g_bs["GT_ms"],
#             marker="s",
#             linestyle="-",
#             label="Ground Truth",
#         )
#         plt.plot(
#             g_bs["TP"],
#             g_bs["SIM_ms"],
#             marker="o",
#             linestyle="--",
#             label="Simulation",
#         )

#         plt.title(f"{model_name} - BS={bs}")
#         plt.xlabel("TP degree")
#         plt.ylabel("TPOP (ms)")
#         plt.legend(fontsize="small")
#         plt.grid(True, linestyle="--", linewidth=0.4)
#         plt.tight_layout()

#         fname = f"{_slugify(model_name)}_BS{bs}.png"
#         plt.savefig(fname, dpi=300)
#         print(f"Saved {fname}")
for model_name, g_model in df.groupby("Model"):
    for bs in [1, 8, 16]:
        g_bs = g_model[g_model["BS"] == bs].sort_values("TP")
        if g_bs.empty or 1 not in g_bs["TP"].values:
            continue                

        base_gt  = g_bs.loc[g_bs["TP"] == 1, "GT_ms"].iloc[0]
        base_sim = g_bs.loc[g_bs["TP"] == 1, "SIM_ms"].iloc[0]

        g_bs = g_bs.assign(
            GT_speedup  = base_gt  / g_bs["GT_ms"],
            SIM_speedup = base_sim / g_bs["SIM_ms"],
        )

        plt.figure(figsize=(5, 3))
        plt.plot(g_bs["TP"], g_bs["GT_speedup"],  marker="s", linestyle="-",  label="Ground Truth")
        plt.plot(g_bs["TP"], g_bs["SIM_speedup"], marker="o", linestyle="--", label="Simulation")

        plt.title(f"{model_name} - BS={bs}")
        plt.xlabel("TP degree")
        plt.ylabel("Speedup")
        plt.legend(fontsize="small")
        plt.grid(True, linestyle="--", linewidth=0.4)
        plt.tight_layout()

        fname = f"{_slugify(model_name)}_BS{bs}.png"
        plt.savefig(fname, dpi=300)
        print(f"Saved {fname}")

plt.show()
