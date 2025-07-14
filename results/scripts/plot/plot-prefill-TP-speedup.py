import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Read the data ------------------------------------------------
# -------------------------------------------------------------
# Excel file was uploaded by the user.  It contains (for each TP)
# the measured ground?truth latency and the simulator?predicted
# latency for two different models.
FILE_PATH = "prefill-TP.xlsx"

# Read the entire workbook (just in case there are multiple sheets)
# then concatenate them together for convenience.
raw_sheets = pd.read_excel(FILE_PATH, sheet_name=None)
df = pd.concat(raw_sheets.values(), ignore_index=True)

# The first *real* header row is duplicated inside the sheet; we
# remove that and give the columns explicit names.
df.columns = ["Model", "Arch", "TP", "Ground truth (ms)", "Simulation (ms)"]
df = df[df["Model"] != "Model"]               # Drop the duplicate header row

# Make sure numeric columns are actually numeric.
df["TP"] = df["TP"].astype(int)
df["Ground truth (ms)"] = df["Ground truth (ms)"].astype(float)
df["Simulation (ms)"] = df["Simulation (ms)"].astype(float)

# -------------------------------------------------------------
# Compute speed?up numbers ------------------------------------
# -------------------------------------------------------------
# Speed?up is defined as (latency at TP = 1) / (latency at TP = k).
# We calculate it separately for ground?truth and simulator data.
speedup_frames = []
for model_name, sub_df in df.groupby("Model"):
    sub_df = sub_df.sort_values("TP").copy()
    base_gt = sub_df.loc[sub_df["TP"] == 1, "Ground truth (ms)"].iloc[0]
    base_sim = sub_df.loc[sub_df["TP"] == 1, "Simulation (ms)"].iloc[0]

    sub_df["GT_speedup"] = base_gt / sub_df["Ground truth (ms)"]
    sub_df["Sim_speedup"] = base_sim / sub_df["Simulation (ms)"]
    speedup_frames.append(sub_df)

speedup_df = pd.concat(speedup_frames, ignore_index=True)

# -------------------------------------------------------------
# Plot --------------------------------------------------------
# -------------------------------------------------------------
# One figure per model (no sub?plots).  Each figure contains two
# lines: ground?truth and simulation.
for model_name, sub_df in speedup_df.groupby("Model"):
    plt.figure()  # new figure  complies with "no subplots" rule
    plt.plot(
        sub_df["TP"],
        sub_df["GT_speedup"],
        marker="s",
        linestyle="-",
        label="Ground Truth",
    )
    plt.plot(
        sub_df["TP"],
        sub_df["Sim_speedup"],
        marker="o",
        linestyle="--",
        label="Simulation",
    )
    plt.xlabel("TP")              # X?axis label
    plt.ylabel("Speedup")         # Y?axis label
    plt.title(f"{model_name} - Speedup vs TP")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.4)

# Display the figures in the notebook
plt.show()
