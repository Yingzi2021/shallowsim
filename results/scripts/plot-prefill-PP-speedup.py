# encoding: gb2312

import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 读取 Excel（第 8 行才是表头 → header=7）
# -------------------------------------------------------------
FILE = "prefill-PP.xlsx"          
df = pd.read_excel(FILE, header=7)

# 只保留需要的列
df = df[["pp_size", "GT-speedup", "simulator-speedup"]].dropna()
df["pp_size"] = df["pp_size"].astype(int)   # 保证整数并排序
df = df.sort_values("pp_size")

# -------------------------------------------------------------
# 准备数据
# -------------------------------------------------------------
x_vals = df["pp_size"]
gt_vals  = df["GT-speedup"]
sim_vals = df["simulator-speedup"]

bar_width = 0.35
indices   = range(len(x_vals))

# -------------------------------------------------------------
# 画图
# -------------------------------------------------------------
plt.figure(figsize=(6, 4))

plt.bar([i - bar_width/2 for i in indices], gt_vals,  width=bar_width, label="Ground Truth")
plt.bar([i + bar_width/2 for i in indices], sim_vals, width=bar_width, label="Simulation")

plt.xticks(indices, x_vals)
plt.xlabel("PP degree")
plt.ylabel("Speedup")
plt.title("Speedup vs Pipeline Parallel (PP) Degree")
plt.legend()
plt.grid(axis="y", linestyle="--", linewidth=0.4)
plt.tight_layout()
plt.savefig("prefill-pp-speedup.png", dpi=300)
plt.show()      # 如果想保存，用 plt.savefig("pp_speedup.png", dpi=300)