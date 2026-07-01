#!/usr/bin/env python3
"""Result bar charts for the README (OSWorld + Minecraft).

Data from the paper's tab:main and tab:minecraft.
Palette follows the report style (violet accents).
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
    "font.weight": "bold", "axes.labelweight": "bold",
    "axes.edgecolor": "#3a3350", "axes.linewidth": 1.0,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# report palette (from structagent_report.sty)
OURS = "#7C3AED"       # violet-600 — StructAgent highlight
OURS_EDGE = "#5B21B6"
INK = "#2B2540"
GAIN = "#1E7D46"       # green relative-gain callout
# muted violet-grey ramp for baselines
SINGLE = "#D8CFEA"
AS3 = "#B7A6DA"
OSS = "#9A84C8"
VLAA = "#6E6685"
FRONT_BASE = "#B4A7D2"  # frontier baselines (muted)

fig = plt.figure(figsize=(16.5, 4.5))
gs = fig.add_gridspec(1, 2, width_ratios=[1.9, 1.5], wspace=0.28)
axL = fig.add_subplot(gs[0, 0])
axR = fig.add_subplot(gs[0, 1])

# ── Left: open backbones (self-run), StructAgent vs frameworks ──────────────
methods = [("Single", SINGLE), ("Agent-S3", AS3), ("OS-Symphony", OSS),
           ("VLAA-GUI", VLAA), ("StructAgent", OURS)]
data9 = [27.0, 40.8, 43.3, 40.2, 46.9]
data27 = [31.6, 60.2, 52.8, 48.2, 62.2]
groups = [("Qwen3.5-9B", data9), ("Qwen3.5-27B", data27)]
w = 0.155
YB = 20  # zoomed baseline so relative gaps read larger
xticks, xlabels = [], []
x = 0.0
for gname, vals in groups:
    for i, (v, (mname, col)) in enumerate(zip(vals, methods)):
        bx = x + i * w
        ours = mname == "StructAgent"
        axL.bar(bx, v - YB, w * 0.92, bottom=YB, color=col, zorder=3,
                edgecolor=(OURS_EDGE if ours else "white"), linewidth=(1.6 if ours else 0.9))
        axL.text(bx, v + 0.8, f"{v:.1f}", ha="center", va="bottom",
                 fontsize=11, fontweight="bold", color=(INK if ours else "#5a5568"))
        if ours:
            gain = (v - vals[0]) / vals[0] * 100
            axL.annotate(f"+{gain:.0f}%", xy=(bx, v + 4.2), ha="center", va="bottom",
                         fontsize=14, fontweight="bold", color=GAIN)
    xticks.append(x + 2 * w); xlabels.append(gname)
    x += 5 * w + 0.22

axL.set_xticks(xticks); axL.set_xticklabels(xlabels, fontsize=14.5, fontweight="bold")
axL.set_ylim(YB, 74)
axL.set_ylabel("OSWorld-Verified success rate (%)", fontsize=12.5, fontweight="bold")
axL.set_title("Open backbones",
              fontsize=15, fontweight="bold", pad=10, loc="left", color=INK)
for t in axL.get_yticklabels():
    t.set_fontsize(10.5); t.set_fontweight("bold")
axL.set_axisbelow(True); axL.yaxis.grid(True, color="#ece9f5", lw=0.9, zorder=0)
for s in ("top", "right"):
    axL.spines[s].set_visible(False)
leg = [Patch(fc=c, ec=("none"), label=m) for m, c in methods[:-1]]
leg.append(Patch(fc=OURS, ec=OURS_EDGE, label="StructAgent (ours)"))
axL.legend(handles=leg, ncol=5, fontsize=9.5, frameon=False, loc="upper center",
           bbox_to_anchor=(0.5, -0.11), handlelength=1.0, columnspacing=1.0, handletextpad=0.5)

# ── Right: frontier backbones (reported) — our open agent tops them ─────────
front = [
    ("Claude Sonnet 4.6", 72.1, FRONT_BASE),
    ("Agent-S3 (Opus/GPT)", 72.6, FRONT_BASE),
    ("Qwen3.7-Plus", 73.3, FRONT_BASE),
    ("MiniMax-M3 (single)", 75.2, FRONT_BASE),
    ("VLAA-GUI (Opus 4.5)", 76.3, FRONT_BASE),
    ("StructAgent (M3, ours)", 78.9, OURS),
]
YBR = 62
xr = np.arange(len(front))
# ascending violet gradient — climbs to the deepest (StructAgent) bar
ramp = LinearSegmentedColormap.from_list("v", ["#EFE9FA", "#BCA9E8", "#7C3AED", "#5B21B6"])
n = len(front)
for xi, (name, v, _) in zip(xr, front):
    ours = "ours" in name
    col = ramp(xi / (n - 1))
    axR.bar(xi, v - YBR, 0.66, bottom=YBR, color=col, zorder=3,
            edgecolor=(OURS_EDGE if ours else "white"), linewidth=(1.6 if ours else 0.9))
    axR.text(xi, v + 0.18, f"{v:.1f}", ha="center", va="bottom",
             fontsize=11, fontweight="bold", color=(INK if ours else "#5a5568"))
# gold star + "best overall" badge above our bar
axR.plot(xr[-1] - 0.34, 80.6, marker="*", ms=16, color="#F2A900",
         mec=OURS_EDGE, mew=0.7, zorder=6, clip_on=False)
axR.text(xr[-1] - 0.20, 80.5, "best overall", ha="left", va="center",
         fontsize=12, fontweight="bold", color=GAIN)
axR.set_xticks(xr)
axR.set_xticklabels([f[0] for f in front], fontsize=9.5, fontweight="bold",
                    color=INK, rotation=24, ha="right", rotation_mode="anchor")
axR.set_ylim(YBR, 82)
axR.set_title("Frontier backbones",
              fontsize=15, fontweight="bold", pad=10, loc="left", color=INK)
for t in axR.get_yticklabels():
    t.set_fontsize(10.5); t.set_fontweight("bold")
axR.set_axisbelow(True); axR.yaxis.grid(True, color="#ece9f5", lw=0.9, zorder=0)
for s in ("top", "right"):
    axR.spines[s].set_visible(False)

fig.savefig("assets/results_osworld.png", dpi=200, bbox_inches="tight", facecolor="white")
print("wrote assets/results_osworld.png")

# ── Minecraft: StructAgent vs Optimus-1, 5 tiers ────────────────────────────
tiers = ["Wooden", "Stone", "Iron", "Golden", "Redstone"]
ours_sr = [100.0, 70.0, 64.7, 85.7, 57.1]
opt_sr = [98.6, 92.4, 46.7, 8.5, 25.0]
fig2, ax2 = plt.subplots(figsize=(9.8, 3.4))
xi = np.arange(len(tiers)); bw = 0.37
ax2.bar(xi - bw / 2, ours_sr, bw, color=OURS, zorder=3, edgecolor=OURS_EDGE,
        linewidth=1.3, label="StructAgent (ours)")
ax2.bar(xi + bw / 2, opt_sr, bw, color="#BCB0CE", zorder=3, edgecolor="white",
        linewidth=0.9, label="Optimus-1")
for xs, v in zip(xi - bw / 2, ours_sr):
    ax2.text(xs, v + 1.3, f"{v:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold", color=INK)
for xs, v in zip(xi + bw / 2, opt_sr):
    ax2.text(xs, v + 1.3, f"{v:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold", color="#5a5568")
ax2.set_xticks(xi); ax2.set_xticklabels(tiers, fontsize=13, fontweight="bold", color=INK)
ax2.set_ylim(0, 112)
ax2.set_ylabel("Success rate (%)", fontsize=12.5, fontweight="bold")
ax2.set_title("Minecraft generalization: success rate by crafting tier",
              fontsize=15, fontweight="bold", pad=12, loc="left", color=INK)
for t in ax2.get_yticklabels():
    t.set_fontsize(11); t.set_fontweight("bold")
ax2.set_axisbelow(True); ax2.yaxis.grid(True, color="#ece9f5", lw=0.9, zorder=0)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
ax2.legend(fontsize=11.5, frameon=False, loc="upper right", handlelength=1.3)
fig2.tight_layout()
fig2.savefig("assets/results_minecraft.png", dpi=200, bbox_inches="tight", facecolor="white")
print("wrote assets/results_minecraft.png")
