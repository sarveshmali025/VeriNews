"""
utils/gauge.py - Fixed: proper sizing, all labels visible
"""
import io, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st


def render_gauge(prediction: str, confidence: float):
    if prediction == "FAKE":
        verdict_color = "#f74a4a"
        zone_label    = "HIGH RISK — LIKELY FAKE"
    elif prediction == "REAL":
        verdict_color = "#4af7a0"
        zone_label    = "LOW RISK — LIKELY REAL"
    else:
        verdict_color = "#f7c84a"
        zone_label    = "UNCERTAIN"

    # Needle: left=REAL(0%), right=FAKE(100%)
    if prediction == "FAKE":
        needle_pct = 50 + (confidence - 50)
    elif prediction == "REAL":
        needle_pct = 50 - (confidence - 50)
    else:
        needle_pct = 50.0
    needle_pct = max(3.0, min(97.0, needle_pct))
    needle_rad = math.radians(180 - (needle_pct / 100) * 180)

    # ── Figure — wider & taller so nothing gets cut ────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#12121e")
    ax.set_facecolor("#12121e")
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-0.65, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")

    theta = np.linspace(0, np.pi, 300)

    # Track background
    ax.plot(np.cos(theta), np.sin(theta),
            color="#1c1c2e", linewidth=28, solid_capstyle="round", zorder=1)

    # Coloured arc
    segs = [
        (np.linspace(np.pi,       np.pi*0.62, 90), "#4af7a0", 0.90),
        (np.linspace(np.pi*0.62,  np.pi*0.46, 40), "#a8f060", 0.78),
        (np.linspace(np.pi*0.46,  np.pi*0.35, 25), "#f7c84a", 0.85),
        (np.linspace(np.pi*0.35,  np.pi*0.18, 40), "#f79050", 0.78),
        (np.linspace(np.pi*0.18,  0,           90), "#f74a4a", 0.90),
    ]
    for seg_t, col, alpha in segs:
        ax.plot(np.cos(seg_t), np.sin(seg_t),
                color=col, linewidth=22, alpha=alpha,
                solid_capstyle="round", zorder=2)

    # Inner shine
    ax.plot(0.90 * np.cos(theta), 0.90 * np.sin(theta),
            color="white", linewidth=2, alpha=0.04,
            solid_capstyle="round", zorder=3)

    # Tick marks
    for i in range(9):
        ang      = np.pi - (i / 8) * np.pi
        is_major = (i % 4 == 0)
        r_in     = 0.78 if is_major else 0.83
        col      = "#606080" if is_major else "#2e2e48"
        lw       = 2.0 if is_major else 1.1
        ax.plot([r_in * math.cos(ang), 0.96 * math.cos(ang)],
                [r_in * math.sin(ang), 0.96 * math.sin(ang)],
                color=col, linewidth=lw, zorder=4)

    # REAL / FAKE labels — pushed out further so they don't clip
    ax.text(-1.38, 0.05, "REAL", ha="center", va="center",
            fontfamily="monospace", fontsize=11, fontweight="bold",
            color="#4af7a0", zorder=5)
    ax.text( 1.38, 0.05, "FAKE", ha="center", va="center",
            fontfamily="monospace", fontsize=11, fontweight="bold",
            color="#f74a4a", zorder=5)

    # % markers
    ax.text(-1.20, -0.28, "0%",   ha="center", fontfamily="monospace",
            fontsize=8, color="#303055")
    ax.text( 0,    1.18,  "50%",  ha="center", fontfamily="monospace",
            fontsize=8, color="#303055")
    ax.text( 1.20, -0.28, "100%", ha="center", fontfamily="monospace",
            fontsize=8, color="#303055")

    # Needle glow + needle
    tip_x = 0.84 * math.cos(needle_rad)
    tip_y = 0.84 * math.sin(needle_rad)
    ax.plot([0, tip_x], [0, tip_y], color=verdict_color,
            linewidth=10, alpha=0.12, solid_capstyle="round", zorder=5)
    ax.plot([0, tip_x], [0, tip_y], color=verdict_color,
            linewidth=2.5, solid_capstyle="round", zorder=6)
    ax.scatter([tip_x], [tip_y], color=verdict_color, s=40,
               zorder=7, linewidths=0)

    # Hub
    ax.add_patch(plt.Circle((0, 0), 0.07, color="#0b0b12", zorder=8))
    ax.add_patch(plt.Circle((0, 0), 0.07, color=verdict_color,
                             fill=False, linewidth=2.5, zorder=9))
    ax.scatter([0], [0], color=verdict_color, s=22, zorder=10, linewidths=0)

    # Centre text
    ax.text(0, -0.22, f"{int(confidence)}%",
            ha="center", va="center", fontfamily="monospace",
            fontsize=30, fontweight="bold", color=verdict_color, zorder=10)
    ax.text(0, -0.42, prediction,
            ha="center", va="center", fontfamily="monospace",
            fontsize=14, fontweight="bold", color=verdict_color, zorder=10)
    ax.text(0, -0.55, zone_label,
            ha="center", va="center", fontfamily="monospace",
            fontsize=7.5, color="#404065", zorder=10)

    # Eyebrow
    ax.text(0, 1.28, "ML MODEL VERDICT  —  MULTINOMIAL NAIVE BAYES",
            ha="center", va="center", fontfamily="monospace",
            fontsize=7, color="#32325a", zorder=10)

    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="#12121e")
    buf.seek(0)
    plt.close(fig)

    col_l, col_c, col_r = st.columns([1, 4, 1])
    with col_c:
        st.image(buf, use_container_width=True)
