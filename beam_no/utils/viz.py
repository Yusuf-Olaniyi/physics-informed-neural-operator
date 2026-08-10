import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def plot_beam_problem_schematic(support_label: str = "Simply supported", save_path: str = None):
    """Sketch of the static beam problem: beam line, a representative q(x)
    load arrows above it, the resulting deflected shape w(x) below/over it,
    and support symbols consistent with `support_label`.
    """
    fig, ax = plt.subplots(figsize=(9, 3.2))

    L = 10.0
    x = np.linspace(0, L, 200)

    # undeformed beam (dashed) and deformed shape (solid)
    ax.plot([0, L], [0, 0], color="0.5", linestyle="--", linewidth=1.5, label="Undeformed beam")
    w = -0.6 * np.sin(np.pi * x / L)  # illustrative deflected shape only
    ax.plot(x, w, color="tab:blue", linewidth=2.5, label="Deflected shape w(x)")

    # distributed load arrows q(x)
    q_amp = 0.9
    for xi in np.linspace(0.5, L - 0.5, 14):
        ax.annotate(
            "", xy=(xi, 0.05), xytext=(xi, q_amp),
            arrowprops=dict(arrowstyle="->", color="tab:orange", linewidth=1.2),
        )
    ax.plot([0, L], [q_amp, q_amp], color="tab:orange", linewidth=1.5)
    ax.text(L / 2, q_amp + 0.15, "q(x)", color="tab:orange", ha="center", fontsize=12)

    # supports
    label = support_label.upper().replace(" ", "_")
    if "CANTILEVER" in label:
        ax.add_patch(patches.Rectangle((-0.3, -1.0), 0.3, 2.0, color="0.3"))
    elif "SIMPLY" in label:
        _triangle_support(ax, 0)
        _triangle_support(ax, L)
    elif "FIXED_FIXED" in label:
        ax.add_patch(patches.Rectangle((-0.3, -1.0), 0.3, 2.0, color="0.3"))
        ax.add_patch(patches.Rectangle((L, -1.0), 0.3, 2.0, color="0.3"))
    elif "FIXED_PINNED" in label:
        ax.add_patch(patches.Rectangle((-0.3, -1.0), 0.3, 2.0, color="0.3"))
        _triangle_support(ax, L)

    ax.set_xlim(-1, L + 1)
    ax.set_ylim(-1.2, 1.6)
    ax.set_xlabel("x")
    ax.set_title(f"Static Euler-Bernoulli beam problem ({support_label})")
    ax.axis("off")
    ax.legend(loc="lower right", frameon=False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


def _triangle_support(ax, x0):
    tri = patches.Polygon(
        [[x0 - 0.35, -0.9], [x0 + 0.35, -0.9], [x0, -0.1]],
        closed=True, color="0.3",
    )
    ax.add_patch(tri)


def plot_fno_architecture(save_path: str = None):
    """Block diagram: input -> lifting (P) -> Fourier layers -> projection (Q) -> output."""
    fig, ax = plt.subplots(figsize=(11, 2.6))

    blocks = [
        ("Input\n[signal(x), x]", "0.85"),
        ("Lifting\nLinear P", "tab:blue"),
        ("Fourier\nLayer 1", "tab:orange"),
        ("Fourier\nLayer 2", "tab:orange"),
        ("...", "0.95"),
        ("Fourier\nLayer L", "tab:orange"),
        ("Projection\nLinear Q", "tab:blue"),
        ("Output\nfield", "0.85"),
    ]

    n = len(blocks)
    box_w, box_h, gap = 1.3, 1.0, 0.35
    total_w = n * box_w + (n - 1) * gap
    x0 = -total_w / 2

    for i, (label, color) in enumerate(blocks):
        x = x0 + i * (box_w + gap)
        face = "none" if color == "0.95" and label == "..." else color
        if label == "...":
            ax.text(x + box_w / 2, box_h / 2, "...", ha="center", va="center", fontsize=18)
        else:
            rect = patches.FancyBboxPatch(
                (x, 0), box_w, box_h, boxstyle="round,pad=0.02,rounding_size=0.05",
                linewidth=1.2, edgecolor="black",
                facecolor=color if isinstance(color, str) and color.startswith("tab") else color,
                alpha=0.85 if isinstance(color, str) and color.startswith("tab") else 1.0,
            )
            ax.add_patch(rect)
            text_color = "white" if isinstance(color, str) and color.startswith("tab") else "black"
            ax.text(x + box_w / 2, box_h / 2, label, ha="center", va="center",
                     fontsize=9, color=text_color)

        if i < n - 1:
            ax.annotate("", xy=(x + box_w + gap, box_h / 2), xytext=(x + box_w, box_h / 2),
                        arrowprops=dict(arrowstyle="->", linewidth=1.3))

    # Fourier-layer internal detail annotation
    ax.text(0, -0.55,
            r"Each Fourier layer: $v_{k+1}(x)=\sigma\left(Wv_k(x)+\mathcal{F}^{-1}(R_k\cdot\mathcal{F}(v_k))\right)$",
            ha="center", fontsize=10)

    ax.set_xlim(x0 - 0.5, x0 + total_w + 0.5)
    ax.set_ylim(-1.0, 1.3)
    ax.axis("off")
    ax.set_title("Fourier Neural Operator architecture")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig
