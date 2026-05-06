"""
Communication-Computation Efficient Gradient Coding
Ye & Abbe, ICML 2018
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# PALETTE
# ─────────────────────────────────────────────────────────────
# C = {
#     "bg":     "#0d1117",
#     "panel":  "#161b22",
#     "border": "#21262d",
#     "text":   "#e6edf3",
#     "muted":  "#8b949e",
#     "naive":  "#58a6ff",
#     "tandon": "#f0883e",
#     "ours":   "#3fb950",
#     "accent": "#bc8cff",
#     "red":    "#ff7b72",
#     "yellow": "#e3b341",
# }

C = {
    "bg":     "#ffffff",  # page background
    "panel":  "#f6f8fa",  # plot panels / axes background
    "border": "#d0d7de",  # borders, grid accents
    "text":   "#24292f",  # primary text
    "muted":  "#57606a",  # secondary text

    "naive":  "#0969da",  # blue
    "tandon": "#bc4c00",  # orange
    "ours":   "#1a7f37",  # green
    "accent": "#8250df",  # purple

    "red":    "#cf222e",  # warning/error
    "yellow": "#9a6700",  # caution/highlight
}

RC = {
    "figure.facecolor":  C["bg"],
    "axes.facecolor":    C["panel"],
    "axes.edgecolor":    C["border"],
    "axes.labelcolor":   C["text"],
    "axes.titlecolor":   C["text"],
    "xtick.color":       C["muted"],
    "ytick.color":       C["muted"],
    "text.color":        C["text"],
    "grid.color":        C["border"],
    "grid.linewidth":    0.6,
    "legend.facecolor":  C["panel"],
    "legend.edgecolor":  C["border"],
    "font.family":       "monospace",
    "font.size":         9,
}


# ═══════════════════════════════════════════════════════════════
# 1.  CORE CODING SCHEME
# ═══════════════════════════════════════════════════════════════

def build_theta(n):
    if n % 2 == 0:
        return np.array([(1 + i / 2) for i in range(n // 2)]
                        + [-(1 + i / 2) for i in range(n // 2)])
    else:
        return np.array([0]
                        + [(1 + i / 2) for i in range(n // 2)]
                        + [-(1 + i / 2) for i in range(n // 2)])


def build_polynomials(n, d, s, m, theta):
    if d != s + m:
        raise ValueError(f"Scheme requires d = s + m, got d={d}, s={s}, m={m}.")
    if len(theta) != n:
        raise ValueError("len(theta) must equal n.")

    poly_len = n - s

    P = []
    for i in range(n):
        roots = [theta[(i + j) % n] for j in range(1, n - d + 1)]
        desc = np.poly(roots)
        asc = np.zeros(poly_len)
        for j, c in enumerate(reversed(desc)):
            if j < poly_len:
                asc[j] = c

        P_i = [asc]

        for u in range(2, m + 1):
            prev = P_i[-1]
            xp = np.zeros(poly_len)
            xp[1:] = prev[:-1]
            idx = n - d - 1
            c_sub = prev[idx] if 0 <= idx < poly_len else 0.0
            P_i.append(xp - c_sub * P_i[0])

        P.append(P_i)

    return P


def build_B(n, s, m, P):
    B = np.zeros((m * n, n - s))
    for i in range(n):
        for u in range(m):
            B[i * m + u, :] = P[i][u]
    return B


def encode(n, d, s, m, l, theta, B, G):
    assert l % m == 0, "l must be divisible by m."
    q = l // m
    F = np.zeros((n, q))

    for i in range(n):
        A_i = np.array([theta[i] ** p for p in range(n - s)])
        for b in range(q):
            block = slice(b * m, (b + 1) * m)
            z_v = G[:, block].ravel()
            F[i, b] = z_v @ (B @ A_i)

    return F


def decode(n, d, s, m, l, theta, F, straggler_set):
    assert l % m == 0
    q = l // m
    stragglers = set(straggler_set)
    survivors  = [i for i in range(n) if i not in stragglers]
    assert len(survivors) >= n - s, "More stragglers than the scheme can handle."
    survivors = survivors[:n - s]

    A_surv = np.array([[theta[i] ** j for j in range(n - s)] for i in survivors])

    try:
        A_inv = np.linalg.inv(A_surv)
    except np.linalg.LinAlgError:
        raise RuntimeError("Vandermonde matrix is singular.")

    recovered = np.zeros(l)
    for b in range(q):
        F_b = F[survivors, b]
        Cv  = A_inv @ F_b
        recovered[b * m : b * m + m] = Cv[n - d : n - d + m]

    return recovered


def run_demo(n, d, s, m, l, seed=0):
    theta = build_theta(n)
    P     = build_polynomials(n, d, s, m, theta)
    B     = build_B(n, s, m, P)

    rng = np.random.default_rng(seed)
    G   = rng.random((n, l))
    true_sum = G.sum(axis=0)

    F = encode(n, d, s, m, l, theta, B, G)

    stragglers = rng.choice(n, size=s, replace=False).tolist()
    recovered  = decode(n, d, s, m, l, theta, F, stragglers)

    err = np.linalg.norm(true_sum - recovered)
    return true_sum, recovered, err


# ═══════════════════════════════════════════════════════════════
# 2.  TIMING MODEL
# ═══════════════════════════════════════════════════════════════

def sample_shifted_exp(lam, shift, shape):
    return shift + np.random.exponential(1.0 / lam, size=shape)


def expected_runtime(n, d, m, lam1=0.8, lam2=0.1, t1=1.6, t2=6.0, n_trials=20_000):
    s = d - m
    if s < 0 or n - s <= 0 or n - s > n:
        return np.inf

    T_comp = sample_shifted_exp(lam1, t1, (n_trials, n))
    T_comm = sample_shifted_exp(lam2, t2, (n_trials, n))

    T_worker = d * T_comp + T_comm / m
    T_sorted  = np.sort(T_worker, axis=1)
    T_wait    = T_sorted[:, n - s - 1]

    return float(np.mean(T_wait))


def naive_runtime(n, lam1=0.8, lam2=0.1, t1=1.6, t2=6.0, n_trials=20_000):
    T_comp = sample_shifted_exp(lam1, t1, (n_trials, n))
    T_comm = sample_shifted_exp(lam2, t2, (n_trials, n))
    return float(np.mean(np.max(T_comp + T_comm, axis=1)))


def tandon_best_runtime(n, lam1=0.8, lam2=0.1, t1=1.6, t2=6.0, n_trials=20_000):
    best = np.inf
    for d in range(1, n + 1):
        rt = expected_runtime(n, d, 1, lam1, lam2, t1, t2, n_trials)
        if rt < best:
            best = rt
    return best


# ═══════════════════════════════════════════════════════════════
# 3.  COMPUTE ALL DATA  (once, reused by individual + combined)
# ═══════════════════════════════════════════════════════════════

def compute_all_data(lam1=0.8, lam2=0.1, t1=1.6, t2=6.0, N_TRIALS=15_000):
    data = {}

    # ── Plot 1 ──────────────────────────────────────────────────
    print("Computing runtime simulations (Plot 1) …")
    n_vals = [6, 8, 10, 12, 15]
    naive_rts  = [naive_runtime(n, lam1, lam2, t1, t2, N_TRIALS) for n in n_vals]
    tandon_rts = [tandon_best_runtime(n, lam1, lam2, t1, t2, N_TRIALS) for n in n_vals]

    def our_best(n):
        best = np.inf
        for d in range(2, n + 1):
            for m in range(2, d + 1):
                rt = expected_runtime(n, d, m, lam1, lam2, t1, t2, N_TRIALS)
                if rt < best:
                    best = rt
        return best

    our_rts = [our_best(n) for n in n_vals]
    data["p1"] = dict(n_vals=n_vals, naive_rts=naive_rts,
                      tandon_rts=tandon_rts, our_rts=our_rts)

    # ── Plot 2 ──────────────────────────────────────────────────
    print("Computing heatmap (Plot 2, n=8) …")
    n_h = 8
    d_vals = range(1, n_h + 1)
    m_vals_h = range(1, n_h + 1)
    Z = np.full((n_h, n_h), np.nan)
    for di, d in enumerate(d_vals):
        for mi, m in enumerate(m_vals_h):
            if m <= d:
                Z[mi, di] = expected_runtime(n_h, d, m, lam1, lam2, t1, t2, n_trials=8_000)
    data["p2"] = dict(n_h=n_h, Z=Z)

    # ── Plot 3 ──────────────────────────────────────────────────
    n_tr = 12
    print(f"Tradeoff curves (Plot 3, n={n_tr}) …")
    d_choices = [3, 4, 5, 6]
    tradeoff_curves = {}
    for d_fixed in d_choices:
        s_range = list(range(0, d_fixed + 1))
        rts = []
        for s_val in s_range:
            m_val = d_fixed - s_val
            if m_val < 1 or m_val > d_fixed:
                rts.append(np.nan)
                continue
            rt = expected_runtime(n_tr, d_fixed, m_val, lam1, lam2, t1, t2, n_trials=8_000)
            rts.append(rt)
        tradeoff_curves[d_fixed] = (s_range, rts)
    naive_n12 = (naive_rts[n_vals.index(n_tr)] if n_tr in n_vals
                 else naive_runtime(n_tr, lam1, lam2, t1, t2, 5_000))
    data["p3"] = dict(n_tr=n_tr, d_choices=d_choices,
                      tradeoff_curves=tradeoff_curves, naive_n12=naive_n12)

    # ── Plot 4 ──────────────────────────────────────────────────
    print("Gradient deviation vs n (Plot 4) …")
    n_range = list(range(4, 22, 2))
    errors_n = []
    for n_v in n_range:
        d_v, s_v, m_v, l_v = n_v // 2, n_v // 4, n_v // 4, 16
        s_v = max(1, s_v); m_v = max(1, m_v); d_v = s_v + m_v
        if d_v > n_v:
            errors_n.append(np.nan); continue
        try:
            errs = [run_demo(n_v, d_v, s_v, m_v, l_v, seed=seed)[2] for seed in range(20)]
            errors_n.append(np.median(errs))
        except Exception:
            errors_n.append(np.nan)
    data["p4"] = dict(n_range=n_range, errors_n=errors_n)

    # ── Plot 5 ──────────────────────────────────────────────────
    print("Gradient deviation vs s (Plot 5) …")
    n_f, l_f = 10, 20
    m_choices_s = [1, 2, 4, 5]
    curves_s = {}
    for m_v in m_choices_s:
        s_range_s = list(range(0, n_f - m_v + 1))
        errs_s = []
        for s_v in s_range_s:
            d_v = s_v + m_v
            if d_v > n_f:
                errs_s.append(np.nan); continue
            try:
                errs_trial = [run_demo(n_f, d_v, s_v, m_v, l_f, seed=seed)[2]
                              for seed in range(15)]
                errs_s.append(np.median(errs_trial))
            except Exception:
                errs_s.append(np.nan)
        curves_s[m_v] = (s_range_s, errs_s)
    data["p5"] = dict(n_f=n_f, l_f=l_f, m_choices_s=m_choices_s, curves_s=curves_s)

    # ── Plot 6 ──────────────────────────────────────────────────
    print("Gradient deviation vs m (Plot 6) …")
    n_f2, s_fixed, l_f2 = 10, 2, 20
    m_range = list(range(1, n_f2 - s_fixed + 1))
    errs_m = []
    for m_v in m_range:
        d_v = s_fixed + m_v
        if d_v > n_f2:
            errs_m.append(np.nan); continue
        try:
            errs_trial = [run_demo(n_f2, d_v, s_fixed, m_v, l_f2, seed=seed)[2]
                          for seed in range(15)]
            errs_m.append(np.median(errs_trial))
        except Exception:
            errs_m.append(np.nan)
    data["p6"] = dict(n_f2=n_f2, s_fixed=s_fixed, l_f2=l_f2,
                      m_range=m_range, errs_m=errs_m)

    # ── Plot 7 ──────────────────────────────────────────────────
    print("Condition number of Vandermonde vs n (Plot 7) …")
    n_cond = list(range(3, 22))
    conds  = []
    for n_v in n_cond:
        theta = build_theta(n_v)
        V = np.array([[theta[i] ** j for j in range(n_v)] for i in range(n_v)])
        conds.append(np.linalg.cond(V))
    data["p7"] = dict(n_cond=n_cond, conds=conds)

    return data


# ═══════════════════════════════════════════════════════════════
# 4.  INDIVIDUAL DRAW FUNCTIONS  (each takes an Axes)
# ═══════════════════════════════════════════════════════════════

def draw_p1(ax, d):
    """Expected Runtime per Iteration"""
    p = d["p1"]
    n_vals, naive_rts, tandon_rts, our_rts = (
        p["n_vals"], p["naive_rts"], p["tandon_rts"], p["our_rts"])

    x = np.arange(len(n_vals))
    w = 0.26
    ax.bar(x - w, naive_rts,  width=w, label="Naive (d=1, m=1)",
           color=C["naive"],  alpha=0.85, linewidth=0)
    ax.bar(x,     tandon_rts, width=w, label="Tandon et al. (m=1, best d)",
           color=C["tandon"], alpha=0.85, linewidth=0)
    ax.bar(x + w, our_rts,   width=w, label="This paper (m≥2, best d,m)",
           color=C["ours"],   alpha=0.85, linewidth=0)

    for xi, (_, tr, or_) in enumerate(zip(naive_rts, tandon_rts, our_rts)):
        pct = 100 * (tr - or_) / tr
        ax.text(xi + w, or_ + 0.15, f"−{pct:.0f}%\nvs Tandon",
                ha="center", va="bottom", fontsize=7, color=C["accent"])

    ax.set_xticks(x)
    ax.set_xticklabels([f"n={v}" for v in n_vals])
    ax.set_ylabel("E[Runtime] (seconds)")
    ax.set_title("Expected Runtime per Iteration\n"
                 "(Shifted-Exponential Model, λ₁=0.8 λ₂=0.1 t₁=1.6 t₂=6)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.4)
    ax.set_axisbelow(True)


def draw_p2(ax, d):
    """Runtime Heatmap"""
    p = d["p2"]
    n_h, Z = p["n_h"], p["Z"]

    cmap = LinearSegmentedColormap.from_list(
        "gcmap", [C["ours"], C["yellow"], C["red"]], N=256)
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap=cmap,
                   extent=[0.5, n_h + 0.5, 0.5, n_h + 0.5])
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("E[T_tot] (s)", color=C["muted"], fontsize=8)
    cb.ax.yaxis.set_tick_params(color=C["muted"])
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=C["muted"])

    valid_mask = ~np.isnan(Z)
    if valid_mask.any():
        min_idx = np.unravel_index(np.nanargmin(Z), Z.shape)
        ax.plot(min_idx[1] + 1, min_idx[0] + 1, "w*", ms=12, label="Optimal")
        ax.legend(fontsize=7, loc="upper right")

    ax.set_xlabel("d  (computation load)")
    ax.set_ylabel("m  (comm. reduction)")
    ax.set_title(f"Runtime Heatmap\n(n=k={n_h})")
    ax.xaxis.label.set_color(C["muted"])
    ax.yaxis.label.set_color(C["muted"])


def draw_p3(ax, d):
    """Comm–Straggler Tradeoff"""
    p = d["p3"]
    n_tr, d_choices = p["n_tr"], p["d_choices"]
    tradeoff_curves, naive_n12 = p["tradeoff_curves"], p["naive_n12"]

    colors_tr = [C["naive"], C["tandon"], C["ours"], C["accent"], C["red"]]
    for cidx, d_fixed in enumerate(d_choices):
        s_range, rts = tradeoff_curves[d_fixed]
        color = colors_tr[cidx % len(colors_tr)]
        ax.plot(s_range, rts, "o-", color=color, linewidth=1.8, markersize=5,
                label=f"d={d_fixed} (load ratio {d_fixed/n_tr:.2f})")

    ax.axhline(naive_n12, color=C["naive"], linestyle="--", linewidth=1.2,
               label="Naive baseline", alpha=0.7)
    ax.set_xlabel("s  (straggler tolerance)")
    ax.set_ylabel("E[Runtime] (seconds)")
    ax.set_title(
        f"Comm–Straggler Tradeoff (n={n_tr})\n"
        "Each curve = fixed d; move right → tolerate more stragglers → less comm. reduction"
    )
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.4)
    ax.set_axisbelow(True)


def draw_p4(ax, d):
    """Gradient Error vs n"""
    p = d["p4"]
    n_range, errors_n = p["n_range"], p["errors_n"]

    ax.semilogy(n_range, errors_n, "o-", color=C["accent"], linewidth=1.8, markersize=5)
    ax.set_xlabel("n (number of workers)")
    ax.set_ylabel("‖recovered − true‖₂  (log scale)")
    ax.set_title("Gradient Error vs n\n(Vandermonde stability)")
    ax.grid(alpha=0.4)
    ax.set_axisbelow(True)

    safe_n = [nv for nv, e in zip(n_range, errors_n)
              if e is not None and not np.isnan(e) and e < 1e-6]
    if safe_n:
        ax.axvspan(min(n_range), max(safe_n), alpha=0.08,
                   color=C["ours"], label="Error < 1e-6")
        ax.legend(fontsize=7)


def draw_p5(ax, d):
    """Gradient Error vs s"""
    p = d["p5"]
    n_f, l_f = p["n_f"], p["l_f"]
    m_choices_s, curves_s = p["m_choices_s"], p["curves_s"]

    palette = [C["naive"], C["tandon"], C["ours"], C["accent"]]
    for idx, m_v in enumerate(m_choices_s):
        s_range_s, errs_s = curves_s[m_v]
        ax.semilogy(s_range_s, errs_s, "o-", color=palette[idx],
                    linewidth=1.8, markersize=5, label=f"m={m_v}")

    ax.set_xlabel("s  (stragglers tolerated)")
    ax.set_ylabel("‖recovered − true‖₂")
    ax.set_title(f"Gradient Error vs s\n(n={n_f}, fixed d=s+m, l={l_f})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.4)
    ax.set_axisbelow(True)


def draw_p6(ax, d):
    """Gradient Error vs m"""
    p = d["p6"]
    n_f2, s_fixed, l_f2 = p["n_f2"], p["s_fixed"], p["l_f2"]
    m_range, errs_m = p["m_range"], p["errs_m"]

    ax.semilogy(m_range, errs_m, "s-", color=C["yellow"], linewidth=1.8, markersize=5)
    ax.set_xlabel("m  (communication reduction factor)")
    ax.set_ylabel("‖recovered − true‖₂")
    ax.set_title(f"Gradient Error vs m\n(n={n_f2}, s={s_fixed} fixed, l={l_f2})")
    ax.grid(alpha=0.4)
    ax.set_axisbelow(True)


def draw_p7(ax, d):
    """Vandermonde Condition Number vs n"""
    p = d["p7"]
    n_cond, conds = p["n_cond"], p["conds"]

    ax.semilogy(n_cond, conds, "^-", color=C["red"], linewidth=1.8, markersize=5)
    ax.axhline(1 / np.finfo(float).eps, color=C["muted"], linestyle=":",
               linewidth=1.2, label="Numerical precision limit")
    ax.set_xlabel("n  (number of workers / evaluation points)")
    ax.set_ylabel("κ(V)  (condition number, log scale)")
    ax.set_title("Vandermonde Condition Number vs n\n(governs numerical stability)")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.4)
    ax.set_axisbelow(True)


# ═══════════════════════════════════════════════════════════════
# 5.  SAVE INDIVIDUAL PLOTS
# ═══════════════════════════════════════════════════════════════

INDIVIDUAL_SPECS = [
    # (draw_fn,  filename,               figsize,      title_suffix)
    (draw_p1, "plot1_expected_runtime.png",        (9, 5),  None),
    (draw_p2, "plot2_runtime_heatmap.png",         (6, 5),  None),
    (draw_p3, "plot3_comm_straggler_tradeoff.png", (9, 5),  None),
    (draw_p4, "plot4_gradient_error_vs_n.png",     (6, 5),  None),
    (draw_p5, "plot5_gradient_error_vs_s.png",     (6, 5),  None),
    (draw_p6, "plot6_gradient_error_vs_m.png",     (6, 5),  None),
    (draw_p7, "plot7_vandermonde_condition.png",   (8, 5),  None),
]


def save_individual(data, out_dir=""):
    plt.rcParams.update(RC)
    saved = []
    for draw_fn, fname, figsize, _ in INDIVIDUAL_SPECS:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(C["bg"])
        draw_fn(ax, data)
        path = out_dir + fname
        fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=C["bg"])
        plt.close(fig)
        print(f"  Saved → {path}")
        saved.append(path)
    return saved


# ═══════════════════════════════════════════════════════════════
# 6.  COMBINED FIGURE  (original layout)
# ═══════════════════════════════════════════════════════════════

def make_combined(data):
    plt.rcParams.update(RC)

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor(C["bg"])
    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.52, wspace=0.38,
                           left=0.06, right=0.97,
                           top=0.93, bottom=0.06)

    ax_runtime  = fig.add_subplot(gs[0, :2])
    ax_heatmap  = fig.add_subplot(gs[0, 2])
    ax_tradeoff = fig.add_subplot(gs[1, :2])
    ax_devn     = fig.add_subplot(gs[1, 2])
    ax_devm     = fig.add_subplot(gs[2, 0])
    ax_devsm    = fig.add_subplot(gs[2, 1])
    ax_cond     = fig.add_subplot(gs[2, 2])

    fig.suptitle(
        "Gradient Coding: Communication-Computation Tradeoff  "
        "[Ye & Abbe, ICML 2018]",
        fontsize=13, fontweight="bold", color=C["text"], y=0.975,
    )

    draw_p1(ax_runtime,  data)
    draw_p2(ax_heatmap,  data)
    draw_p3(ax_tradeoff, data)
    draw_p4(ax_devn,     data)
    draw_p5(ax_devm,     data)
    draw_p6(ax_devsm,    data)
    draw_p7(ax_cond,     data)

    return fig


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick correctness check
    print("─" * 60)
    print("CORRECTNESS CHECKS")
    print("─" * 60)
    test_cases = [
        (5,  3,  1, 2,  4),
        (6,  3,  1, 2,  6),
        (8,  4,  2, 2,  8),
        (10, 5,  3, 2, 20),
        (10, 4,  2, 2, 20),
        (10, 3,  1, 2, 20),
        (12, 6,  3, 3, 12),
        (15, 5,  2, 3, 15),
    ]
    for args in test_cases:
        n, d, s, m, l = args
        true_sum, recovered, err = run_demo(n, d, s, m, l)
        status = "✓" if err < 1e-9 else f"✗  err={err:.2e}"
        print(f"  n={n:2d} d={d} s={s} m={m} l={l:3d}  |  {status}")

    print("\n" + "─" * 60)
    print("COMPUTING DATA  (shared across all plots)")
    print("─" * 60)
    data = compute_all_data()

    OUT = ""

    print("\n" + "─" * 60)
    print("SAVING INDIVIDUAL PLOTS")
    print("─" * 60)
    save_individual(data, out_dir=OUT)

    print("\n" + "─" * 60)
    print("SAVING COMBINED FIGURE")
    print("─" * 60)
    fig_combined = make_combined(data)
    combined_path = OUT + "gradient_coding_analysis_combined.png"
    fig_combined.savefig(combined_path, dpi=140, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig_combined)
    print(f"  Saved → {combined_path}")

    print("\nAll done! ✓")