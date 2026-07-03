import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

RUN = "028749"
WINDOW_TAG = "window_000_3_303s"
WINDOW_DIR = Path(f"/scratch/s5496527/results/{RUN}/window_0")

PEAK_PATH = Path(f"/scratch/s5496527/results/{RUN}/peak_dict.pkl")

summary_paths = {
    "powerlaw": WINDOW_DIR / f"{WINDOW_TAG}_powerlaw_gof_refit_summary.pkl",
    "exp_additive": WINDOW_DIR / f"{WINDOW_TAG}_exp_additive_gof_refit_summary.pkl",
    "pure_exp": WINDOW_DIR / f"{WINDOW_TAG}_pure_exp_gof_refit_summary.pkl",
    "switch": WINDOW_DIR / f"{WINDOW_TAG}_exp_plaw_gof_refit_summary.pkl",
}

for name, path in summary_paths.items():
    print(name, path.exists(), path)

with open(PEAK_PATH, "rb") as f:
    peak_dictionary = pickle.load(f)

pS2s = peak_dictionary["pS2s"]
DEs = peak_dictionary["DEs"]
S1s = peak_dictionary["S1s"]

S1_times = S1s["time_since_start"].astype(float)

pS2_run = pS2s[
    (pS2s["time_since_start"] >= 0)
    & (pS2s["time_since_start"] < 303*1e3)
].copy()


def count_DEs_after_first_pS2s_only(
    pS2s,
    DEs,
    after_ms=200.0,
    delay_start_ms=11.5,
    group_ms=None,
    window_start_ms=None,
    window_stop_ms=None,
):
    """
    Count DEs after pS2s, but only keep the first pS2 in a close local group.

    If group_ms is None, group_ms is set equal to after_ms. This means:
    once a pS2 is selected, all later pS2s within the next after_ms are skipped.

    This prevents later pS2s in a local pS2 cluster from being assigned DEs
    that may have been caused by earlier pS2s in the same cluster.
    """

    if group_ms is None:
        group_ms = after_ms

    p_times = pS2s["time_since_start"].astype(float)
    de_times = DEs["time_since_start"].astype(float)

    # Sort pS2s by time while preserving original indices
    order = np.argsort(p_times)
    p_times_sorted = p_times[order]

    rows = []
    skip_until = -np.inf

    for sorted_pos, tp in enumerate(p_times_sorted):
        original_idx = order[sorted_pos]

        if window_start_ms is not None and tp < window_start_ms:
            continue
        if window_stop_ms is not None and tp >= window_stop_ms:
            continue

        # Skip later pS2s in the same local cluster
        if tp < skip_until:
            continue

        lo = tp + delay_start_ms
        hi = tp + after_ms

        # Require the full post-pS2 window to lie inside the fit window
        if window_stop_ms is not None and hi > window_stop_ms:
            continue

        # Count pS2s in the local group for diagnostics
        group_mask = (p_times >= tp) & (p_times < tp + group_ms)
        n_pS2_group = int(np.sum(group_mask))
        if n_pS2_group > 1:
            continue

        # Count DEs after this first pS2
        de_mask = (de_times >= lo) & (de_times < hi)
        n_de = int(np.sum(de_mask))

        row = {
            "pS2_index": int(original_idx),
            "pS2_time_ms": float(tp),
            "N_DE": n_de,
            "n_pS2_in_group": n_pS2_group,
            "window_start_ms": lo,
            "window_stop_ms": hi,
            "group_stop_ms": tp + group_ms,
        }

        for field in ["area", "range_50p_area", "r", "x", "y", "subtype"]:
            if field in pS2s.dtype.names:
                row[field] = pS2s[field][original_idx]

        rows.append(row)

        # After selecting this pS2, skip later pS2s in the same local group
        skip_until = tp + group_ms

    df = pd.DataFrame(rows)

    if len(df) == 0:
        return df

    return df.sort_values("N_DE", ascending=False).reset_index(drop=True)

window_start_ms = 3_000.0
window_stop_ms = 303_000.0
pS2_counts = count_DEs_after_first_pS2s_only(
    pS2_run,
    DEs,
    after_ms=200.0,
    delay_start_ms=11.5,
    group_ms=200.0,
    window_start_ms=3_000.0,
    window_stop_ms=303_000.0,
)

pS2_counts.head(10)

def plot_observed_after_pS2(
    source_t,
    pS2s,
    DEs,
    S1_times=None,
    after_ms=200.0,
    before_ms=10.0,
    bin_width_ms=2.0,
    tmin=11.5,
    s1_dead_ms=4.6,
    title=None,
    ax=None,
    show_dead_zones=True,
    show_event_ticks=True,
):
    """
    Plot observed DEs and pS2s around one source pS2.
    No model rate is plotted.

    x-axis is time relative to source_t.
    """

    if S1_times is None:
        S1_times = np.empty(0, dtype=float)

    p_times = pS2s["time_since_start"].astype(float)
    de_times = DEs["time_since_start"].astype(float)
    s1_times = np.asarray(S1_times, dtype=float)

    abs_start = source_t - before_ms
    abs_stop = source_t + after_ms

    # Select local events
    de_local = de_times[(de_times >= abs_start) & (de_times < abs_stop)]
    pS2_local = p_times[(p_times >= abs_start) & (p_times < abs_stop)]
    s1_local = s1_times[(s1_times >= abs_start) & (s1_times < abs_stop)]

    de_rel = de_local - source_t
    pS2_rel = pS2_local - source_t
    s1_rel = s1_local - source_t

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    bins = np.arange(-before_ms, after_ms + bin_width_ms, bin_width_ms)

    counts, edges, _ = ax.hist(
        de_rel,
        bins=bins,
        histtype="step",
        linewidth=1.5,
        color="black",
        label="Observed DE count",
    )

    # Mark individual DE times as ticks near the bottom
    if show_event_ticks and len(de_rel) > 0:
        ymax = max(np.max(counts), 1)
        ax.vlines(
            de_rel,
            ymin=0,
            ymax=0.08 * ymax,
            alpha=0.5,
            linewidth=0.8,
            color="black",
            label="Observed DE times",
        )

    # Target pS2 at zero
    ax.axvline(0.0, linestyle="-", linewidth=1.5, label="Target pS2")

    # Other pS2s in window
    for x in pS2_rel:
        if np.isclose(x, 0.0, atol=1e-6):
            continue
        ax.axvline(x, linestyle="--", linewidth=1.0, alpha=0.7)

    # Dead-zone shading
    if show_dead_zones:
        # pS2 dead zones
        used_s2_label = False
        for x in pS2_rel:
            lo = max(x, -before_ms)
            hi = min(x + tmin, after_ms)
            if hi > lo:
                ax.axvspan(
                    lo,
                    hi,
                    color="green",
                    alpha=0.12,
                    label="pS2 dead zone" if not used_s2_label else None,
                )
                used_s2_label = True

        # S1 dead zones
        used_s1_label = False
        for x in s1_rel:
            lo = max(x, -before_ms)
            hi = min(x + s1_dead_ms, after_ms)
            if hi > lo:
                ax.axvspan(
                    lo,
                    hi,
                    color="purple",
                    alpha=0.10,
                    label="S1 dead zone" if not used_s1_label else None,
                )
                used_s1_label = True

    n_post = np.sum((de_times >= source_t) & (de_times < source_t + after_ms))
    n_post_live = np.sum((de_times >= source_t + tmin) & (de_times < source_t + after_ms))

    if title is None:
        title = (
            f"pS2 at {source_t:.3f} ms: "
            f"N_DE(0-{after_ms:.0f} ms)={n_post}, "
            f"N_DE({tmin:.1f}-{after_ms:.0f} ms)={n_post_live}"
        )

    ax.set_title(title)
    ax.set_xlabel("Time relative to target pS2 [ms]")
    ax.set_ylabel(f"DE count / {bin_width_ms:g} ms")
    ax.set_xlim(-before_ms, after_ms)
    ymax = max(np.max(counts), 1)
    ax.set_ylim(0, 1.1 * ymax)
    ax.legend(loc="best", fontsize="small")
    ax.ticklabel_format(useOffset=False, style="plain", axis="x")

    return fig, ax

Path("DE_plots").mkdir(exist_ok=True)

high_examples = pS2_counts.head(5).copy()

low_examples = (
    pS2_counts[pS2_counts["N_DE"] < 3]
    .sample(n=5, random_state=1)
    .copy()
)

for i, (_, row) in enumerate(high_examples.iterrows()):
    source_t = row["pS2_time_ms"]

    fig, ax = plot_observed_after_pS2(
        source_t,
        pS2_run,
        DEs,
        S1_times=S1_times,
        after_ms=200.0,
        before_ms=20.0,
        bin_width_ms=2.0,
        title=(
            f"High-DE pS2 group: "
            f"N_DE={row['N_DE']}, "
            f"N_pS2_group={row['n_pS2_in_group']}, "
            f"t={row['pS2_time_ms']:.3f} ms"
        ),
    )

    outpath = f"DE_plots/high_DE_pS2_{i:02d}_{source_t:.0f}ms.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


for i, (_, row) in enumerate(low_examples.iterrows()):
    source_t = row["pS2_time_ms"]

    fig, ax = plot_observed_after_pS2(
        source_t,
        pS2_run,
        DEs,
        S1_times=S1_times,
        after_ms=200.0,
        before_ms=20.0,
        bin_width_ms=2.0,
        title=(
            f"Low-DE pS2 group: "
            f"N_DE={row['N_DE']}, "
            f"N_pS2_group={row['n_pS2_in_group']}, "
            f"t={row['pS2_time_ms']:.3f} ms"
        ),
    )

    outpath = f"DE_plots/low_DE_pS2_{i:02d}_{source_t:.0f}ms.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

def local_live_intervals_after_source(
    source_t,
    pS2s,
    S1_times,
    after_ms=200.0,
    tmin=11.5,
    s1_dead_ms=4.6,
):
    """
    Live intervals inside [source_t, source_t + after_ms],
    using the same pS2/S1 dead-zone logic.
    """

    start = source_t
    stop = source_t + after_ms

    p_times = pS2s["time_since_start"].astype(float)
    p_local = p_times[(p_times >= start) & (p_times < stop)]

    s1_arr = np.asarray(S1_times, dtype=float)
    s1_local = s1_arr[(s1_arr >= start) & (s1_arr < stop)]

    dead = []

    for tp in p_local:
        dead.append((max(tp, start), min(tp + tmin, stop)))

    for ts in s1_local:
        dead.append((max(ts, start), min(ts + s1_dead_ms, stop)))

    dead = [(a, b) for a, b in dead if b > a]
    dead = sorted(dead)

    # merge dead intervals
    merged = []
    for a, b in dead:
        if not merged or a > merged[-1][1]:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)

    # complement gives live intervals
    live = []
    cur = start
    for a, b in merged:
        if a > cur:
            live.append((cur, a))
        cur = max(cur, b)

    if cur < stop:
        live.append((cur, stop))

    return live


def live_duration_ms(live_intervals):
    return sum(b - a for a, b in live_intervals)

def source_yield_from_area_width(area_PE, width_ns, params, A_ref, W_ref_ms):
    """
    N_p = s (A/A_ref)^c (W/W_ref)^d

    width_ns comes from the table.
    W_ref_ms is the reference width in ms.
    """

    width_ms = width_ns / 1e6

    return (
        params["s"]
        * (area_PE / A_ref) ** params["c"]
        * (width_ms / W_ref_ms) ** params["d"]
    )
#%%
def predict_single_pS2_window(
    row,
    params,
    model,
    A_ref,
    W_ref_ms,
    pS2s,
    S1_times,
    after_ms=200.0,
):
    """
    Predict source-only, k-only, and total expected DE count
    for one pS2 window.

    row must contain:
      - t_pS2_ms
      - Area_PE
      - Width_ns
      - N_DE_obs
    """

    tmin = params["tmin"]
    source_t = row["t_pS2_ms"]

    live_intervals = local_live_intervals_after_source(
        source_t,
        pS2s,
        S1_times,
        after_ms=after_ms,
        tmin=tmin,
    )

    T_live = live_duration_ms(live_intervals)

    A_over_ref = row["Area_PE"] / A_ref
    W_over_ref = (row["Width_ns"] / 1e6) / W_ref_ms

    A_alpha = A_over_ref ** params["c"]
    W_beta = W_over_ref ** params["d"]

    Np = source_yield_from_area_width(
        row["Area_PE"],
        row["Width_ns"],
        params,
        A_ref,
        W_ref_ms,
    )

    # integrate the source kernel only over local live intervals
    F_total = 0.0

    for a, b in live_intervals:
        lo = a - source_t
        hi = b - source_t

        if model == "powerlaw":
            F_total += F_plaw(lo, hi, params["n"], params["tmin"])

        elif model == "exp_additive":
            F_total += F_plaw_plus_exp(
                lo, hi,
                params["n"],
                params["tau"],
                params["f_exp"],
                params["tmin"],
            )

        elif model == "pure_exp":
            F_total += F_exp_shifted(
                lo, hi,
                params["tau"],
                params["tmin"],
            )

        elif model == "switch":
            F_total += F_plaw_to_exp_switch(
            lo, hi,
            params["n"],
            params["tau"],
            params["tmin"],
            params["t_switch"],
        )
        else:
            raise ValueError(f"Model not implemented here: {model}")

    mu_source = Np * F_total
    mu_k = params["k"] * T_live
    mu_total = mu_source + mu_k

    return {
        f"A_alpha_{model}": A_alpha,
        f"W_beta_{model}": W_beta,
        f"Np_{model}": Np,
        f"F_{model}": F_total,
        f"mu_source_{model}": mu_source,
        f"mu_k_{model}": mu_k,
        f"mu_total_{model}": mu_total,
        f"obs_minus_mu_{model}": row["N_DE_obs"] - mu_total,
        f"T_live_ms_{model}": T_live,
    }
def build_prediction_table_with_k(
    examples,
    model_params,
    A_ref,
    W_ref_ms,
    pS2s,
    S1_times,
    after_ms=200.0,
):
    """
    model_params should be like:
    {
        "powerlaw": powerlaw_params,
        "exp_additive": exp_additive_params,
        "pure_exp": pure_exp_params,
    }
    """

    out = examples.copy()

    out["A_over_ref"] = out["Area_PE"] / A_ref
    out["W_over_ref"] = (out["Width_ns"] / 1e6) / W_ref_ms

    for model, params in model_params.items():
        rows = []

        for _, row in out.iterrows():
            rows.append(
                predict_single_pS2_window(
                    row,
                    params,
                    model,
                    A_ref,
                    W_ref_ms,
                    pS2s,
                    S1_times,
                    after_ms=after_ms,
                )
            )

        pred_df = pd.DataFrame(rows)
        out = pd.concat([out.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)

    return out
def values_to_params(model, values):
    """
    Convert list-like Minuit values to a named parameter dictionary.
    """

    values = list(values)

    if model == "powerlaw":
        names = ["s", "n", "tmin", "c", "d", "k"]

    elif model == "exp_additive":
        names = ["s", "n", "tau", "f_exp", "tmin", "c", "d", "k"]

    elif model == "pure_exp":
        names = ["s", "tau", "tmin", "c", "d", "k"]

    elif model == "exp_plaw":
        names = ["s", "n", "tau", "t_switch", "tmin", "c", "d", "k"]

    elif model == "multi_exp":
        names = ["s", "n", "tmin", "c", "d", "k"]

    else:
        raise ValueError(f"Unknown model: {model}")

    return dict(zip(names, values))

def F_plaw(lo, hi, n, tmin):
    """
    Fraction of normalized power-law PDF in delay interval [lo, hi].
    h(u) = (n-1)/tmin * (u/tmin)^(-n), u > tmin.
    """
    if n <= 1.0 or hi <= tmin or hi <= lo:
        return 0.0

    lo_eff = max(lo, tmin)

    def C(u):
        if u <= tmin:
            return 0.0
        return 1.0 - (tmin / u)**(n - 1.0)

    return max(0.0, C(hi) - C(lo_eff))


def F_exp_shifted(lo, hi, tau, tmin):
    """
    Fraction of normalized shifted exponential PDF in [lo, hi].
    h(u) = (1/tau) exp[-(u-tmin)/tau], u > tmin.
    """
    if tau <= 0.0 or hi <= tmin or hi <= lo:
        return 0.0

    lo_eff = max(lo, tmin)

    def C(u):
        if u <= tmin:
            return 0.0
        return 1.0 - np.exp(-(u - tmin) / tau)

    return max(0.0, C(hi) - C(lo_eff))


def F_plaw_plus_exp(lo, hi, n, tau, f_exp, tmin):
    """
    Fraction for additive PL+Exp model.
    """
    return (
        (1.0 - f_exp) * F_plaw(lo, hi, n, tmin)
        + f_exp * F_exp_shifted(lo, hi, tau, tmin)
    )


def switch_total_integral(n, tau, tmin, t_switch):
    """
    Unnormalized integral for power-law-to-exponential switch model.

    g(u) = (u/tmin)^(-n), tmin < u < t_switch
    g(u) = (t_switch/tmin)^(-n) exp[-(u-t_switch)/tau], u >= t_switch
    """
    if n <= 1.0 or tau <= 0.0 or tmin <= 0.0 or t_switch <= tmin:
        return 0.0

    pl_int = (tmin / (n - 1.0)) * (
        1.0 - (t_switch / tmin)**(1.0 - n)
    )

    match = (t_switch / tmin)**(-n)
    exp_int = match * tau

    return pl_int + exp_int


def switch_raw_integral(lo, hi, n, tau, tmin, t_switch):
    """
    Raw unnormalized interval integral for PL-to-Exp switch model.
    """
    if n <= 1.0 or tau <= 0.0 or tmin <= 0.0 or t_switch <= tmin:
        return 0.0
    if hi <= lo or hi <= tmin:
        return 0.0

    lo = max(lo, tmin)
    total = 0.0

    # PL part: [tmin, t_switch)
    if lo < t_switch:
        pl_lo = lo
        pl_hi = min(hi, t_switch)

        if pl_hi > pl_lo:
            total += (tmin**n) * (
                pl_lo**(1.0 - n) - pl_hi**(1.0 - n)
            ) / (n - 1.0)

    # Exp part: [t_switch, infinity)
    if hi > t_switch:
        exp_lo = max(lo, t_switch)
        exp_hi = hi

        if exp_hi > exp_lo:
            match = (t_switch / tmin)**(-n)
            total += match * tau * (
                np.exp(-(exp_lo - t_switch) / tau)
                - np.exp(-(exp_hi - t_switch) / tau)
            )

    return total


def F_plaw_to_exp_switch(lo, hi, n, tau, tmin, t_switch):
    I = switch_total_integral(n, tau, tmin, t_switch)
    if I <= 0.0:
        return 0.0
    return switch_raw_integral(lo, hi, n, tau, tmin, t_switch) / I

#%%
def Np_from_area_width(area_PE, width_ns, params, A_ref, W_ref_ms):
    """
    Expected total DE yield from one pS2.

    width_ns is from your table.
    W_ref_ms should be in ms.
    """
    width_ms = width_ns / 1e6

    return (
        params["s"]
        * (area_PE / A_ref)**params["c"]
        * (width_ms / W_ref_ms)**params["d"]
    )


def mu_single_pS2_window(row, params, model, A_ref, W_ref_ms,
                         lo_ms=0.0, hi_ms=200.0):
    """
    Expected source-only DE count from a single pS2 in [lo_ms, hi_ms].
    Does not include background k or other nearby pS2s.
    """
    Np = Np_from_area_width(
        row["Area_PE"],
        row["Width_ns"],
        params,
        A_ref,
        W_ref_ms,
    )

    if model == "plaw":
        F = F_plaw(lo_ms, hi_ms, params["n"], params["tmin"])

    elif model == "plaw_exp":
        F = F_plaw_plus_exp(
            lo_ms, hi_ms,
            params["n"],
            params["tau"],
            params["f_exp"],
            params["tmin"],
        )

    elif model == "pure_exp":
        F = F_exp_shifted(
            lo_ms, hi_ms,
            params["tau"],
            params["tmin"],
        )

    elif model == "plaw_to_exp_switch":
        F = F_plaw_to_exp_switch(
            lo_ms, hi_ms,
            params["n"],
            params["tau"],
            params["tmin"],
            params["t_switch"],
        )

    else:
        raise ValueError(f"Unknown model: {model}")

    return Np, F, Np * F
#%%
def load_values_alt_real(path):
    with open(path, "rb") as f:
        summary = pickle.load(f)

    values = summary["values_alt_real"]

    if isinstance(values, pd.Series):
        if len(values) == 1 and hasattr(values.iloc[0], "__len__"):
            values = values.iloc[0]
        else:
            values = values.to_numpy()

    return list(values)

def examples_from_selected_pS2s(high_examples, low_examples):
    examples_raw = pd.concat(
        [
            high_examples.assign(sample="high"),
            low_examples.assign(sample="low"),
        ],
        ignore_index=True,
    )

    examples = examples_raw.rename(
        columns={
            "pS2_time_ms": "t_pS2_ms",
            "N_DE": "N_DE_obs",
            "area": "Area_PE",
            "range_50p_area": "Width_ns",
            "r": "r_cm",
            "x": "x_cm",
            "y": "y_cm",
        }
    ).copy()

    needed_cols = [
        "sample",
        "t_pS2_ms",
        "N_DE_obs",
        "Area_PE",
        "Width_ns",
        "r_cm",
        "x_cm",
        "y_cm",
    ]

    return examples[needed_cols]


examples = examples_from_selected_pS2s(high_examples, low_examples)


values_powerlaw = load_values_alt_real(summary_paths["powerlaw"])
values_add = load_values_alt_real(summary_paths["exp_additive"])
values_exp = load_values_alt_real(summary_paths["pure_exp"])
values_exp_plaw = load_values_alt_real(summary_paths["switch"])

powerlaw_params = values_to_params("powerlaw", values_powerlaw)
expadd_params = values_to_params("exp_additive", values_add)
pureexp_params = values_to_params("pure_exp", values_exp)
switch_params = values_to_params("exp_plaw", values_exp_plaw)

model_params = {
    "powerlaw": powerlaw_params,
    "exp_additive": expadd_params,
    "pure_exp": pureexp_params,
    "switch": switch_params,
}
A_ref = np.median(pS2_run["area"].astype(float))
W_ref_ms = np.median(pS2_run["range_50p_area"].astype(float)) / 1e6

pred_table = build_prediction_table_with_k(
    examples,
    model_params,
    A_ref=A_ref,
    W_ref_ms=W_ref_ms,
    pS2s=pS2_run,
    S1_times=S1_times,
    after_ms=200.0,
)

cols = [
    "sample", "t_pS2_ms", "N_DE_obs",
    "Area_PE", "Width_ns", "r_cm", "x_cm", "y_cm",
    "A_over_ref", "W_over_ref",

    "A_alpha_powerlaw", "W_beta_powerlaw",
    "mu_source_powerlaw", "mu_k_powerlaw", "mu_total_powerlaw",

    "A_alpha_exp_additive", "W_beta_exp_additive",
    "mu_source_exp_additive", "mu_k_exp_additive", "mu_total_exp_additive",

    "A_alpha_pure_exp", "W_beta_pure_exp",
    "mu_source_pure_exp", "mu_k_pure_exp", "mu_total_pure_exp",

    "A_alpha_switch", "W_beta_switch",
    "mu_source_switch", "mu_k_switch", "mu_total_switch",
]

pred_table_display = pred_table[cols].copy()

pred_table_display.to_csv(
    "DE_plots/high_low_pS2_prediction_table_all_models.csv",
    index=False,
)

pred_table_display[pred_table_display["sample"] == "high"].to_csv(
    "DE_plots/high_pS2_prediction_table_all_models.csv",
    index=False,
)

pred_table_display[pred_table_display["sample"] == "low"].to_csv(
    "DE_plots/low_pS2_prediction_table_all_models.csv",
    index=False,
)
