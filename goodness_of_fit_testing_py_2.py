# %%
import os
import time
import pickle
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import strax
import straxen
from register_conor_plugins import register_conor_plugins
from conor_plugins import data_selection
from conor_plugins import model as mod
from scipy.stats import goodness_of_fit
def values_to_list(values):
    """
    Convert Minuit ValueView / dict / list / numpy array to a plain ordered list.
    """
    if isinstance(values, dict):
        return list(values.values())

    return list(values)
# %%
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_gof_D_column(df):
    """
    Multi-bin recompute may use D_recomputed.
    Original GOF-refit output uses D.
    """
    if "D_recomputed" in df.columns:
        return "D_recomputed"
    if "D" in df.columns:
        return "D"
    raise KeyError("Could not find a D column. Expected 'D_recomputed' or 'D'.")


def ensure_plot_dir(savedir, subdir="plots"):
    plot_dir = Path(savedir) / subdir
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir

def make_fake_se_roi_like(se_template, fake_times_ms, run_start_ns=None):
    """
    Create a fake structured SE array with the same dtype as se_template,
    but with simulated time_since_start values.

    Only fields relevant for fitting need to be correct.
    """
    fake = np.zeros(len(fake_times_ms), dtype=se_template.dtype)

    fake["time_since_start"] = fake_times_ms

    if "time" in fake.dtype.names:
        if run_start_ns is None:
            fake["time"] = (fake_times_ms * 1e6).astype(fake["time"].dtype)
        else:
            fake["time"] = (run_start_ns + fake_times_ms * 1e6).astype(fake["time"].dtype)

    return fake
# %%
def get_live_intervals_for_fit(
    window_start_ms,
    window_stop_ms,
    pS2s_for_deadtime,
    s1_times,
    tmin=11.5,
):
    if hasattr(s1_times, "dtype") and s1_times.dtype.names is not None:
        s1_arr = s1_times["time_since_start"].astype(float)
    else:
        s1_arr = np.asarray(s1_times, dtype=float)

    dead_intervals = mod.build_dead_intervals(
        window_start_ms,
        window_stop_ms,
        np.sort(pS2s_for_deadtime["time_since_start"].astype(float)),
        np.sort(s1_arr),
        tmin,
    )

    live_intervals = mod.build_live_intervals(
        window_start_ms,
        window_stop_ms,
        dead_intervals,
    )

    return live_intervals
# %%
def mask_rate_to_live_intervals(t_grid, rate, live_intervals):
    mask = np.zeros_like(t_grid, dtype=bool)

    for a, b in live_intervals:
        mask |= (t_grid >= a) & (t_grid < b)

    rate_live = np.array(rate, dtype=float, copy=True)
    rate_live[~mask] = 0.0

    return rate_live
# %%
def simulate_from_inhomogeneous_rate(
    t_grid,
    rate,
    rng=None,
):
    """
    Simulate event times from an inhomogeneous Poisson process
    using a grid approximation to lambda(t).

    Parameters
    ----------
    t_grid : array, ms
    rate : array, expected events per ms
    rng : np.random.Generator

    Returns
    -------
    fake_times : sorted array of simulated event times in ms
    mu_grid : expected count from numerical integration
    """
    if rng is None:
        rng = np.random.default_rng()

    t_grid = np.asarray(t_grid, dtype=float)
    rate = np.asarray(rate, dtype=float)

    rate = np.maximum(rate, 0.0)

    dt = np.diff(t_grid)
    mid_rate = 0.5 * (rate[:-1] + rate[1:])
    weights = mid_rate * dt

    mu_grid = np.sum(weights)

    if mu_grid <= 0:
        return np.empty(0, dtype=float), 0.0

    N_fake = rng.poisson(mu_grid)

    if N_fake == 0:
        return np.empty(0, dtype=float), mu_grid

    cdf = np.cumsum(weights)
    cdf = cdf / cdf[-1]

    u = rng.random(N_fake)

    # sample within intervals using right-edge interpolation
    fake_times = np.interp(u, cdf, t_grid[1:])

    return np.sort(fake_times), mu_grid
# %%
def evaluate_powerlaw_rate(
    t_grid,
    values_plaw,
    pS2s,
    s1_times,
    window_start_ms,
    window_stop_ms,
    history_ms
):
    """
    Replace `multi_powerlaw_wrap` with your actual original power-law wrapper.
    Expected output:
        total_rate, rate
    """
    total_rate, rate = mod.multi_powerlaw_wrap(
        t_grid,
        values_plaw,
        pS2s,
        s1_times,
        window_start_ms,
        window_stop_ms,
        history_ms = history_ms
    )

    return total_rate, rate
# %%
def evaluate_additive_rate(
    t_grid,
    values_add,
    pS2s,
    s1_times,
    window_start_ms,
    window_stop_ms,
    history_ms
):
    """
    Replace `multi_powerlaw_wrap` with your actual original power-law wrapper.
    Expected output:
        total_rate, rate
    """
    total_rate, rate = mod.multi_exp_additive_wrap(
        t_grid,
        values_add,
        pS2s,
        s1_times,
        window_start_ms,
        window_stop_ms,
        history_ms = history_ms
    )

    return total_rate, rate
# %%
def fit_model_by_name(
    model_name,
    run_id,
    pS2s,
    fake_or_real_se,
    s1_times,
    seconds_range,
    history_ms
):
    """
    Fit one of the supported models and return:
        values, errors, covariance, BIC, fval
    """

    if model_name == "powerlaw":
        return mod.cost_func(
            run_id,
            pS2s,
            fake_or_real_se,
            s1_times,
            seconds_range=seconds_range,
            model="new",
            history_ms = history_ms
        )

    elif model_name == "exp_additive":
        return mod.cost_func_exp_additive(
            run_id,
            pS2s,
            fake_or_real_se,
            s1_times,
            seconds_range=seconds_range,
            model="exp_additive",
            history_ms = history_ms
        )
    elif model_name == "pure_exp":
        return mod.cost_func_pure_exp(
            run_id,
            pS2s,
            fake_or_real_se,
            s1_times,
            seconds_range=seconds_range,
            model="pure_exp",
            history_ms=history_ms,
        )

    elif model_name == "exp_plaw":
        return mod.cost_func_exp_powerlaw(
            run_id,
            pS2s,
            fake_or_real_se,
            s1_times,
            seconds_range=seconds_range,
            model="exp",
            history_ms=history_ms,
        )
    elif model_name == "multi_exp":
        return mod.cost_func_multi_exp(
            run_id,
            pS2s,
            fake_or_real_se,
            s1_times,
            seconds_range=seconds_range,
            model="multi_exp",
            history_ms=history_ms,
        )

    else:
        raise ValueError(f"Unknown model_name: {model_name}")
# %%
def evaluate_model_rate_by_name(
    model_name,
    t_grid,
    values,
    pS2s,
    s1_times,
    window_start_ms,
    window_stop_ms,
    history_ms
):
    """
    Evaluate fitted model intensity on t_grid.

    Returns:
        total_rate, rate
    """

    if model_name == "powerlaw":
        return evaluate_powerlaw_rate(
            t_grid,
            values,
            pS2s,
            s1_times,
            window_start_ms,
            window_stop_ms,
            history_ms
        )

    elif model_name == "exp_additive":
        return evaluate_additive_rate(
            t_grid,
            values,
            pS2s,
            s1_times,
            window_start_ms,
            window_stop_ms,
            history_ms
        )
    elif model_name == "pure_exp":
        return mod.multi_pure_exp_wrap(
            t_grid,
            values,
            pS2s,
            s1_times,
            window_start_ms,
            window_stop_ms,
            history_ms=history_ms,
        )

    elif model_name == "exp_plaw":
        return mod.multi_exp_powerlaw_wrap(
            t_grid,
            values,
            pS2s,
            s1_times,
            window_start_ms,
            window_stop_ms,
            history_ms=history_ms,
        )
    elif model_name == "multi_exp":
        return mod.multi_exp_wrap(
            t_grid,
            values,
            pS2s,
            s1_times,
            window_start_ms,
            window_stop_ms,
            history_ms=history_ms,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
# %%
def run_one_lrt_bootstrap(i, seed, fit_inputs):
    """
    One independent bootstrap iteration for PL vs exp_add.

    Returns a small dict. Do not return huge arrays unless necessary.
    """
    import numpy as np
    # Important: import your model module inside the worker
    # so each process has access to it.
    from conor_plugins import model as mod

    rng = np.random.default_rng(seed)

    try:
        run_id = fit_inputs["run_id"]
        s2_region = fit_inputs["s2_region"]
        S1_region = fit_inputs["S1_region"]
        seconds_range = fit_inputs["seconds_range"]
        history_ms = fit_inputs["history_ms"]
        dt_ms = fit_inputs["dt_ms"]
        null_model = fit_inputs["null_model"]
        alt_model = fit_inputs["alt_model"]
        DEs_template = fit_inputs["DEs_template"]
        # Best-fit null model values, e.g. power-law values
        values_null = fit_inputs["values_null"]
        

        # ------------------------------------------------------------
        # 1. Generate fake DEs under the null model
        # ------------------------------------------------------------
        fit_start_ms = seconds_range[0]*1e3
        fit_stop_ms = seconds_range[1]*1e3
        tmin = 11.5
        live_intervals = get_live_intervals_for_fit(
            fit_start_ms,
            fit_stop_ms,
            s2_region,
            S1_region,
            tmin=tmin
        )

        t_grid = np.arange(fit_start_ms, fit_stop_ms, dt_ms)
        mu_model, rate_null = evaluate_model_rate_by_name(
            null_model,
            t_grid,
            values_null,
            s2_region,
            S1_region,
            fit_start_ms,
            fit_stop_ms,
            history_ms=history_ms,
        )

        rate_null = mask_rate_to_live_intervals(
            t_grid,
            rate_null,
            live_intervals,
        )

        fake_times, mu_grid = simulate_from_inhomogeneous_rate(
            t_grid,
            rate_null,
            rng=rng,
        )

        # This is the fake SE region.
        # It contains only fake SEs in the fit window, because fake_times
        # were generated only on t_grid = [fit_start_ms, fit_stop_ms).
        fake_se = make_fake_se_roi_like(
            DEs_template,
            fake_times,
            run_start_ns=None,
        )

        # ------------------------------------------------------------
        # 2. Fit fake data with null model: power law
        # ------------------------------------------------------------
        values_null_b, errors_null_b, cov_null_b, BIC_null_b, fval_null_b = (
            fit_model_by_name(
                null_model,
                run_id,
                s2_region,
                fake_se,
                S1_region,
                seconds_range,
                history_ms=history_ms,
            )
        )

        # ------------------------------------------------------------
        # 3. Fit fake data with alternative model: exp_additive
        # ------------------------------------------------------------
        values_alt_b, errors_alt_b, cov_alt_b, BIC_alt_b, fval_alt_b = (
            fit_model_by_name(
                alt_model,
                run_id,
                s2_region,
                fake_se,
                S1_region,
                seconds_range,
                history_ms=history_ms,
            )
        )

        TS = fval_null_b - fval_alt_b

        return {
            "i": i,
            "seed": seed,
            "success": True,
            "fval_plaw": float(fval_null_b),
            f"fval_{alt_model}": float(fval_alt_b),
            "TS": float(TS),
            "BIC_plaw": float(BIC_null_b),
            f"BIC_{alt_model}": float(BIC_alt_b),
            "values_plaw": values_to_list(values_null_b),
            f"values_{alt_model}": values_to_list(values_alt_b),
        }

    except Exception as e:
        return {
            "i": i,
            "seed": seed,
            "success": False,
            "error": repr(e),
        }
def bootstrap_lrt_general_parallel(
    run_id,
    pS2s,
    DEs_template,
    s1_times,
    null_model,
    alt_model,
    values_null_real,
    fval_null_real,
    fval_alt_real,
    seconds_range,
    n_boot=100,
    dt_ms=0.5,
    rng_seed=12345,
    history_ms=10000.0,
    n_workers=None,
    savedir=".",
    checkpoint_name="lrt_bootstrap_checkpoint.pkl",
):
    savedir = Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = savedir / checkpoint_name

    if n_workers is None:
        n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

    fit_inputs = {
        "run_id": run_id,
        "s2_region": pS2s,
        "DEs_template": DEs_template,
        "S1_region": s1_times,
        "seconds_range": seconds_range,
        "history_ms": history_ms,
        "dt_ms": dt_ms,
        "null_model": null_model,
        "alt_model": alt_model,
        "values_null": values_to_list(values_null_real),
    }

    TS_real = fval_null_real - fval_alt_real

    seeds = [rng_seed + i for i in range(n_boot)]
    results = []

    print(f"Running {n_boot} LRT bootstraps with {n_workers} workers", flush=True)
    print(f"Real-data TS = {TS_real:.6g}", flush=True)

    t0 = time.time()
    print("Testing one LRT bootstrap directly...", flush=True)
    test_result = run_one_lrt_bootstrap(0, rng_seed, fit_inputs)
    print("Direct LRT test:", test_result, flush=True)

    if not test_result["success"]:
        raise RuntimeError(f"Direct LRT test failed: {test_result.get('error')}")
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context = ctx) as executor:
        futures = [
            executor.submit(run_one_lrt_bootstrap, i, seeds[i], fit_inputs)
            for i in range(n_boot)
        ]

        for n_done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)

            print(
                f"LRT bootstrap {n_done}/{n_boot} complete "
                f"(i={result['i']}, success={result['success']})",
                flush=True,
            )

            if n_done % 10 == 0:
                results_sorted = sorted(results, key=lambda x: x["i"])
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(results_sorted, f)

                print(f"Saved checkpoint: {checkpoint_path}", flush=True)

    results = sorted(results, key=lambda x: x["i"])

    with open(checkpoint_path, "wb") as f:
        pickle.dump(results, f)

    boot_df = pd.DataFrame(results)

    TS_boot = boot_df.loc[boot_df["success"], "TS"].to_numpy(dtype=float)

    if len(TS_boot) == 0:
        summary = {
            "null_model": null_model,
            "alt_model": alt_model,
            "TS_real": float(TS_real),
            "N_boot": 0,
            "N_failed": int(len(boot_df)),
            "N_exceed": 0,
            "p_naive": np.nan,
            "p_plus1": np.nan,
            "history_ms": float(history_ms),
            "runtime_min": float((time.time() - t0) / 60),
        }
        return boot_df, summary

    n_exceed = int(np.sum(TS_boot >= TS_real))

    summary = {
        "null_model": null_model,
        "alt_model": alt_model,
        "TS_real": float(TS_real),
        "N_boot": int(len(TS_boot)),
        "N_failed": int(np.sum(~boot_df["success"])),
        "N_exceed": n_exceed,
        "p_naive": float(n_exceed / len(TS_boot)),
        "p_plus1": float((n_exceed + 1) / (len(TS_boot) + 1)),
        "mean_TS_boot": float(np.mean(TS_boot)),
        "std_TS_boot": float(np.std(TS_boot)),
        "min_TS_boot": float(np.min(TS_boot)),
        "max_TS_boot": float(np.max(TS_boot)),
        "history_ms": float(history_ms),
        "runtime_min": float((time.time() - t0) / 60),
    }

    print(f"Finished LRT bootstrap in {summary['runtime_min']:.2f} min", flush=True)

    return boot_df, summary
def run_one_gof_refit_bootstrap(b, seed, fit_inputs, model = "exp_additive"):
    """
    One independent additive-model GOF bootstrap with refit.

    Fake data are generated from the real fitted additive model.
    Then the additive model is refitted to the fake data.
    Then the fake-data binned Poisson deviance is computed against the refitted model.
    """

    import numpy as np
    import pandas as pd
    from conor_plugins import model as mod

    rng = np.random.default_rng(seed)

    try:
        run_id = fit_inputs["run_id"]
        values_real = fit_inputs[f"values_{model}_real"]
        pS2s = fit_inputs["pS2s"]
        DEs_template = fit_inputs["DEs_template"]
        s1_times = fit_inputs["s1_times"]
        seconds_range = fit_inputs["seconds_range"]

        fit_start_ms = fit_inputs["fit_start_ms"]
        fit_stop_ms = fit_inputs["fit_stop_ms"]
        t_grid = fit_inputs["t_grid"]
        bin_edges = fit_inputs["bin_edges"]
        live_intervals = fit_inputs["live_intervals"]
        rate_real = fit_inputs[f"rate_{model}_real"]
        history_ms = fit_inputs["history_ms"]

        fake_times, mu_grid = simulate_from_inhomogeneous_rate(
            t_grid,
            rate_real,
            rng=rng,
        )

        fake_times = fake_times[
            mod.make_live_mask(fake_times, live_intervals)
        ]

        fake_se = make_fake_se_roi_like(
            DEs_template,
            fake_times,
            run_start_ns=None,
        )
        if model == "exp_additive":
            values_fake, errors_fake, cov_fake, BIC_fake, fval_fake = (
                mod.cost_func_exp_additive(
                    run_id,
                    pS2s,
                    fake_se,
                    s1_times,
                    seconds_range=seconds_range,
                    model="exp_additive",
                    history_ms=history_ms,
                )
            )
        elif model == "powerlaw":
            values_fake, errors_fake, cov_fake, BIC_fake, fval_fake = (
                mod.cost_func(
                    run_id,
                    pS2s,
                    fake_se,
                    s1_times,
                    seconds_range=seconds_range,
                    model="new",
                    history_ms=history_ms,
                )
            )
        elif model == "exp_plaw":
            values_fake, errors_fake, cov_fake, BIC_fake, fval_fake = (
                mod.cost_func_exp_powerlaw(
                    run_id,
                    pS2s,
                    fake_se,
                    s1_times,
                    seconds_range=seconds_range,
                    model="exp",
                    history_ms=history_ms,
                )
            )
        elif model == "pure_exp":
            values_fake, errors_fake, cov_fake, BIC_fake, fval_fake = (
                mod.cost_func_pure_exp(
                    run_id,
                    pS2s,
                    fake_se,
                    s1_times,
                    seconds_range=seconds_range,
                    model="pure_exp",
                    history_ms=history_ms,
                )
            )
        elif model == "multi_exp":
            values_fake, errors_fake, cov_fake, BIC_fake, fval_fake = (
                mod.cost_func_multi_exp(
                    run_id,
                    pS2s,
                    fake_se,
                    s1_times,
                    seconds_range=seconds_range,
                    model="multi_exp",
                    history_ms=history_ms,
                )
            )

        _, rate_fake_fit = evaluate_rate_for_gof(
            t_grid,
            values_fake,
            pS2s,
            s1_times,
            fit_start_ms,
            fit_stop_ms,
            history_ms=history_ms,
            model = model
        )

        rate_fake_fit = mask_rate_to_live_intervals(
            t_grid,
            rate_fake_fit,
            live_intervals,
        )

        N_fake, mu_fake = binned_counts_and_expectations(
            fake_times,
            t_grid,
            rate_fake_fit,
            bin_edges,
            live_intervals=live_intervals,
        )

        D_fake = poisson_deviance_total(N_fake, mu_fake)

        return {
            "b": b,
            "seed": seed,
            "N_fake": int(len(fake_times)),
            "D": float(D_fake),
            f"fval_{model}_fake": float(fval_fake),
            f"BIC_{model}_fake": float(BIC_fake),
            f"values_{model}_fake": values_to_list(values_fake),
            f"errors_{model}_fake": values_to_list(errors_fake),
            "success": True,
        }

    except Exception as e:
        return {
            "b": b,
            "seed": seed,
            "N_fake": np.nan,
            "D": np.nan,
            f"fval_{model}_fake": np.nan,
            f"BIC_{model}_fake": np.nan,
            "success": False,
            "error": repr(e),
        }
def bootstrap_gof_with_refit_parallel(
    run_id,
    values_real,
    pS2s,
    DEs_template,
    s1_times,
    seconds_range,
    n_boot=50,
    bin_width_ms=500.0,
    dt_grid_ms=0.5,
    rng_seed=12345,
    history_ms=15000.0,
    n_workers=None,
    savedir=".",
    checkpoint_name="gof_refit_checkpoint.pkl",
    model = "exp_additive"
):
    """
    Parallel general model GOF bootstrap with refit.

    Fake datasets are generated from the real fitted additive model.
    Each fake dataset is refitted independently in a separate worker.
    """

    from conor_plugins import model as mod

    savedir = Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = savedir / checkpoint_name

    if n_workers is None:
        n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

    fit_start_ms = seconds_range[0] * 1e3
    fit_stop_ms = seconds_range[1] * 1e3
    tmin = 11.5

    live_intervals = get_live_intervals_for_fit(
        fit_start_ms,
        fit_stop_ms,
        pS2s,
        s1_times,
        tmin=tmin,
    )

    t_grid = np.arange(fit_start_ms, fit_stop_ms, dt_grid_ms)

    bin_edges = np.arange(
        fit_start_ms,
        fit_stop_ms + bin_width_ms,
        bin_width_ms,
    )
    bin_edges[-1] = fit_stop_ms

    # ------------------------------------------------------------
    # 1. Compute real-data GOF statistic using real best-fit params
    # ------------------------------------------------------------
    if model == "exp_additive":
        _, rate_real = evaluate_rate_for_gof(
            t_grid,
            values_real,
            pS2s,
            s1_times,
            fit_start_ms,
            fit_stop_ms,
            history_ms=history_ms,
            model="exp_additive",
        )
    elif model == "powerlaw":
        _, rate_real = evaluate_rate_for_gof(
            t_grid,
            values_real,
            pS2s,
            s1_times,
            fit_start_ms,
            fit_stop_ms,
            history_ms=history_ms,
            model = "powerlaw"
        )
    elif model == "pure_exp":
        _, rate_real = evaluate_rate_for_gof(
            t_grid,
            values_real,
            pS2s,
            s1_times,
            fit_start_ms,
            fit_stop_ms,
            history_ms=history_ms,
            model = "pure_exp"
        )
    elif model == "exp_plaw":
        _, rate_real = evaluate_rate_for_gof(
            t_grid,
            values_real,
            pS2s,
            s1_times,
            fit_start_ms,
            fit_stop_ms,
            history_ms=history_ms,
            model = "exp_plaw"
        )
    elif model == "multi_exp":
        _, rate_real = evaluate_rate_for_gof(
            t_grid,
            values_real,
            pS2s,
            s1_times,
            fit_start_ms,
            fit_stop_ms,
            history_ms=history_ms,
            model="multi_exp"
        )

    rate_real = mask_rate_to_live_intervals(
        t_grid,
        rate_real,
        live_intervals,
    )

    real_times = DEs_template["time_since_start"].astype(float)
    real_times = real_times[
        (real_times >= fit_start_ms)
        & (real_times < fit_stop_ms)
    ]
    real_times = real_times[
        mod.make_live_mask(real_times, live_intervals)
    ]

    N_real, mu_real = binned_counts_and_expectations(
        real_times,
        t_grid,
        rate_real,
        bin_edges,
        live_intervals=live_intervals,
    )

    D_real = poisson_deviance_total(N_real, mu_real)

    D_real_bin = poisson_deviance_per_bin(N_real, mu_real)

    bins_df = pd.DataFrame({
        "bin_start_ms": bin_edges[:-1],
        "bin_stop_ms": bin_edges[1:],
        "N_real": N_real,
        "mu_real": mu_real,
        "D_real_bin": D_real_bin,
    })

    # ------------------------------------------------------------
    # 2. Package shared inputs for workers
    # ------------------------------------------------------------

    fit_inputs = {
        "run_id": run_id,
        f"values_{model}_real": values_to_list(values_real),
        "pS2s": pS2s,
        "DEs_template": DEs_template,
        "s1_times": s1_times,
        "seconds_range": seconds_range,
        "fit_start_ms": fit_start_ms,
        "fit_stop_ms": fit_stop_ms,
        "t_grid": t_grid,
        "bin_edges": bin_edges,
        "live_intervals": live_intervals,
        f"rate_{model}_real": rate_real,
        "history_ms": history_ms,
    }

    seeds = [rng_seed + b for b in range(n_boot)]

    rows = []

    print(
        f"Running {n_boot} additive GOF-refit bootstraps "
        f"with {n_workers} workers",
        flush=True,
    )
    print(f"D_real = {D_real:.6g}", flush=True)

    t0 = time.time()
    print("Testing one GOF-refit bootstrap directly...", flush=True)
    test_row = run_one_gof_refit_bootstrap(0, rng_seed, fit_inputs, model = model)
    print("Direct GOF test:", test_row, flush=True)

    if not test_row["success"]:
        raise RuntimeError(
            f"Direct GOF-refit bootstrap failed: {test_row.get('error')}\n"
            f"{test_row.get('traceback', '')}"
        )
    # ------------------------------------------------------------
    # 3. Run fake refits in parallel
    # ------------------------------------------------------------
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context = ctx) as executor:
        futures = [
            executor.submit(
                run_one_gof_refit_bootstrap,
                b,
                seeds[b],
                fit_inputs,
                model = model
            )
            for b in range(n_boot)
        ]

        for n_done, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)

            print(
                f"GOF refit bootstrap {n_done}/{n_boot} finished "
                f"(b={row['b']}, success={row['success']})",
                flush=True,
            )

            if n_done % 10 == 0:
                rows_sorted = sorted(rows, key=lambda x: x["b"])

                with open(checkpoint_path, "wb") as f:
                    pickle.dump(rows_sorted, f)

                print(f"Saved checkpoint to {checkpoint_path}", flush=True)

    rows = sorted(rows, key=lambda x: x["b"])

    with open(checkpoint_path, "wb") as f:
        pickle.dump(rows, f)

    boot_gof_df = pd.DataFrame(rows)

    D_boot = boot_gof_df.loc[boot_gof_df["success"], "D"].to_numpy()

    if len(D_boot) == 0:
        summary = {
            "D_real": float(D_real),
            "N_boot": 0,
            "N_failed": int(len(boot_gof_df)),
            "N_exceed": 0,
            "p_naive": np.nan,
            "p_plus1": np.nan,
            "mean_D_boot": np.nan,
            "std_D_boot": np.nan,
            "min_D_boot": np.nan,
            "max_D_boot": np.nan,
            "bin_width_ms": float(bin_width_ms),
            "dt_grid_ms": float(dt_grid_ms),
            "history_ms": float(history_ms),
            "runtime_min": float((time.time() - t0) / 60),
        }

        return boot_gof_df, summary, bins_df

    n_exceed = np.sum(D_boot >= D_real)

    p_naive = n_exceed / len(D_boot)
    p_plus1 = (n_exceed + 1) / (len(D_boot) + 1)

    summary = {
        "D_real": float(D_real),
        "N_boot": int(len(D_boot)),
        "N_failed": int(np.sum(~boot_gof_df["success"])),
        "N_exceed": int(n_exceed),
        "p_naive": float(p_naive),
        "p_plus1": float(p_plus1),
        "mean_D_boot": float(np.mean(D_boot)),
        "std_D_boot": float(np.std(D_boot)),
        "min_D_boot": float(np.min(D_boot)),
        "max_D_boot": float(np.max(D_boot)),
        "bin_width_ms": float(bin_width_ms),
        "dt_grid_ms": float(dt_grid_ms),
        "history_ms": float(history_ms),
        "runtime_min": float((time.time() - t0) / 60),
    }

    return boot_gof_df, summary, bins_df

import matplotlib.pyplot as plt
# %%
def poisson_deviance_per_bin(N, mu, eps=1e-12):
    """
    Per-bin Poisson deviance:
        D_i = 2 [mu_i - N_i + N_i log(N_i / mu_i)]
    with the N_i=0 term handled correctly.
    """
    N = np.asarray(N, dtype=float)
    mu = np.asarray(mu, dtype=float)

    mu_safe = np.maximum(mu, eps)

    D = np.zeros_like(N, dtype=float)

    zero = N == 0
    nonzero = ~zero

    D[zero] = 2.0 * mu_safe[zero]
    D[nonzero] = 2.0 * (
        mu_safe[nonzero]
        - N[nonzero]
        + N[nonzero] * np.log(N[nonzero] / mu_safe[nonzero])
    )

    return D


def poisson_deviance_total(N, mu, eps=1e-12):
    return np.sum(poisson_deviance_per_bin(N, mu, eps=eps))
# %%
def binned_counts_and_expectations(
    event_times_ms,
    t_grid,
    rate,
    bin_edges_ms,
    live_intervals=None,
):
    """
    Compute observed counts N_i and model expected counts mu_i
    in bins.

    event_times_ms : observed/fake DE times in ms
    t_grid : grid where model rate is evaluated, in ms
    rate : model rate on t_grid, in events/ms
    bin_edges_ms : bin edges in ms
    live_intervals : optional list of live intervals, [(a,b), ...]
    """
    event_times_ms = np.asarray(event_times_ms, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)
    rate = np.asarray(rate, dtype=float)

    # Observed counts
    N, _ = np.histogram(event_times_ms, bins=bin_edges_ms)

    # Mask rate outside live intervals, if provided
    rate_use = np.array(rate, copy=True)

    if live_intervals is not None:
        live_mask = np.zeros_like(t_grid, dtype=bool)
        for a, b in live_intervals:
            live_mask |= (t_grid >= a) & (t_grid < b)
        rate_use[~live_mask] = 0.0

    # Expected counts by numerical integration in each bin
    mu = np.zeros(len(bin_edges_ms) - 1, dtype=float)

    for i in range(len(mu)):
        a = bin_edges_ms[i]
        b = bin_edges_ms[i + 1]

        m = (t_grid >= a) & (t_grid < b)

        if np.sum(m) < 2:
            mu[i] = 0.0
        else:
            mu[i] = np.trapezoid(rate_use[m], t_grid[m])

    return N, mu
# %%
def evaluate_rate_for_gof(
    t_grid,
    values,
    pS2s,
    s1_times,
    window_start_ms,
    window_stop_ms,
    history_ms,
    model = "exp_additive"
):
    if model == "exp_additive":
        # values order for additive:
        # s, n, tau, f_exp, tmin, c, d, k
        total_rate, rate = mod.new_exp_additive_pdf(
            t_grid,
            values[0], values[1], values[2], values[3],
            values[4], values[5], values[6], values[7],
            pS2s,
            s1_times,
            window_start_ms,
            window_stop_ms,
            model = "exp_additive",
            history_ms = history_ms
        )
    elif model == "powerlaw":
        # values order for powerlaw:
        # s, n, tmin, c, d, k
        total_rate, rate = mod.new_power_law_pdf(
            t_grid,
            values[0], values[1], values[2], values[3],
            values[4], values[5],
            pS2s,
            s1_times,
            window_start_ms,
            window_stop_ms,
            model="new",
            history_ms=history_ms)
    elif model == "pure_exp":
        # values order for exponential:
        # s, tau, tmin, c, d, k
        total_rate, rate = mod.pure_exp_pdf(
            t_grid,
            values[0], values[1], values[2], values[3],
            values[4], values[5],
            pS2s,
            s1_times,
            window_start_ms,
            window_stop_ms,
            model="pure_exp",
            history_ms=history_ms)
    elif model == "exp_plaw":
        # values order for exponential switch:
        # s, n, tau, t_switch, tmin, c, d, k
        total_rate, rate = mod.exp_power_law_pdf(
            t_grid,
            values[0], values[1], values[2], values[3],
            values[4], values[5], values[6], values[7],
            pS2s,
            s1_times,
            window_start_ms,
            window_stop_ms,
            model="exp",
            history_ms=history_ms)
    elif model == "multi_exp":
        # values order for multi exponential:
        # s, n, tmin, c, d, k
        total_rate, rate = mod.multi_exp_pdf(
            t_grid,
            values[0], values[1], values[2], values[3],
            values[4], values[5],
            pS2s,
            s1_times,
            window_start_ms,
            window_stop_ms,
            model="multi_exp",
            history_ms=history_ms)
        

    return total_rate, rate
# %%
def additive_gof_real_statistic(
    values_add_real,
    pS2s,
    DEs,
    s1_times,
    seconds_range,
    history_ms,
    bin_width_ms=500.0,
    dt_grid_ms=0.5,
):
    fit_start_ms = seconds_range[0] * 1e3
    fit_stop_ms = seconds_range[1] * 1e3

    tmin = 11.5

    live_intervals = get_live_intervals_for_fit(
        fit_start_ms,
        fit_stop_ms,
        pS2s,
        s1_times,
        tmin=tmin,
    )

    # Observed DEs in fit window
    event_times = DEs["time_since_start"].astype(float)
    event_times = event_times[
        (event_times >= fit_start_ms)
        & (event_times < fit_stop_ms)
    ]

    # Apply same live mask
    event_times = event_times[mod.make_live_mask(event_times, live_intervals)]

    # Evaluate model on grid
    t_grid = np.arange(fit_start_ms, fit_stop_ms, dt_grid_ms)

    total_rate, rate = evaluate_expadd_rate_for_gof(
        t_grid,
        values_add_real,
        pS2s,
        s1_times,
        fit_start_ms,
        fit_stop_ms,
        history_ms
    )

    rate = mask_rate_to_live_intervals(t_grid, rate, live_intervals)

    bin_edges = np.arange(fit_start_ms, fit_stop_ms + bin_width_ms, bin_width_ms)
    bin_edges[-1] = fit_stop_ms

    N, mu = binned_counts_and_expectations(
        event_times,
        t_grid,
        rate,
        bin_edges,
        live_intervals=live_intervals,
    )

    D_bins = poisson_deviance_per_bin(N, mu)
    D_total = np.sum(D_bins)

    gof_df = pd.DataFrame({
        "bin_start_ms": bin_edges[:-1],
        "bin_stop_ms": bin_edges[1:],
        "N_obs": N,
        "mu": mu,
        "D": D_bins,
    })

    return D_total, gof_df, event_times, t_grid, rate, live_intervals

# %%
def plot_real_standardized_residual_mosaic(
    bins_multi_df,
    window_tag,
    savedir,
    model_label=None,
    file_prefix=None,
    ncols=3,
):
    """
    One mosaic figure showing standardized Poisson residuals for the real data:

        r_i = (N_i - mu_i) / sqrt(mu_i)

    across all bin widths.
    """
    bin_widths = sorted(bins_multi_df["bin_width_ms"].unique(), reverse=True)

    fig, axes = make_mosaic_axes(
        n_panels=len(bin_widths),
        ncols=ncols,
        panel_width=5.0,
        panel_height=3.4,
    )

    for ax, bin_width_ms in zip(axes, bin_widths):
        df = bins_multi_df[
            bins_multi_df["bin_width_ms"].astype(float) == float(bin_width_ms)
        ].copy()

        df = df[
            np.isfinite(df["N_real"])
            & np.isfinite(df["mu_real"])
            & (df["mu_real"] > 0)
        ]

        if len(df) == 0:
            ax.set_visible(False)
            continue

        x = 0.5 * (
            df["bin_start_ms"].to_numpy(dtype=float)
            + df["bin_stop_ms"].to_numpy(dtype=float)
        )
        x = (x - x.min()) / 1000.0

        N = df["N_real"].to_numpy(dtype=float)
        mu = df["mu_real"].to_numpy(dtype=float)

        residual = np.full_like(mu, np.nan, dtype=float)

        good = np.isfinite(N) & np.isfinite(mu) & (mu > 0)
        residual[good] = (N[good] - mu[good]) / np.sqrt(mu[good])

        ax.plot(
            x[good],
            residual[good],
            marker=".",
            linestyle="none",
            markersize=2,
        )

        ax.axhline(0.0, linestyle="-", linewidth=1)
        ax.axhline(1.0, linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(-1.0, linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(2.0, linestyle=":", linewidth=0.8, alpha=0.7)
        ax.axhline(-2.0, linestyle=":", linewidth=0.8, alpha=0.7)
        ax.text(
         0.02,
         0.98,
         sigma_exceedance_text(residual),
         transform=ax.transAxes,
         ha="left",
         va="top",
         fontsize=8,
         bbox=dict(boxstyle="round", alpha=0.75),
        )

        ax.set_title(f"{bin_width_ms:g} ms")
        ax.set_xlabel("Time since window start [s]")
        ax.set_ylabel(r"$(N-\mu)/\sqrt{\mu}$")

        finite_res = residual[np.isfinite(residual)]
        if len(finite_res) > 0:
            lim = np.nanpercentile(np.abs(finite_res), 99)
            if np.isfinite(lim) and lim > 0:
                ax.set_ylim(-1.2 * max(3.0, lim), 1.2 * max(3.0, lim))

    if model_label is None:
        model_label = "model"

    fig.suptitle(
        f"Standardized Poisson residuals: {model_label}",
        y=1.02,
    )
    fig.tight_layout()

    if file_prefix is None:
        file_prefix = "model"

    save_mosaic_example_panel(
        fig=fig,
        axes=axes,
        panel_bin_widths=bin_widths,
        target_bin_width_ms=500.0,
        savedir=savedir,
        outname=(
            f"{window_tag}_{file_prefix}_real_standardized_residual_{500.0:g}ms_panel"
        ),
        dpi=300,
    )

    if file_prefix is None:
        file_prefix = "model"

    save_plot(
        fig,
        savedir,
        f"{window_tag}_{file_prefix}_real_standardized_residual_mosaic",
        dpi=200,
    )

    plt.close(fig)
    return fig
# %%
def find_parent_bin(child_row, parent_df):
    """
    Return the first parent row whose interval contains the child interval.
    """
    mask = (
        (parent_df["start_s"] <= child_row["start_s"]) &
        (parent_df["stop_s"] >= child_row["stop_s"])
    )
    candidates = parent_df[mask].copy()

    if len(candidates) == 0:
        return None

    # Choose the parent with the smallest width / closest containing bin
    candidates = candidates.sort_values("bin_width_ms")
    return candidates.iloc[0]
# %%
def child_bins_inside_parent(all_bins_df, parent_row, child_width):
    child = all_bins_df[all_bins_df["bin_width_ms"] == child_width].copy()

    inside = child[
        (child["bin_start_ms"] >= parent_row["bin_start_ms"])
        & (child["bin_stop_ms"] <= parent_row["bin_stop_ms"])
    ].copy()

    return inside.sort_values("D_real_bin", ascending=False)
# %%
def combine_all_bins(gof_bins_list, bin_widths):
    rows = []

    for df, bw in zip(gof_bins_list, bin_widths):
        temp = df.copy()
        temp["bin_width_ms"] = bw
        temp["start_s"] = temp["bin_start_ms"] / 1000.0
        temp["stop_s"] = temp["bin_stop_ms"] / 1000.0
        temp["center_s"] = 0.5 * (temp["start_s"] + temp["stop_s"])
        rows.append(temp)

    return pd.concat(rows, ignore_index=True)

# %%
def effective_chi2_from_bootstrap(D_boot):
    D_boot = np.asarray(D_boot, dtype=float)
    D_boot = D_boot[np.isfinite(D_boot)]

    mean = np.mean(D_boot)
    var = np.var(D_boot, ddof=1)

    nu_eff = 2 * mean**2 / var
    scale_eff = var / (2 * mean)

    return nu_eff, scale_eff, mean, var

# %%
from scipy.stats import chi2
# %%
def gof_pvalue_and_dof_summary(
    gof_boot_list,
    gof_summary_list,
    gof_bins_df,
    bin_widths,
    seconds_range,
    n_fit_params=8,
):
    rows = []

    for bw, boot_df, summary, bins_df, in zip(bin_widths, gof_boot_list, gof_summary_list, gof_bins_df):
        D_boot = boot_df.loc[boot_df["success"], "D"].to_numpy(dtype=float)
        D_boot = D_boot[np.isfinite(D_boot)]

        D_real = float(summary["D_real"])

        n_boot = len(D_boot)
        n_exceed = int(np.sum(D_boot >= D_real))

        p_empirical_plus1 = (n_exceed + 1) / (n_boot + 1)

        # If your summary contains number of bins, use it.
        # Otherwise infer from real-bin dataframe separately.
        start_time = seconds_range[0]*1e3
        end_time = seconds_range[1]*1e3
        s2_t = pS2s["time_since_start"].astype(np.float64)
        order = np.argsort(s2_t)
        s2_t_sorted = np.ascontiguousarray(s2_t[order])
        s1_t = S1s["time_since_start"].astype(np.float64)
        order_2 = np.argsort(s1_t)
        s1_t_sorted = np.ascontiguousarray(s1_t[order_2])

        valid = (
                np.isfinite(bins_df["N_real"])
                & np.isfinite(bins_df["mu_real"])
                & (bins_df["mu_real"] > 0)
        )

        n_bins_used = int(valid.sum())

        if np.isfinite(n_bins_used):
            nu_naive = n_bins_used - n_fit_params
            log_p_naive = chi2.logsf(D_real, df=nu_naive)
            minus_log10_p_naive = -log_p_naive / np.log(10)
        else:
            nu_naive = np.nan
            log_p_naive = np.nan
            minus_log10_p_naive = np.nan

        mean_D = np.mean(D_boot)
        var_D = np.var(D_boot, ddof=1)

        # Effective scaled chi-square
        nu_eff = 2 * mean_D**2 / var_D
        scale_eff = var_D / (2 * mean_D)

        log_p_eff = chi2.logsf(D_real / scale_eff, df=nu_eff)
        minus_log10_p_eff = -log_p_eff / np.log(10)

        rows.append({
            "bin_width_ms": bw,
            "D_real": D_real,
            "N_boot": n_boot,
            "N_exceed": n_exceed,
            "p_empirical_plus1": p_empirical_plus1,

            "n_live_bins": n_bins_used,
            "n_fit_params": n_fit_params,
            "nu_naive": nu_naive,
            "minus_log10_p_naive_chi2": minus_log10_p_naive,

            "D_boot_mean": mean_D,
            "D_boot_std": np.sqrt(var_D),
            "nu_eff": nu_eff,
            "scale_eff": scale_eff,
            "minus_log10_p_eff_chi2": minus_log10_p_eff,
        })

    return pd.DataFrame(rows)

def recompute_one_existing_refit_worker(args):
    """
    Worker for one existing GOF-refit bootstrap row.

    Reconstructs fake data and evaluates the already-refitted fake model once,
    then computes D for all requested bin widths.
    """
    import numpy as np
    import pandas as pd
    from conor_plugins import model as mod

    (
        row_dict,
        model_name,
        fake_values_column,
        values_real,
        pS2s,
        s1_times,
        seconds_range,
        bin_widths_ms,
        dt_grid_ms,
        history_ms,
        t_grid,
        live_intervals,
        rate_real,
        real_bin_edges_by_width,
    ) = args

    try:
        b = int(row_dict["b"])
        seed = int(row_dict["seed"])
        values_fake_fit = row_dict[fake_values_column]

        fit_start_ms = seconds_range[0] * 1e3
        fit_stop_ms = seconds_range[1] * 1e3

        rng = np.random.default_rng(seed)

        # Reconstruct the same fake data as the original GOF-refit bootstrap.
        fake_times, mu_grid = simulate_from_inhomogeneous_rate(
            t_grid,
            rate_real,
            rng=rng,
        )

        fake_times = fake_times[
            mod.make_live_mask(fake_times, live_intervals)
        ]

        # Evaluate already-refitted fake model ONCE.
        _, rate_fake_fit = evaluate_model_rate_by_name(
            model_name,
            t_grid,
            values_fake_fit,
            pS2s,
            s1_times,
            fit_start_ms,
            fit_stop_ms,
            history_ms=history_ms,
        )

        rate_fake_fit = mask_rate_to_live_intervals(
            t_grid,
            rate_fake_fit,
            live_intervals,
        )

        rows = []

        for bin_width_ms in bin_widths_ms:
            bw = float(bin_width_ms)
            bin_edges = real_bin_edges_by_width[bw]

            N_fake, mu_fake = binned_counts_and_expectations(
                fake_times,
                t_grid,
                rate_fake_fit,
                bin_edges,
                live_intervals=live_intervals,
            )

            D_fake = poisson_deviance_total(N_fake, mu_fake)

            # Preserve the old row contents, then add recomputed quantities.
            out = dict(row_dict)
            out.update({
                "model_name": model_name,
                "bin_width_ms": bw,
                "N_fake_recomputed": int(len(fake_times)),
                "mu_grid_recomputed": float(mu_grid),
                "D_recomputed": float(D_fake),
                "success_recomputed": True,
            })

            rows.append(out)

        return {
            "b": b,
            "success": True,
            "rows": rows,
            "error": None,
        }

    except Exception as e:
        import traceback
        return {
            "b": int(row_dict.get("b", -1)),
            "success": False,
            "rows": [],
            "error": repr(e),
            "traceback": traceback.format_exc(),
        }
def recompute_gof_multi_bin_from_existing_refits_parallel(
    boot_gof_df,
    model_name,
    values_real,
    fake_values_column,
    pS2s,
    DEs_template,
    s1_times,
    seconds_range,
    bin_widths_ms,
    dt_grid_ms=0.5,
    history_ms=15000.0,
    n_workers=None,
    savedir=".",
    checkpoint_name="multi_bin_recompute_checkpoint.pkl",
):
    """
    Parallel post-processing of existing GOF-refit results.

    This does NOT rerun Minuit refits.
    It evaluates each saved fake fit once, then computes D for all bin widths.
    """
    import os
    import time
    import pickle
    import multiprocessing as mp
    from pathlib import Path
    from concurrent.futures import ProcessPoolExecutor, as_completed

    savedir = Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = savedir / checkpoint_name

    if n_workers is None:
        n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

    fit_start_ms = seconds_range[0] * 1e3
    fit_stop_ms = seconds_range[1] * 1e3
    tmin = 11.5

    live_intervals = get_live_intervals_for_fit(
        fit_start_ms,
        fit_stop_ms,
        pS2s,
        s1_times,
        tmin=tmin,
    )

    t_grid = np.arange(fit_start_ms, fit_stop_ms, dt_grid_ms)

    # Real fitted model rate. This is the simulation truth used originally.
    _, rate_real = evaluate_model_rate_by_name(
        model_name,
        t_grid,
        values_real,
        pS2s,
        s1_times,
        fit_start_ms,
        fit_stop_ms,
        history_ms=history_ms,
    )

    rate_real = mask_rate_to_live_intervals(
        t_grid,
        rate_real,
        live_intervals,
    )

    # Real observed data.
    real_times = DEs_template["time_since_start"].astype(float)
    real_times = real_times[
        (real_times >= fit_start_ms)
        & (real_times < fit_stop_ms)
    ]
    real_times = real_times[
        mod.make_live_mask(real_times, live_intervals)
    ]

    # Precompute real-data D and bin edges for all bin widths.
    real_bin_edges_by_width = {}
    real_summary_info = {}
    bins_rows = []

    for bin_width_ms in bin_widths_ms:
        bw = float(bin_width_ms)

        bin_edges = np.arange(
            fit_start_ms,
            fit_stop_ms + bw,
            bw,
        )
        bin_edges[-1] = fit_stop_ms

        real_bin_edges_by_width[bw] = bin_edges

        N_real, mu_real = binned_counts_and_expectations(
            real_times,
            t_grid,
            rate_real,
            bin_edges,
            live_intervals=live_intervals,
        )

        D_real_bin = poisson_deviance_per_bin(N_real, mu_real)
        D_real = poisson_deviance_total(N_real, mu_real)

        real_summary_info[bw] = {
            "D_real": float(D_real),
            "N_real_total": int(np.sum(N_real)),
        }

        bins_rows.append(pd.DataFrame({
            "bin_width_ms": bw,
            "bin_start_ms": bin_edges[:-1],
            "bin_stop_ms": bin_edges[1:],
            "N_real": N_real,
            "mu_real": mu_real,
            "D_real_bin": D_real_bin,
        }))

    successful = boot_gof_df[boot_gof_df["success"]].copy()

    print(
        f"Parallel recomputing multi-bin GOF for {len(successful)} existing refits "
        f"with {n_workers} workers",
        flush=True,
    )
    print(f"Bin widths: {bin_widths_ms}", flush=True)

    # Convert rows to dictionaries so they pickle cleanly.
    tasks = []
    for _, row in successful.iterrows():
        tasks.append((
            row.to_dict(),
            model_name,
            fake_values_column,
            values_to_list(values_real),
            pS2s,
            s1_times,
            seconds_range,
            [float(bw) for bw in bin_widths_ms],
            dt_grid_ms,
            history_ms,
            t_grid,
            live_intervals,
            rate_real,
            real_bin_edges_by_width,
        ))

    all_boot_rows = []
    failures = []

    t0 = time.time()

    # Use spawn because this code touches numba/compiled code.
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        futures = [
            executor.submit(recompute_one_existing_refit_worker, task)
            for task in tasks
        ]

        for n_done, future in enumerate(as_completed(futures), start=1):
            result = future.result()

            if result["success"]:
                all_boot_rows.extend(result["rows"])
            else:
                failures.append(result)

            if n_done % 10 == 0:
                print(
                    f"Processed {n_done}/{len(tasks)} existing refits "
                    f"in {(time.time() - t0) / 60:.2f} min",
                    flush=True,
                )

                with open(checkpoint_path, "wb") as f:
                    pickle.dump(
                        {
                            "boot_rows": all_boot_rows,
                            "failures": failures,
                        },
                        f,
                    )

    boot_multi_df = pd.DataFrame(all_boot_rows)
    bins_multi_df = pd.concat(bins_rows, ignore_index=True)

    # Build summary per bin width.
    summary_rows = []

    for bin_width_ms in bin_widths_ms:
        bw = float(bin_width_ms)

        sub = boot_multi_df[
            boot_multi_df["bin_width_ms"].astype(float) == bw
        ].copy()

        D_boot = sub["D_recomputed"].to_numpy(dtype=float)
        D_boot = D_boot[np.isfinite(D_boot)]

        D_real = real_summary_info[bw]["D_real"]
        n_exceed = int(np.sum(D_boot >= D_real))

        summary_rows.append({
            "model_name": model_name,
            "bin_width_ms": bw,
            "D_real": float(D_real),
            "N_boot": int(len(D_boot)),
            "N_failed_original": int(len(boot_gof_df) - len(successful)),
            "N_failed_recompute": int(len(failures)),
            "N_exceed": int(n_exceed),
            "p_naive": float(n_exceed / len(D_boot)) if len(D_boot) else np.nan,
            "p_plus1": float((n_exceed + 1) / (len(D_boot) + 1)) if len(D_boot) else np.nan,
            "mean_D_boot": float(np.mean(D_boot)) if len(D_boot) else np.nan,
            "std_D_boot": float(np.std(D_boot)) if len(D_boot) else np.nan,
            "min_D_boot": float(np.min(D_boot)) if len(D_boot) else np.nan,
            "max_D_boot": float(np.max(D_boot)) if len(D_boot) else np.nan,
            "dt_grid_ms": float(dt_grid_ms),
            "history_ms": float(history_ms),
            "runtime_min": float((time.time() - t0) / 60),
        })

    summary_multi_df = pd.DataFrame(summary_rows)
    failures_df = pd.DataFrame(failures)

    with open(checkpoint_path, "wb") as f:
        pickle.dump(
            {
                "boot_rows": all_boot_rows,
                "failures": failures,
                "summary": summary_rows,
            },
            f,
        )

    print(
        f"Finished multi-bin recompute in {(time.time() - t0) / 60:.2f} min",
        flush=True,
    )

    if len(failures_df) > 0:
        print("Some recompute workers failed:", flush=True)
        print(failures_df[["b", "error"]].head(20), flush=True)

    return boot_multi_df, summary_multi_df, bins_multi_df, failures_df
def plot_lrt_ts_distribution(
    boot_df,
    summary,
    window_tag,
    savedir,
    outname=None,
    bins=40,
):
    """
    Plot bootstrap LRT TS distribution and mark real-data TS.
    """
    plot_dir = ensure_plot_dir(savedir)

    TS_real = float(summary["TS_real"])

    TS_boot = boot_df.loc[boot_df["success"], "TS"].to_numpy(dtype=float)
    TS_boot = TS_boot[np.isfinite(TS_boot)]

    if len(TS_boot) == 0:
        raise RuntimeError("No successful LRT bootstrap TS values found.")

    p_plus1 = summary.get("p_plus1", np.nan)
    n_boot = summary.get("N_boot", len(TS_boot))
    n_failed = summary.get("N_failed", np.nan)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        TS_boot,
        bins=bins,
        histtype="stepfilled",
        alpha=0.6,
        label="Bootstrap TS under null",
    )

    ax.axvline(
        TS_real,
        linestyle="--",
        linewidth=2,
        label=fr"Real TS = {TS_real:.3g}",
    )

    ax.set_xlabel(r"Likelihood-ratio statistic $TS = f_{\rm null} - f_{\rm alt}$")
    ax.set_ylabel("Bootstrap count")
    ax.set_title(f"LRT bootstrap distribution: {window_tag}")

    text = (
        f"N boot = {n_boot}\n"
        f"N failed = {n_failed}\n"
        f"p(+1) = {p_plus1:.4g}"
    )

    ax.text(
        0.98,
        0.95,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    ax.legend()
    fig.tight_layout()

    if outname is None:
        outname = f"{window_tag}_lrt_TS_distribution.png"

    fig.savefig(plot_dir / outname, bbox_inches="tight")
    plt.close(fig)

    return fig, ax


def make_fake_gof_bins_for_replicate(
    row,
    model_name,
    values_real,
    fake_values_column,
    pS2s,
    DEs_template,
    s1_times,
    seconds_range,
    bin_width_ms,
    dt_grid_ms=0.5,
    history_ms=15000.0,
):
    """
    Reconstruct one fake GOF bootstrap replicate and return per-bin fake
    observed/expected information.

    row should come from boot_gof_df or boot_multi_df and must contain:
      - seed
      - fake_values_column, e.g. 'values_add_fake'
    """
    fit_start_ms = seconds_range[0] * 1e3
    fit_stop_ms = seconds_range[1] * 1e3
    tmin = 11.5

    live_intervals = get_live_intervals_for_fit(
        fit_start_ms,
        fit_stop_ms,
        pS2s,
        s1_times,
        tmin=tmin,
    )

    t_grid = np.arange(fit_start_ms, fit_stop_ms, dt_grid_ms)

    _, rate_real = evaluate_model_rate_by_name(
        model_name,
        t_grid,
        values_real,
        pS2s,
        s1_times,
        fit_start_ms,
        fit_stop_ms,
        history_ms=history_ms,
    )

    rate_real = mask_rate_to_live_intervals(
        t_grid,
        rate_real,
        live_intervals,
    )

    rng = np.random.default_rng(int(row["seed"]))

    fake_times, mu_grid = simulate_from_inhomogeneous_rate(
        t_grid,
        rate_real,
        rng=rng,
    )

    fake_times = fake_times[
        mod.make_live_mask(fake_times, live_intervals)
    ]

    values_fake_fit = row[fake_values_column]

    _, rate_fake_fit = evaluate_model_rate_by_name(
        model_name,
        t_grid,
        values_fake_fit,
        pS2s,
        s1_times,
        fit_start_ms,
        fit_stop_ms,
        history_ms=history_ms,
    )

    rate_fake_fit = mask_rate_to_live_intervals(
        t_grid,
        rate_fake_fit,
        live_intervals,
    )

    bin_edges = np.arange(
        fit_start_ms,
        fit_stop_ms + bin_width_ms,
        bin_width_ms,
    )
    bin_edges[-1] = fit_stop_ms

    N_fake, mu_fake = binned_counts_and_expectations(
        fake_times,
        t_grid,
        rate_fake_fit,
        bin_edges,
        live_intervals=live_intervals,
    )

    D_fake_bin = poisson_deviance_per_bin(N_fake, mu_fake)

    fake_bins_df = pd.DataFrame({
        "bin_width_ms": float(bin_width_ms),
        "bin_start_ms": bin_edges[:-1],
        "bin_stop_ms": bin_edges[1:],
        "N_fake": N_fake,
        "mu_fake": mu_fake,
        "D_fake_bin": D_fake_bin,
    })

    return fake_bins_df, fake_times, t_grid, rate_fake_fit

def add_naive_chi2_pvalues_to_summary(
    summary_multi_df,
    bins_multi_df,
    n_fit_params=7,
):
    """
    Add naive chi-square p-values to the multi-bin GOF summary.

    Uses:
        dof = number of bins with finite mu_real > 0 - n_fit_params

    No effective chi-square scaling or bootstrap adjustment is applied.
    """

    from scipy.stats import chi2

    rows = []

    for _, row in summary_multi_df.iterrows():
        bw = float(row["bin_width_ms"])
        D_real = float(row["D_real"])

        bins_bw = bins_multi_df[
            bins_multi_df["bin_width_ms"].astype(float) == bw
        ].copy()

        valid = (
            np.isfinite(bins_bw["N_real"])
            & np.isfinite(bins_bw["mu_real"])
            & (bins_bw["mu_real"] > 0)
        )

        n_live_bins = int(valid.sum())
        dof = n_live_bins - n_fit_params

        out = row.to_dict()
        out["n_live_bins"] = n_live_bins
        out["n_fit_params"] = int(n_fit_params)
        out["chi2_dof_naive"] = int(dof)

        if dof > 0:
            p_chi2 = chi2.sf(D_real, df=dof)
            log_p_chi2 = chi2.logsf(D_real, df=dof)

            out["p_chi2_naive"] = float(p_chi2)
            out["minus_log10_p_chi2_naive"] = float(-log_p_chi2 / np.log(10))
        else:
            out["p_chi2_naive"] = np.nan
            out["minus_log10_p_chi2_naive"] = np.nan

        rows.append(out)

    return pd.DataFrame(rows)

from scipy.stats import chi2, norm
def add_reference_pvalues_to_summary(
    boot_multi_df,
    summary_multi_df,
    bins_multi_df,
    model_name,
    n_fit_params,
):
    """
    Adds:
      - bootstrap p-value
      - fitted Gaussian p-value
      - fitted chi2 p-value
      - naive chi2 p-value

    All are upper-tail p-values:
        p = P(D >= D_real)
    """

    rows = []
    D_col = get_gof_D_column(boot_multi_df)

    for _, row in summary_multi_df.iterrows():
        bw = float(row["bin_width_ms"])
        D_real = float(row["D_real"])

        boot_bw = boot_multi_df[
            boot_multi_df["bin_width_ms"].astype(float) == bw
        ].copy()

        if "success_recomputed" in boot_bw.columns:
            boot_bw = boot_bw[boot_bw["success_recomputed"] == True]
        elif "success" in boot_bw.columns:
            boot_bw = boot_bw[boot_bw["success"] == True]

        D_boot = boot_bw[D_col].to_numpy(dtype=float)
        D_boot = D_boot[np.isfinite(D_boot)]

        bins_bw = bins_multi_df[
            bins_multi_df["bin_width_ms"].astype(float) == bw
        ].copy()

        valid_bins = (
            np.isfinite(bins_bw["N_real"])
            & np.isfinite(bins_bw["mu_real"])
            & (bins_bw["mu_real"] > 0)
        )

        n_live_bins = int(valid_bins.sum())
        dof_naive = n_live_bins - int(n_fit_params)

        out = row.to_dict()
        out["model_name"] = model_name
        out["n_fit_params"] = int(n_fit_params)
        out["n_live_bins"] = n_live_bins
        out["chi2_dof_naive"] = int(dof_naive)

        if len(D_boot) == 0:
            out["p_gaussian"] = np.nan
            out["p_chi2_fit"] = np.nan
            out["p_chi2_naive"] = np.nan
            rows.append(out)
            continue

        # Bootstrap p-value
        n_exceed = int(np.sum(D_boot >= D_real))
        p_bootstrap = (n_exceed + 1.0) / (len(D_boot) + 1.0)

        # Gaussian fit to bootstrap D distribution
        gauss_mu = float(np.mean(D_boot))
        gauss_sigma = float(np.std(D_boot, ddof=1))

        if gauss_sigma > 0 and np.isfinite(gauss_sigma):
            p_gaussian = float(norm.sf(D_real, loc=gauss_mu, scale=gauss_sigma))
            p_gaussian_log = float(norm.logsf(D_real, loc=gauss_mu, scale=gauss_sigma))
        else:
            p_gaussian = np.nan
            p_gaussian_log = np.nan

        # Scaled chi2 fit to bootstrap D distribution
        mean_D = float(np.mean(D_boot))
        var_D = float(np.var(D_boot, ddof=1))

        if mean_D > 0 and var_D > 0:
            chi2_fit_nu = float(2.0 * mean_D**2 / var_D)
            chi2_fit_scale = float(var_D / (2.0 * mean_D))
            p_chi2_fit = float(chi2.sf(D_real / chi2_fit_scale, df=chi2_fit_nu))
            p_chi2_fit_log = float(chi2.logsf(D_real / chi2_fit_scale, df = chi2_fit_nu))
        else:

            chi2_fit_nu = np.nan
            chi2_fit_scale = np.nan
            p_chi2_fit = np.nan
            p_chi2_fit_log = np.nan

        # Naive chi2
        if dof_naive > 0:
            p_chi2_naive = float(chi2.sf(D_real, df=dof_naive))
            p_chi2_naive_log = float(chi2.logsf(D_real, df = dof_naive))
        else:

            p_chi2_naive = np.nan
            p_chi2_naive_log = np.nan

        out["D_boot_mean"] = mean_D
        out["D_boot_std"] = float(np.sqrt(var_D))
        out["gauss_mu"] = gauss_mu
        out["gauss_sigma"] = gauss_sigma
        out["chi2_fit_nu"] = chi2_fit_nu
        out["chi2_fit_scale"] = chi2_fit_scale

        out["p_gaussian"] = p_gaussian
        out["p_chi2_fit"] = p_chi2_fit
        out["p_chi2_naive"] = p_chi2_naive

        out["minus_log10_p_gaussian"] = float(-(p_gaussian_log / np.log(10)))
        out["minus_log10_p_chi2_fit"] = float(-(p_chi2_fit_log / np.log(10)))
        out["minus_log10_p_chi2_naive"] = float(-(p_chi2_naive_log / np.log(10)))

        rows.append(out)

    return pd.DataFrame(rows)
# def plot_gof_minus_log10_pvalues_vs_bin_width(
#     summary_with_chi2_df,
#     window_tag,
#     savedir,
#     outname=None,
# ):
#     """
#     Plot -log10(p) for empirical/bootstrap and naive chi-square p-values.
#     """
#     plot_dir = ensure_plot_dir(savedir)
#
#     df = summary_with_chi2_df.copy()
#     df = df.sort_values("bin_width_ms")
#
#     bin_width = df["bin_width_ms"].to_numpy(dtype=float)
#
#     fig, ax = plt.subplots(figsize=(8, 5))
#
#     if "p_plus1" in df.columns:
#         p_emp = df["p_plus1"].to_numpy(dtype=float)
#         minus_log10_emp = -np.log10(np.maximum(p_emp, 1e-300))
#
#         ax.plot(
#             bin_width,
#             minus_log10_emp,
#             marker="o",
#             label=r"Bootstrap $-\log_{10} p(+1)$",
#         )
#
#     minus_log10_chi2 = df["minus_log10_p_chi2_naive"].to_numpy(dtype=float)
#
#     ax.plot(
#         bin_width,
#         minus_log10_chi2,
#         marker="s",
#         label=r"Naive $\chi^2$ $-\log_{10} p$",
#     )
#
#     ax.set_xscale("log")
#
#     ax.set_xlabel("Bin width [ms]")
#     ax.set_ylabel(r"$-\log_{10}(p)$")
#     ax.set_title(f"GOF p-values vs bin width: {window_tag}")
#
#     ax.axhline(-np.log10(0.05), linestyle="--", linewidth=1, label="p = 0.05")
#
#     ax.legend()
#     fig.tight_layout()
#
#     if outname is None:
#         outname = f"{window_tag}_gof_minus_log10_pvalues_vs_bin_width.png"
#
#     fig.savefig(plot_dir / outname, bbox_inches="tight")
#     plt.close(fig)
#
#     return fig, ax

def plot_multi_model_gof_pvalues(
    comparison_df,
    window_tag,
    savedir,
):
    """
    Plot -log10(p) vs bin width for multiple models.
    Produces one figure for the Gaussian p-values and one for the chi^2 p-values.
    """

    df = comparison_df.copy()
    df = df.sort_values(["model_label", "bin_width_ms"])

    panels = [
        ("minus_log10_p_gaussian", "Gaussian-fit p-value", "gaussian"),
        ("minus_log10_p_chi2_fit", r"$\chi^2$-fit p-value", "chi2"),
    ]

    for col, title, suffix in panels:

        fig, ax = plt.subplots(
            figsize=(8.5, 5),
            constrained_layout=True,
        )

        for model_label, sub in df.groupby("model_label"):
            sub = sub.sort_values("bin_width_ms")
            color = "black" if model_label == "powerlaw" else None
            ax.plot(
                sub["bin_width_ms"],
                sub[col],
                marker="o",
                label=model_label,
                color = color
            )

        ax.axhline(
            -np.log10(0.05),
            linestyle="--",
            linewidth=1,
            label="p = 0.05",
        )

        ax.set_xscale("log")
        ax.set_xlabel("Bin width [ms]")
        ax.set_ylabel(r"$-\log_{10}(p)$")
        ax.set_title(f"{title}, run: {run_id}")
        ax.legend(fontsize=8)

        save_plot(
            fig,
            savedir,
            f"{window_tag}_multi_model_gof_{suffix}",
            dpi=300,
        )
        plt.close(fig)

# %%
SAVE_PDF_TOO = True
def save_plot(fig, savedir, stem, dpi=200, save_pdf=SAVE_PDF_TOO):
    """
    Save plots primarily as PNG. Optionally also save PDF.
    """
    plot_dir = ensure_plot_dir(savedir)

    stem = Path(stem).stem

    png_path = plot_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")

    if save_pdf:
        pdf_path = plot_dir / f"{stem}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")

    return png_path
# %%
from matplotlib.transforms import Bbox
def save_mosaic_example_panel(
    fig,
    axes,
    panel_bin_widths,
    target_bin_width_ms,
    savedir,
    outname,
    dpi=300,
    subdir="plots",
    pad_x=0.08,
    pad_y=1.18,
):
    """
    Save one selected panel from a mosaic figure as a standalone figure.

    This crops the existing axis from the mosaic, including its axis labels and title.
    """
    if target_bin_width_ms is None:
        return None

    plot_dir = ensure_plot_dir(savedir, subdir=subdir)

    widths = np.asarray([float(bw) for bw in panel_bin_widths], dtype=float)
    target = float(target_bin_width_ms)

    matches = np.where(np.isclose(widths, target, rtol=0, atol=1e-9))[0]

    if len(matches) == 0:
        print(
            f"Requested example panel {target:g} ms not found. "
            f"Available bin widths: {widths}"
        )
        return None

    j = int(matches[0])
    ax = axes[j]

    if not ax.get_visible():
        print(f"Requested panel {target:g} ms exists but is not visible.")
        return None

    # Important: make sure layout is finalized before cropping.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Tight bbox includes tick labels, axis labels, and title.
    bbox = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
    bottom_pad = 0.0
    top_pad = 0.18
    bbox = Bbox.from_extents(bbox.x0 - pad_x,bbox.y0-bottom_pad,bbox.x1 + pad_x,bbox.y1+top_pad)

    stem = str(outname).replace(" ", "_")
    png_path = plot_dir / f"{stem}.png"
    pdf_path = plot_dir / f"{stem}.pdf"

    fig.savefig(png_path, bbox_inches=bbox, dpi=dpi)
    fig.savefig(pdf_path, bbox_inches=bbox)

    return png_path
# %%
def make_mosaic_axes(n_panels, ncols=3, panel_width=5.0, panel_height=3.6):
    """
    Create a roughly square mosaic of axes for multiple bin-width plots.
    """
    n_panels = int(n_panels)
    ncols = min(ncols, n_panels)
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(panel_width * ncols, panel_height * nrows),
        squeeze=False,
    )

    axes_flat = axes.ravel()

    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    return fig, axes_flat[:n_panels]
# def plot_gof_deviance_mosaic(
#     boot_multi_df,
#     summary_multi_df,
#     window_tag,
#     savedir,
#     ncols=3,
# ):
#     """
#     One mosaic figure showing GOF deviance bootstrap distributions
#     for all bin widths in one window.
#     """
#     bin_widths = sorted(summary_multi_df["bin_width_ms"].unique(), reverse=True)
#
#     fig, axes = make_mosaic_axes(
#         n_panels=len(bin_widths),
#         ncols=ncols,
#         panel_width=5.0,
#         panel_height=3.6,
#     )
#
#     for ax, bin_width_ms in zip(axes, bin_widths):
#         boot_bw = boot_multi_df[
#             (boot_multi_df["bin_width_ms"] == bin_width_ms)
#             & (boot_multi_df["success_recomputed"] == True)
#         ].copy()
#
#         summary_bw = summary_multi_df[
#             summary_multi_df["bin_width_ms"] == bin_width_ms
#         ].iloc[0]
#
#         D_col = get_gof_D_column(boot_bw)
#         D_boot = boot_bw[D_col].to_numpy(dtype=float)
#         D_boot = D_boot[np.isfinite(D_boot)]
#
#         D_real = float(summary_bw["D_real"])
#
#         ax.hist(D_boot, bins=40, histtype="step", density=True)
#         ax.axvline(D_real, linestyle="--", linewidth=1.5)
#
#         p_plus1 = summary_bw.get("p_plus1", np.nan)
#
#         ax.set_title(
#             f"{bin_width_ms:g} ms\n"
#             f"D={D_real:.2g}, p={p_plus1:.3g}"
#         )
#         ax.set_xlabel("Poisson deviance")
#         ax.set_ylabel("Density")
#
#     fig.suptitle(f"GOF deviance distributions: {window_tag}", y=1.02)
#     fig.tight_layout()
#
#     save_plot(
#         fig,
#         savedir,
#         f"{window_tag}_gof_deviance_distributions_mosaic",
#         dpi=200,
#     )
#     plt.close(fig)
#
#     return fig

def plot_gof_deviance_mosaic_with_fits(
    boot_multi_df,
    summary_refs_df,
    window_tag,
    savedir,
    model_label,
    file_prefix,
    ncols=3,
):
    """
    Mosaic of GOF bootstrap deviance distributions.

    Each panel shows:
      - simulated D distribution
      - D_real vertical line
      - fitted Gaussian curve
      - fitted scaled-chi2 curve
    """

    D_col = get_gof_D_column(boot_multi_df)

    bin_widths = sorted(summary_refs_df["bin_width_ms"].unique(), reverse=True)

    fig, axes = make_mosaic_axes(
        n_panels=len(bin_widths),
        ncols=ncols,
        panel_width=5.2,
        panel_height=3.8,
    )

    for ax, bw in zip(axes, bin_widths):
        bw = float(bw)

        boot_bw = boot_multi_df[
            boot_multi_df["bin_width_ms"].astype(float) == bw
        ].copy()

        if "success_recomputed" in boot_bw.columns:
            boot_bw = boot_bw[boot_bw["success_recomputed"] == True]
        elif "success" in boot_bw.columns:
            boot_bw = boot_bw[boot_bw["success"] == True]

        D_boot = boot_bw[D_col].to_numpy(dtype=float)
        D_boot = D_boot[np.isfinite(D_boot)]

        summary_bw = summary_refs_df[
            summary_refs_df["bin_width_ms"].astype(float) == bw
        ].iloc[0]

        D_real = float(summary_bw["D_real"])

        if len(D_boot) == 0:
            ax.set_title(f"{bw:g} ms\nno bootstrap")
            ax.axis("off")
            continue

        ax.hist(
            D_boot,
            bins=40,
            density=True,
            histtype="stepfilled",
            alpha=0.35,
            label="Simulated deviance",
        )

        x_min = min(np.min(D_boot), D_real)
        x_max = max(np.max(D_boot), D_real)
        pad = 0.08 * (x_max - x_min) if x_max > x_min else 1.0
        x = np.linspace(max(0.0, x_min - pad), x_max + pad, 500)

        ax.axvline(
            D_real,
            linestyle="-",
            linewidth=1.8,
            label=rf"$D_\mathrm{{real}}={D_real:.2g}$",
        )

        # Gaussian fit
        mu = float(summary_bw["gauss_mu"])
        sigma = float(summary_bw["gauss_sigma"])

        if np.isfinite(mu) and np.isfinite(sigma) and sigma > 0:
            ax.plot(
                x,
                norm.pdf(x, loc=mu, scale=sigma),
                linestyle="--",
                linewidth=1.5,
                label="Gaussian fit",
            )

        # chi2 fit
        nu = float(summary_bw["chi2_fit_nu"])
        scale = float(summary_bw["chi2_fit_scale"])

        if np.isfinite(nu) and np.isfinite(scale) and nu > 0 and scale > 0:
            ax.plot(
                x,
                chi2.pdf(x / scale ,df=nu) / scale,
                linestyle=":",
                linewidth=1.8,
                label=rf"$scaled \chi^2$ fit, $\nu={nu:.0f}$",
            )

        p_gauss = summary_bw["minus_log10_p_gaussian"]
        p_chi2 = summary_bw["minus_log10_p_chi2_fit"]

        #ax.set_title(
            #rf"$-\log_{{10}}(p_G)={p_gauss:.2f}$, "
            #rf"$-\log_{{10}}(p_{{\chi^2}})={p_chi2:.2f}$"
        #)
        ax.set_title(f"{bw:g} ms", fontsize=10)
        ax.text(
           0.03,
           0.97,
           rf"$-\log_{{10}}(p_G)={p_gauss:.1f}$" "\n"
           rf"$-\log_{{10}}(p_{{\chi^2}})={p_chi2:.1f}$",
           transform=ax.transAxes,
           ha="left",
           va="top",
           fontsize=8,
           bbox=dict(
               boxstyle="round",
               facecolor="white",
               alpha=0.8,
               edgecolor="0.7",
           ),
        )

        ax.set_xlabel("Poisson deviance")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)

    fig.suptitle(
        f"GOF deviance with Gaussian and chi2 fits for {model_label}, and run: {run_id}",
        y=1.02,
    )
    fig.tight_layout()

    if file_prefix is None:
        file_prefix = "model"

    save_mosaic_example_panel(
        fig=fig,
        axes=axes,
        panel_bin_widths=bin_widths,
        target_bin_width_ms=500.0,
        savedir=savedir,
        outname=(
            f"{window_tag}_{file_prefix}_gof_deviance_{500.0:g}ms_panel"
        ),
        dpi=300,
    )

    save_plot(
        fig,
        savedir,
        f"{window_tag}_{file_prefix}_gof_deviance_mosaic_with_fits",
        dpi=200,
    )

    plt.close(fig)
    return fig
#%%
def plot_fake_standardized_residual_mosaic_random_seed(
    boot_gof_df,
    model_name,
    values_real,
    fake_values_column,
    pS2s,
    DEs_template,
    s1_times,
    seconds_range,
    bin_widths_ms,
    window_tag,
    savedir,
    model_label=None,
    file_prefix=None,
    dt_grid_ms=0.5,
    history_ms=15000.0,
    random_state=None,
    b=None,
    ncols=3,
):
    """
    Plot standardized Poisson residuals for one fake pseudo-experiment:

        r_i = (N_fake - mu_fake) / sqrt(mu_fake)

    If b is None, choose a random successful pseudo-experiment.
    If b is given, use that specific pseudo-experiment index.
    """

    rng = np.random.default_rng(random_state)

    df = boot_gof_df.copy()

    if "success" in df.columns:
        df = df[df["success"] == True]

    df = df[
        np.isfinite(df["seed"])
        & df[fake_values_column].notna()
    ].copy()

    # If using boot_multi_df by accident, there may be repeated rows per bin width.
    # Keep only one row per pseudo-experiment.
    if "b" in df.columns:
        df = df.drop_duplicates(subset=["b"])

    if len(df) == 0:
        raise RuntimeError("No successful fake pseudo-experiment rows available.")

    if b is None:
        row = df.iloc[rng.integers(0, len(df))]
    else:
        matches = df[df["b"].astype(int) == int(b)]
        if len(matches) == 0:
            raise RuntimeError(f"No successful fake pseudo-experiment found with b={b}.")
        row = matches.iloc[0]

    chosen_b = int(row["b"]) if "b" in row else -1
    chosen_seed = int(row["seed"])

    bin_widths_ms = [float(bw) for bw in bin_widths_ms]
    bin_widths_ms = sorted(bin_widths_ms, reverse=True)

    labels = [f"panel_{j}" for j in range(len(bin_widths_ms))]
    nrows = int(np.ceil(len(labels) / ncols))

    mosaic = []
    k = 0
    for _ in range(nrows):
        mosaic_row = []
        for _ in range(ncols):
            if k < len(labels):
                mosaic_row.append(labels[k])
            else:
                mosaic_row.append(".")
            k += 1
        mosaic.append(mosaic_row)

    fig, axd = plt.subplot_mosaic(
        mosaic,
        figsize=(5.0 * ncols, 3.4 * nrows),
        constrained_layout=True,
        empty_sentinel=".",
    )

    axes = [axd[label] for label in labels]

    for ax, bw in zip(axes, bin_widths_ms):
        fake_bins_df, fake_times, t_grid, rate_fake_fit = make_fake_gof_bins_for_replicate(
            row=row,
            model_name=model_name,
            values_real=values_real,
            fake_values_column=fake_values_column,
            pS2s=pS2s,
            DEs_template=DEs_template,
            s1_times=s1_times,
            seconds_range=seconds_range,
            bin_width_ms=bw,
            dt_grid_ms=dt_grid_ms,
            history_ms=history_ms,
        )

        df_bw = fake_bins_df.copy()

        valid = (
            np.isfinite(df_bw["N_fake"])
            & np.isfinite(df_bw["mu_fake"])
            & (df_bw["mu_fake"] > 0)
        )

        df_bw = df_bw[valid].copy()

        if len(df_bw) == 0:
            ax.set_visible(False)
            continue

        x = 0.5 * (
            df_bw["bin_start_ms"].to_numpy(dtype=float)
            + df_bw["bin_stop_ms"].to_numpy(dtype=float)
        )

        # Time relative to the start of this fitted window
        x = (x - seconds_range[0] * 1e3) / 1000.0

        N = df_bw["N_fake"].to_numpy(dtype=float)
        mu = df_bw["mu_fake"].to_numpy(dtype=float)

        residual = np.full_like(mu, np.nan, dtype=float)

        good = np.isfinite(N) & np.isfinite(mu) & (mu > 0)
        residual[good] = (N[good] - mu[good]) / np.sqrt(mu[good])

        ax.plot(
            x[good],
            residual[good],
            marker=".",
            linestyle="none",
            markersize=2,
        )

        ax.axhline(0.0, linestyle="-", linewidth=1)
        ax.axhline(1.0, linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(-1.0, linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(2.0, linestyle=":", linewidth=0.8, alpha=0.7)
        ax.axhline(-2.0, linestyle=":", linewidth=0.8, alpha=0.7)
        ax.text(
         0.02,
         0.98,
         sigma_exceedance_text(residual),
         transform=ax.transAxes,
         ha="left",
         va="top",
         fontsize=8,
         bbox=dict(boxstyle="round", alpha=0.75),
        )
        ax.set_title(f"{bw:g} ms")
        ax.set_xlabel("Time since window start [s]")
        ax.set_ylabel(r"$(N_{\rm fake}-\mu_{\rm fake})/\sqrt{\mu_{\rm fake}}$")

        finite_res = residual[np.isfinite(residual)]
        if len(finite_res) > 0:
            lim = np.nanpercentile(np.abs(finite_res), 99)
            if np.isfinite(lim) and lim > 0:
                lim = 1.2 * max(3.0, lim)
                ax.set_ylim(-lim, lim)

    if model_label is None:
        model_label = model_name

    fig.suptitle(
        (
            f"Fake-data standardized Poisson residuals: {model_label}\n"
            f"pseudo-experiment b={chosen_b}, seed={chosen_seed}"
        ),
        y=1.03,
    )
    if file_prefix is None:
        file_prefix = "model"

    save_mosaic_example_panel(
        fig=fig,
        axes=axes,
        panel_bin_widths=bin_widths_ms,
        target_bin_width_ms=500.0,
        savedir=savedir,
        outname=(
            f"{window_tag}_{file_prefix}_fake_standardized_residual_{500.0:g}ms_panel"
        ),
        dpi=300,
    )

    if file_prefix is None:
        file_prefix = model_name

    save_plot(
        fig,
        savedir,
        f"{window_tag}_{file_prefix}_fake_standardized_residual_mosaic_b{chosen_b}_seed{chosen_seed}",
        dpi=200,
    )

    plt.close(fig)

    return fig, chosen_b, chosen_seed

def sigma_exceedance_text(residual):
    """
    Count how many finite residual points are outside +/-1 and +/-2.

    residual = (N - mu) / sqrt(mu)
    """
    r = np.asarray(residual, dtype=float)
    r = r[np.isfinite(r)]

    n = len(r)

    if n == 0:
        return "N bins = 0\n|r|>1: --\n|r|>2: --"

    n_gt1 = int(np.sum(np.abs(r) > 1.0))
    n_gt2 = int(np.sum(np.abs(r) > 2.0))

    f_gt1 = 100.0 * n_gt1 / n
    f_gt2 = 100.0 * n_gt2 / n

    return (
        f"N bins = {n}\n"
        f"|r| > 1: {n_gt1} ({f_gt1:.1f}%)\n"
        f"|r| > 2: {n_gt2} ({f_gt2:.1f}%)"
    )
#%%
def plot_real_and_fake_ratio_overlay_mosaic(
    summary_multi_df,
    bins_multi_df,
    first_success,
    values_alt_real,
    pS2s,
    S1s,
    live_intervals,
    window_start_ms,
    window_stop_ms,
    window_tag,
    savedir,
    ncols=3,
):
    """
    One mosaic figure comparing real and one fake replicate across bin widths.
    """
    bin_widths = sorted(summary_multi_df["bin_width_ms"].unique(), reverse=True)

    fig, axes = make_mosaic_axes(
        n_panels=len(bin_widths),
        ncols=ncols,
        panel_width=5.0,
        panel_height=3.4,
    )

    for ax, bin_width_ms in zip(axes, bin_widths):
        fake_bins_df, fake_times, t_grid, rate_fake_fit = make_fake_gof_bins_for_replicate(
            row=first_success,
            bin_width_ms=bin_width_ms,
            values_alt_real=values_alt_real,
            pS2s=pS2s,
            S1s=S1s,
            live_intervals=live_intervals,
            window_start_ms=window_start_ms,
            window_stop_ms=window_stop_ms,
        )

        real_df = bins_multi_df[
            bins_multi_df["bin_width_ms"] == bin_width_ms
        ].copy()

        real_df = real_df[
            np.isfinite(real_df["N_real"])
            & np.isfinite(real_df["mu_real"])
            & (real_df["mu_real"] > 0)
        ]

        fake_bins_df = fake_bins_df[
            np.isfinite(fake_bins_df["N_fake"])
            & np.isfinite(fake_bins_df["mu_fake_fit"])
            & (fake_bins_df["mu_fake_fit"] > 0)
        ]

        x_real = 0.5 * (
            real_df["bin_start_ms"].to_numpy()
            + real_df["bin_stop_ms"].to_numpy()
        )
        x_real = (x_real - window_start_ms) / 1000.0

        ratio_real = real_df["N_real"].to_numpy(dtype=float) / real_df["mu_real"].to_numpy(dtype=float)

        x_fake = 0.5 * (
            fake_bins_df["bin_start_ms"].to_numpy()
            + fake_bins_df["bin_stop_ms"].to_numpy()
        )
        x_fake = (x_fake - window_start_ms) / 1000.0

        ratio_fake = fake_bins_df["N_fake"].to_numpy(dtype=float) / fake_bins_df["mu_fake_fit"].to_numpy(dtype=float)

        ax.plot(x_real, ratio_real, marker=".", linestyle="none", markersize=2, label="Real")
        ax.plot(x_fake, ratio_fake, marker=".", linestyle="none", markersize=2, alpha=0.6, label="Fake")

        ax.axhline(1.0, linestyle="--", linewidth=1)

        ax.set_title(f"{bin_width_ms:g} ms")
        ax.set_xlabel("Time since window start [s]")
        ax.set_ylabel(r"$N/\mu$")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    fig.suptitle(f"Real vs fake observed/expected ratio: {window_tag}", y=1.02)
    fig.tight_layout()

    if file_prefix is None:
        file_prefix = "model"

    save_mosaic_example_panel(
        fig=fig,
        axes=axes,
        panel_bin_widths=bin_widths,
        target_bin_width_ms=500.0,
        savedir=savedir,
        outname=(
            f"{window_tag}_{file_prefix}_real_fake_ratio_{500.0:g}ms_panel"
        ),
        dpi=300,
    )

    save_plot(
        fig,
        savedir,
        f"{window_tag}_real_fake_ratio_overlay_mosaic",
        dpi=200,
    )
    plt.close(fig)

    return fig

MODEL_SPECS = {
    "powerlaw": {
        "label": "Power law",
        "file_prefix": "powerlaw",
        "fake_values_column": "values_powerlaw_fake",
        "n_fit_params": 5,
    },
    "exp_additive": {
        "label": "PL + Exp additive",
        "file_prefix": "exp_additive",
        "fake_values_column": "values_exp_additive_fake",
        "n_fit_params": 7,
    },
    "pure_exp": {
        "label": "Pure exponential",
        "file_prefix": "pure_exp",
        "fake_values_column": "values_pure_exp_fake",
        "n_fit_params": 5,
    },
    "exp_plaw": {
        "label": "Exp switch",
        "file_prefix": "exp_plaw",
        "fake_values_column": "values_exp_plaw_fake",
        "n_fit_params": 7,
    },
    "multi_exp": {
        "label": "Multi exponential",
        "file_prefix": "multi_exp",
        "fake_values_column": "values_multi_exp_fake",
        "n_fit_params": 5,
    },
}

# %%
if __name__ == "__main__":
     RUN_LRT = False
     RUN_GOF_REFITS = False
     RECOMPUTE_GOF_BINS_ONLY = False
     MAKE_PLOTS = True

     st = straxen.contexts.xenonnt_online()
     st = register_conor_plugins(st)

     scratch_dir = "/home6/s5496527/projects/xenon_analysis/strax_data"

     st.storage = [strax.DataDirectory(scratch_dir, readonly=True, deep_scan=True),]+st.storage
     run_id = "028749"
     savedir = f"/scratch/s5496527/results/{run_id}"
     os.makedirs(savedir, exist_ok=True)
     runs_df = st.get_df(run_id, targets = ("peak_basics", "peaks"))
     # %%
     print("storage:")
     for s in st.storage:
         print(s)

     runs_df = st.select_runs(run_id=run_id)
     print(runs_df)
     print("len:", len(runs_df))
     # %%
     #pS2s, DEs, S1s, peaks, _ , debug= data_selection.data_selection_new(runs_df)
     #peak_dict = {"pS2s": pS2s, "DEs": DEs, "S1s": S1s, "peaks": peaks, "debug_info": debug}
     #with open(f'{savedir}/peak_dict.pkl', 'wb') as f:
         #pickle.dump(peak_dict, f)
     with open(f'{savedir}/peak_dict.pkl', 'rb') as f:
           peak_dict = pickle.load(f)
     pS2s = peak_dict["pS2s"]
     DEs = peak_dict["DEs"]
     S1s = peak_dict["S1s"]
     peaks = peak_dict["peaks"]
     # %%
     import importlib
     importlib.reload(mod)
      #run 026195
     #run_start = int(1628071231446000000)
     #run_end  = int(1628071324212000000)
      #run 028749
     run_start = int(1632192923000000000)
     run_end   = int(1632194726417555200)
      #run 044281
     #run_start = int(1655106353000000000)
     #run_end = int(1655106567173528320)
     run_duration = (run_end - run_start) / 1e9

     window_size = 300 #seconds
     base_width = 0.5 #seconds
     margin = 3 #seconds

     usable_start = margin
     usable_end = run_duration - margin
     usable_duration = usable_end - usable_start

     n_windows = int(np.ceil(usable_duration / window_size))
     for i in range(n_windows):
        start_sec = usable_start + i * window_size
        end_sec = min(start_sec + window_size, usable_end)

        if end_sec <= start_sec:
            continue

        seconds_range = (start_sec, end_sec)

        fit_start_ms = start_sec * 1e3
        fit_stop_ms = end_sec * 1e3

        history_ms = 15000.0
        source_start_ms = max(0.0, fit_start_ms - history_ms)
        source_stop_ms = fit_stop_ms

        s2_region = pS2s[
            (pS2s["time_since_start"] >= source_start_ms)
            & (pS2s["time_since_start"] <= source_stop_ms)
        ]

        se_region = DEs[
            (DEs["time_since_start"] >= fit_start_ms)
            & (DEs["time_since_start"] < fit_stop_ms)
        ]

        s1_times_full = S1s["time_since_start"]
        S1_region = s1_times_full[
            (s1_times_full >= fit_start_ms)
            & (s1_times_full < fit_stop_ms)
        ]

        print(f"\nWindow {i}: {start_sec:.3f} - {end_sec:.3f} s")
        print("N pS2:", len(s2_region), "N DE:", len(se_region), "N S1:", len(S1_region))

        try:
            window_tag = f"window_{i:03d}_{start_sec:.0f}_{end_sec:.0f}s"
            if RUN_LRT or RUN_GOF_REFITS:
                 GOF_MODELS = ["powerlaw", "exp_additive", "pure_exp", "exp_plaw", "multi_exp"]
                 for model in GOF_MODELS:

                    values_null_real, errors_null_real, cov_null_real, BIC_null_real, fval_null_real = fit_model_by_name(
                     "powerlaw",
                      run_id,
                      s2_region,
                      se_region,
                      S1_region,
                      seconds_range,
                      history_ms=history_ms,
                    )
            
                    values_alt_real, errors_alt_real, cov_alt_real, BIC_alt_real, fval_alt_real = fit_model_by_name(
                      model,
                      run_id,
                      s2_region,
                      se_region,
                      S1_region,
                      seconds_range,
                      history_ms=history_ms,
                    )
                    if RUN_LRT:
                        boot_df, summary = bootstrap_lrt_general_parallel(
                        run_id=run_id,
                        pS2s=s2_region,
                        DEs_template=se_region,
                        s1_times=S1_region,
                        null_model="powerlaw",
                        alt_model=model,
                        values_null_real=values_null_real,
                        fval_null_real=fval_null_real,
                        fval_alt_real=fval_alt_real,
                        seconds_range=seconds_range,
                        n_boot=200,
                        dt_ms=0.5,
                        rng_seed=12345,
                        history_ms=history_ms,
                        n_workers=40,
                        savedir=savedir,
                        checkpoint_name= f"window_{i}/{window_tag}_lrt_plaw_vs_{model}_checkpoint.pkl",
                        )
            
                        boot_df.to_pickle(Path(savedir) / f"window_{i}/{window_tag}_lrt_plaw_vs_{model}_boot_df.pkl")
            
                        with open(Path(savedir) / f"window_{i}/{window_tag}_lrt_plaw_vs_{model}_summary.pkl", "wb") as f:
                           pickle.dump(summary, f)
                    if RUN_GOF_REFITS:

                        values_real, errors_real, cov_real, BIC_real, fval_real = fit_model_by_name(
                           model,
                           run_id,
                           s2_region,
                           se_region,
                           S1_region,
                           seconds_range,
                           history_ms=history_ms,
                        )

                        boot_gof_df, summary_gof, bins_df = bootstrap_gof_with_refit_parallel(
                        run_id=run_id,
                        values_real=values_real,
                        pS2s=s2_region,
                        DEs_template=se_region,
                        s1_times=S1_region,
                        seconds_range=seconds_range,
                        n_boot=400,
                        bin_width_ms=500.0,
                        dt_grid_ms=0.5,
                        rng_seed=12345,
                        history_ms=history_ms,
                        n_workers=40,
                        savedir=savedir,
                        checkpoint_name= f"window_{i}/{window_tag}_{model}_gof_refit_checkpoint.pkl",
                        model = model
                        )

                        boot_gof_df.to_pickle(Path(savedir) / f"window_{i}/{window_tag}_{model}_gof_refit_boot_df.pkl")
                        bins_df.to_pickle(Path(savedir) / f"window_{i}/{window_tag}_{model}_gof_refit_bins_df.pkl")

                        summary_gof["model_name"] = model
                        summary_gof["values_alt_real"] = values_to_list(values_real)
                        summary_gof["errors_alt_real"] = values_to_list(errors_real)
                        summary_gof["BIC_alt_real"] = float(BIC_real)
                        summary_gof["fval_alt_real"] = float(fval_real)
                        summary_gof["history_ms"] = float(history_ms)
                        summary_gof["seconds_range"] = tuple(seconds_range)

                        with open(Path(savedir) / f"window_{i}/{window_tag}_{model}_gof_refit_summary.pkl", "wb") as f:
                            pickle.dump(summary_gof, f)

            if RECOMPUTE_GOF_BINS_ONLY:
                GOF_MODELS = ["powerlaw", "exp_additive", "pure_exp", "exp_plaw", "multi_exp"]
                for model in GOF_MODELS:

                    boot_gof_path = Path(savedir) / f"window_{i}/{window_tag}_{model}_gof_refit_boot_df.pkl"
                    summary_gof_path = Path(savedir) / f"window_{i}/{window_tag}_{model}_gof_refit_summary.pkl"

                    boot_gof_df = pd.read_pickle(boot_gof_path)

                    if summary_gof_path.exists():
                        with open(summary_gof_path, "rb") as f:
                            summary_gof_old = pickle.load(f)

                        if "values_real" in summary_gof_old:
                            values_alt_real = summary_gof_old["values_real"]
                        else:
                            values_alt_real, errors_alt_real, cov_alt_real, BIC_alt_real, fval_alt_real = fit_model_by_name(
                                model,
                                run_id,
                                s2_region,
                                se_region,
                                S1_region,
                                seconds_range,
                                history_ms=history_ms,
                            )
                    else:
                        values_alt_real, errors_alt_real, cov_alt_real, BIC_alt_real, fval_alt_real = fit_model_by_name(
                            model,
                            run_id,
                            s2_region,
                            se_region,
                            S1_region,
                            seconds_range,
                            history_ms=history_ms,
                        )

                    bin_widths_ms = [10000.0, 5000.0, 2000.0, 1000.0, 500.0, 250.0, 100.0, 50.0, 10.0]
                    spec = MODEL_SPECS[model]
                    boot_multi_df, summary_multi_df, bins_multi_df, failures_df = recompute_gof_multi_bin_from_existing_refits_parallel(
                        boot_gof_df=boot_gof_df,
                        model_name=model,
                        values_real=values_alt_real,
                        fake_values_column=spec["fake_values_column"],
                        pS2s=s2_region,
                        DEs_template=se_region,
                        s1_times=S1_region,
                        seconds_range=seconds_range,
                        bin_widths_ms=bin_widths_ms,
                        dt_grid_ms=0.5,
                        history_ms=history_ms,
                        n_workers=40,
                        savedir=savedir,
                        checkpoint_name= f"window_{i}/{window_tag}_{model}_gof_refit_multi_bin_recompute_checkpoint.pkl",
                    )

                    boot_multi_df.to_pickle(
                        Path(savedir) / f"window_{i}/{window_tag}_{model}_gof_refit_multi_bin_boot_df.pkl"
                    )

                    summary_multi_df.to_pickle(
                        Path(savedir) / f"window_{i}/{window_tag}_{model}_gof_refit_multi_bin_summary_df.pkl"
                    )

                    bins_multi_df.to_pickle(
                        Path(savedir) / f"window_{i}/{window_tag}_{model}_gof_refit_multi_bin_bins_df.pkl"
                    )

                    failures_df.to_pickle(
                        Path(savedir) / f"window_{i}/{window_tag}_{model}_gof_refit_multi_bin_failures_df.pkl"
                    )
            if MAKE_PLOTS:
                if run_id == "028749":

                    window_tags = [
                        "window_000_3_303s",
                        "window_001_303_603s",
                        "window_002_603_903s",
                        "window_003_903_1203s",
                        "window_004_1203_1503s",
                        "window_005_1503_1800s",
                    ]
                elif run_id == "026195":
                     window_tags = ["window_000_3_90s"]
                else:
                     window_tags =  [
                        "window_000_3_303s",
                        "window_001_303_603s",
                        "window_002_603_903s",
                        "window_003_903_1203s",
                        "window_004_1203_1503s",
                        "window_005_1503_1800s",
                    ]
                GOF_MODELS = ["powerlaw", "exp_additive", "pure_exp", "exp_plaw", "multi_exp"]

                comparison_rows = []

                for model_name in GOF_MODELS:
                    spec = MODEL_SPECS[model_name]

                    label = spec["label"]
                    prefix = spec["file_prefix"]
                    n_fit_params = spec["n_fit_params"]

                    boot_path = Path(savedir) / f"window_{i}/{window_tag}_{prefix}_gof_refit_multi_bin_boot_df.pkl"
                    summary_path = Path(savedir) / f"window_{i}/{window_tag}_{prefix}_gof_refit_multi_bin_summary_df.pkl"
                    bins_path = Path(savedir) / f"window_{i}/{window_tag}_{prefix}_gof_refit_multi_bin_bins_df.pkl"

                    if not (boot_path.exists() and summary_path.exists() and bins_path.exists()):
                        print(f"Missing GOF files for {model_name}; skipping.")
                        continue

                    boot_multi_df = pd.read_pickle(boot_path)
                    summary_multi_df = pd.read_pickle(summary_path)
                    bins_multi_df = pd.read_pickle(bins_path)

                    summary_refs_df = add_reference_pvalues_to_summary(
                        boot_multi_df=boot_multi_df,
                        summary_multi_df=summary_multi_df,
                        bins_multi_df=bins_multi_df,
                        model_name=model_name,
                        n_fit_params=n_fit_params,
                    )

                    summary_refs_df["model_label"] = label
                    summary_refs_df["file_prefix"] = prefix

                    summary_refs_df.to_pickle(
                        Path(savedir) / f"window_{i}/{window_tag}_{prefix}_summary_with_reference_pvalues.pkl"
                    )

                    summary_refs_df.to_csv(
                        Path(savedir) / f"window_{i}/{window_tag}_{prefix}_summary_with_reference_pvalues.csv",
                        index=False,
                    )

                    plot_real_standardized_residual_mosaic(
                        bins_multi_df=bins_multi_df,
                        window_tag=window_tag,
                        savedir=Path(savedir) / f"window_{i}",
                        model_label=label,
                        file_prefix=prefix,
                    )
                    
                    fake_values_column = f"values_{model_name}_fake"
                    summary_gof_path = Path(savedir) / f"window_{i}/{window_tag}_{prefix}_gof_refit_summary.pkl"
                    with open(summary_gof_path, "rb") as f:
                         summary_gof = pickle.load(f)
                    bin_widths_ms = [10000.0, 5000.0, 2000.0, 1000.0, 500.0, 250.0, 100.0, 50.0, 10.0]
                    values_real = summary_gof["values_alt_real"]

                    plot_fake_standardized_residual_mosaic_random_seed(
                        boot_gof_df=boot_multi_df,
                        model_name=model_name,
                        values_real=values_real,
                        fake_values_column=fake_values_column,
                        pS2s=s2_region,
                        DEs_template=se_region,
                        s1_times=S1_region,
                        seconds_range=seconds_range,
                        bin_widths_ms=bin_widths_ms,
                        window_tag=window_tag,
                        savedir=Path(savedir) / f"window_{i}",
                        model_label=label,
                        file_prefix=prefix,
                        dt_grid_ms=0.5,
                        history_ms=15000.0,
                        random_state=12345,
                    )

                    plot_gof_deviance_mosaic_with_fits(
                        boot_multi_df=boot_multi_df,
                        summary_refs_df=summary_refs_df,
                        window_tag=window_tag,
                        savedir=Path(savedir) / f"window_{i}",
                        model_label=label,
                        file_prefix=prefix,
                    )

                    comparison_rows.append(summary_refs_df)

                if comparison_rows:
                    comparison_df = pd.concat(comparison_rows, ignore_index=True)

                    comparison_df.to_csv(
                        Path(savedir) / f"window_{i}/{window_tag}_multi_model_gof_pvalue_comparison.csv",
                        index=False,
                    )

                    plot_multi_model_gof_pvalues(
                        comparison_df=comparison_df,
                        window_tag=window_tag,
                        savedir=Path(savedir) / f"window_{i}",
                    )

        except Exception as e:
            print(f"Window {i} failed: {repr(e)}", flush=True)
            continue
