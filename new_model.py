import numpy as np
import matplotlib.pyplot as plt
from iminuit import cost, Minuit
from jacobi import propagate
import pandas as pd
# from scipy import stats
from numba import njit, prange
import time
#-----------------------------------------------------------------------------------------------------------------------
#basically just the norms of each rate, both s*A^c*W^d and q*n_{e,rec}
@njit(cache=False)
def _compute_norms_basic(s, c, d, areas, ranges):
    # areas in phe, ranges in ms (you already divide by 1e6 for range_50p_area upstream)
    return s * (areas**c)  * (ranges**d)

@njit(cache=False)
def _compute_norms_source_ne(q, n_electron_rec):
    return q * n_electron_rec
#-----------------------------------------------------------------------------------------------------------------------
# Here we calculate the cdf for the power law (cdf_scalar) and for the exponential (exp_cdf_scalar)
@njit(cache=False)
def _cdf_scalar(x, tmin, n):
    # CDF of the single-pS2 power law at 'x' (scalar); used for the "cut" term
    return 0.0 if x < tmin else 1.0 - (tmin / x) ** (n - 1.0)

@njit(cache=False)
def _exp_cdf_scalar(x, tau, tmin):
    if tau <= 0.0:
        return 0.0

    if x <= tmin:
        return 0.0

    return 1.0 - np.exp(-(x - tmin) / tau)
#-----------------------------------------------------------------------------------------------------------------------
#below here we calculate the pdf for the power law (powerlaw_kernel_value_normed) and for the exponential (exp_kernel_value_normed)
# for plaw we need to ensure n < 1.0, otherwise the integral diverges, also if dt is less than tmin it doesn't really make sense.
#similarly tau being <=0 would also be problematic so we account for that (though tbh I'm not sure how often it even comes up if at all)
#basically all thats happening here is actually getting integral values either at a specific point or within a range
@njit(cache=False)
def _powerlaw_kernel_value_normed(dt, n, tmin):
    """
    Creates a single value of the normalized power-law kernel at a time dt after tmin

    uses: h(u) = (n - 1)/tmin * (dt/tmin)^(-n), for dt > tmin and n < 1.0.
    """
    if n < 1.0 or dt < tmin:
        return 0.0

    scale = (n - 1.0) / tmin
    h_pl = scale * (dt / tmin) ** (-n)
    return h_pl
@njit(cache=False)
def _powerlaw_kernel_integral_normed(u0, u1, tmin, n):
    """
    Integral of the normalized power-law kernel over [u0, u1].

    uses: h(u) = (n - 1)/tmin * (u/tmin)^(-n), for u > tmin and n < 1.0.
    """

    if (n <= 1.0) or (u1 <= u0) or (u1 <= tmin):
        return 0.0

    lo = u0
    hi = u1

    if lo < tmin:
        lo = tmin

    if hi <= lo:
        return 0.0

    val = _cdf_scalar(hi, tmin, n) - _cdf_scalar(lo, tmin, n)

    if val < 0.0:
        return 0.0

    return val

@njit(cache=False)
def _exp_kernel_value_normed(dt, tau, tmin):
    """
    Creates a single value of the normalized exponential kernel at a time dt after tmin

    uses: h(u) = (1/tau)*exp[-(dt-tmin)/tau)] for dt > tmin and tau > 0.
    """
    if (tau <= 0.0) or (dt <= tmin):
        return 0.0

    return (1.0 / tau) * np.exp(-(dt - tmin) / tau)

@njit(cache=False)
def _exp_kernel_integral_normed(u0, u1, tau, tmin):
    """
    Integral of the normalized exponential kernel over the interval [u0, u1].

    uses: h(u) = (1/tau)*exp[-(dt-tmin)/tau)] for u > tmin and tau > 0.
    """
    if (tau <= 0.0) or (u1 <= u0) or (u1 <= tmin):
        return 0.0

    lo = u0
    hi = u1

    if lo < tmin:
        lo = tmin

    if hi <= lo:
        return 0.0

    val = _exp_cdf_scalar(hi, tau, tmin) - _exp_cdf_scalar(lo, tau, tmin)

    if val < 0.0:
        return 0.0

    return val


@njit(cache=False)
def _additive_kernel_value_normed(dt, n, tau, f_exp, tmin):
    """
    Creates value from the normalized additive kernel at the time dt after tmin.

    uses: h(u) = f_exp * h_exp(u) + (1 - f_exp) * h_pl(u) where f_exp in [0, 1] and h_exp, h_pl are normalized.
    """
    if f_exp < 0.0 or f_exp > 1.0:
        return 0.0

    h_exp = _exp_kernel_value_normed(dt, tau, tmin)
    h_pl = _powerlaw_kernel_value_normed(dt, n, tmin)

    return f_exp * h_exp + (1.0 - f_exp) * h_pl


@njit(cache=False)
def _additive_kernel_integral_normed(u0, u1, n, tau, f_exp, tmin):
    """
    Integral of the normalized additive kernel over the interval [u0, u1].

    uses: h(u) = f_exp * h_exp(u) + (1 - f_exp) * h_pl(u) where f_exp in [0, 1] and h_exp, h_pl are normalized.
    """
    if f_exp < 0.0 or f_exp > 1.0:
        return 0.0

    exp_int = _exp_kernel_integral_normed(u0, u1, tau, tmin)
    pl_int = _powerlaw_kernel_integral_normed(u0, u1, tmin, n)

    return f_exp * exp_int + (1.0 - f_exp) * pl_int

#-----------------------------------------------------------------------------------------------------------------------
#all of this here basically just exists to ensure that when we make dead zones for S1's or pS2's that we don't double count them
#so first we compute a set that contains all the dead intervals, time which we don't want to include in the integrals or pointwise rate
#then from there we build the live intervals, which we do want to use.
#but some of those live intervals will overlap, to account for that, we merge the intervals so any overlapping intervals become a single one

def build_dead_intervals(window_start_ms, window_stop_ms, s2_t_sorted, s1_sorted, tmin):
    """returns the dead intervals, i.e. the areas which are too close to pS2's or S1's"""
    dead_intervals = []

    for t_s2 in s2_t_sorted:
        a = max(t_s2, window_start_ms)
        b = min(t_s2 + tmin, window_stop_ms)
        if b > a:
            dead_intervals.append((a, b))

    for t_s1 in s1_sorted:
        a = max(t_s1, window_start_ms)
        b = min(t_s1 + 4.6, window_stop_ms)
        if b > a:
            dead_intervals.append((a, b))

    return dead_intervals

def merge_intervals(intervals):
    """
    Merge overlapping livetime intervals.

    intervals: list of (start, stop)
    returns: list of non-overlapping (start, stop)
    """
    if len(intervals) == 0:
        return []

    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]

    for start, stop in intervals[1:]:
        last_start, last_stop = merged[-1]

        if start <= last_stop:
            merged[-1] = (last_start, max(last_stop, stop))
        else:
            merged.append((start, stop))

    return merged

def build_live_intervals(window_start_ms, window_stop_ms, dead_intervals):
    """returns the live time intervals having removed the dead intervals, i.e. areas comfortably far enough from the last pS2 or S1"""
    dead_intervals = merge_intervals(dead_intervals)

    live_intervals = []

    cursor = window_start_ms
    for start, stop in dead_intervals:
        if start > cursor:
            live_intervals.append((cursor, start))

        cursor = max(cursor, stop)

    if cursor < window_stop_ms:
        live_intervals.append((cursor, window_stop_ms))

    return live_intervals
def make_live_mask(times_ms, live_intervals):
    times_ms = np.asarray(times_ms)
    mask = np.zeros(times_ms.size, dtype=bool)

    for a, b in live_intervals:
        mask |= (times_ms >= a) & (times_ms <= b)

    return mask
#-----------------------------------------------------------------------------------------------------------------------
#here we actually make the pdfs, to be honest im not entirely sure why we need both pdf and pdf_basic_consistent
#but thats how it was done in Conors code and I assume for a reason (probably one I just glossed over) so I'm doing it too.
#you'll notice I also have 'weak source' and 'burst source' Technically they both still use the additive model, I just
#set them up a little bit weirdly mostly to ensure that they were definitely correct.
@njit(cache=False, parallel=True, fastmath=True)
def _weak_source_like_pdf_basic_consistent(t_grid, q, n, tau, f_exp, tmin,
                                       source_t_sorted, source_ne_sorted,
                                       live_starts, live_stops):
    # Guard invalid parameters
    if tau <= 0.0 or n <= 1.0 or f_exp < 0.0 or f_exp > 1.0 or tmin <= 0.0:
        diff_bad = np.zeros(t_grid.size)
        return 0.0, diff_bad

    # Compute source-dependent norms
    norms = _compute_norms_source_ne(q, source_ne_sorted)

    # Pointwise rate lambda(t_i)
    diff = np.zeros(t_grid.size)

    for i in prange(t_grid.size):
        ti = t_grid[i]

        acc = 0.0

        for j in range(source_t_sorted.size):
            dt = ti - source_t_sorted[j]

            if dt > tmin:
                acc += norms[j] * _additive_kernel_value_normed(
                    dt, n, tau, f_exp, tmin
                )

        diff[i] += acc

    # Integrated expected count Lambda over live intervals
    total_rate = 0.0

    # S2-correlated contribution
    for j in range(source_t_sorted.size):
        norm_j = norms[j]

        for ell in range(live_starts.size):
            u0 = live_starts[ell] - source_t_sorted[j]
            u1 = live_stops[ell] - source_t_sorted[j]

            total_rate += norm_j * _additive_kernel_integral_normed(
                u0, u1, n, tau, f_exp, tmin
            )

    return total_rate, diff

@njit(cache=False, parallel=True, fastmath=True)
def _exp_additive_pdf_basic_consistent(t_grid, s, n, tau, f_exp, tmin,
                                       c, d, k,
                                       s2_t_sorted,
                                       s2_rec_sorted,
                                       s2_area_sorted,
                                       s2_rng_sorted,
                                       live_starts, live_stops):
    """
    Core additive exponential + power-law evaluator.

    Pointwise rate:
        lambda(t) = k + sum_p N_p h_add(t - t_p)

    Expected count:
        Lambda = k T_live
                 + sum_p N_p sum_l int_{L_l0 - t_p}^{L_l1 - t_p} h_add(u) du

    where

        h_add(u) = f_exp h_exp(u) + (1 - f_exp) h_pl(u)

    and both component kernels are normalized.
    """

    # Guard invalid parameters
    if tau <= 0.0 or n <= 1.0 or f_exp < 0.0 or f_exp > 1.0 or tmin <= 0.0:
        diff_bad = np.zeros(t_grid.size)
        return 0.0, diff_bad

    # Compute S2-dependent norms
    norms = _compute_norms_basic(s, c, d, s2_area_sorted, s2_rng_sorted)

    # Pointwise rate lambda(t_i)
    diff = np.full(t_grid.size, k)

    for i in prange(t_grid.size):
        ti = t_grid[i]

        # Dead zone -> zero rate
        # if _in_dead_s2(ti, s2_t_sorted, tmin) or _in_dead_s1(ti, s1_t_sorted):
        #     diff[i] = 0.0
        #     continue

        acc = 0.0

        for j in range(s2_t_sorted.size):
            dt = ti - s2_t_sorted[j]

            if dt > tmin:
                acc += norms[j] * _additive_kernel_value_normed(
                    dt, n, tau, f_exp, tmin
                )

        diff[i] += acc

    # Integrated expected count Lambda over live intervals
    total_rate = 0.0

    # Background contribution
    for ell in range(live_starts.size):
        dt_live = live_stops[ell] - live_starts[ell]

        if dt_live > 0.0:
            total_rate += k * dt_live

    # S2-correlated contribution
    for j in range(s2_t_sorted.size):
        norm_j = norms[j]

        for ell in range(live_starts.size):
            u0 = live_starts[ell] - s2_t_sorted[j]
            u1 = live_stops[ell] - s2_t_sorted[j]

            total_rate += norm_j * _additive_kernel_integral_normed(
                u0, u1, n, tau, f_exp, tmin
            )

    return total_rate, diff
def weak_source_like_pdf(t_grid,
        q, n, tau, f_exp, tmin,
        source_like_struct,
        live_starts,
        live_stops, model = 'extra_source'):
    source_t = source_like_struct['time_since_start'].astype(np.float64)

    # Sort Sources consistently
    order = np.argsort(source_t)
    source_t_sorted = np.ascontiguousarray(source_t[order])

    source_ne = source_like_struct["n_electron_rec"].astype(np.float64)
    source_ne_sorted = np.ascontiguousarray(source_ne[order])


    return _weak_source_like_pdf_basic_consistent(
        np.ascontiguousarray(t_grid.astype(np.float64)),
        float(q), float(n), float(tau), float(f_exp), float(tmin),
        source_t_sorted, source_ne_sorted,
        live_starts,
        live_stops
    )

def new_exp_additive_pdf(t_grid, s, n, tau, f_exp, tmin, c, d, k,
                         pS2s, s1_times_ms,
                         window_start_ms, window_stop_ms):
    """
    Additive exponential + power-law model.

    For dt > tmin:

        h_add(dt) =
            f_exp * h_exp(dt)
            + (1 - f_exp) * h_pl(dt)

    with

        h_exp(dt) = (1/tau) exp(-(dt - tmin)/tau)

    and

        h_pl(dt) = ((n - 1)/tmin) (dt/tmin)^(-n)

    Both kernels are normalized, so the S2 norm remains an expected yield.
    """

    # Extract S2 fields
    s2_t = pS2s['time_since_start'].astype(np.float64)
    s2_area = pS2s['area'].astype(np.float64)
    s2_rng = (pS2s['range_50p_area'] / 1e6).astype(np.float64)
    s2_rec = pS2s['n_electron_rec'].astype(np.float64)

    # Sort S2s consistently
    order = np.argsort(s2_t)
    s2_t_sorted = np.ascontiguousarray(s2_t[order])
    s2_area_sorted = np.ascontiguousarray(s2_area[order])
    s2_rng_sorted = np.ascontiguousarray(s2_rng[order])
    s2_rec_sorted = np.ascontiguousarray(s2_rec[order])

    # Scale area and width/range to dimensionless values
    area_ref = np.median(s2_area_sorted)
    width_ref = np.median(s2_rng_sorted)

    if area_ref <= 0:
        area_ref = 1.0

    if width_ref <= 0:
        width_ref = 1.0

    s2_area_scaled = np.ascontiguousarray(s2_area_sorted / area_ref)
    s2_rng_scaled = np.ascontiguousarray(s2_rng_sorted / width_ref)

    # Sort S1s
    if s1_times_ms is not None and len(s1_times_ms) > 0:
        s1_sorted = np.ascontiguousarray(np.sort(s1_times_ms.astype(np.float64)))
    else:
        s1_sorted = np.zeros(0, dtype=np.float64)

    #remanant of old code where this pdf function made its own dead/live intervals.
    live_starts = window_start_ms
    live_stops = window_stop_ms

    return _exp_additive_pdf_basic_consistent(
        np.ascontiguousarray(t_grid.astype(np.float64)),
        float(s), float(n), float(tau), float(f_exp), float(tmin),
        float(c), float(d), float(k),
        s2_t_sorted,
        s2_rec_sorted,
        s2_area_scaled,
        s2_rng_scaled,
        live_starts,
        live_stops
    )
@njit(cache=False, parallel=True, fastmath=True)
def _burst_source_pdf_basic_consistent(t_grid, q, n, tau,
                                       f_exp, tmin, source_t_sorted,
                                       source_ne_sorted, live_starts,
                                       live_stops ):
    # Guard invalid parameters
    if tau <= 0.0 or n <= 1.0 or f_exp < 0.0 or f_exp > 1.0 or tmin <= 0.0:
        diff_bad = np.zeros(t_grid.size)
        return 0.0, diff_bad

    # Compute source-dependent norms
    norms = _compute_norms_source_ne(q, source_ne_sorted)

    # Pointwise rate lambda(t_i)
    diff = np.zeros(t_grid.size)

    for i in prange(t_grid.size):
        ti = t_grid[i]

        acc = 0.0

        for j in range(source_t_sorted.size):
            dt = ti - source_t_sorted[j]

            if dt > tmin:
                acc += norms[j] * _additive_kernel_value_normed(
                    dt, n, tau, f_exp, tmin
                )

        diff[i] += acc

    # Integrated expected count Lambda over live intervals
    total_rate = 0.0

    # S2-correlated contribution
    for j in range(source_t_sorted.size):
        norm_j = norms[j]

        for ell in range(live_starts.size):
            u0 = live_starts[ell] - source_t_sorted[j]
            u1 = live_stops[ell] - source_t_sorted[j]

            total_rate += norm_j * _additive_kernel_integral_normed(
                u0, u1, n, tau, f_exp, tmin
            )

    return total_rate, diff
def burst_source_like_pdf(t_grid,
        q, n, tau, f_exp, tmin,
        burst_source_struct,
        live_starts,
        live_stops, model = 'extra_source'):
    source_t = burst_source_struct['time_since_start'].astype(np.float64)

    # Sort Sources consistently
    order = np.argsort(source_t)
    source_t_sorted = np.ascontiguousarray(source_t[order])

    source_ne = burst_source_struct["n_electron_rec"].astype(np.float64)
    source_ne_sorted = np.ascontiguousarray(source_ne[order])


    return _burst_source_pdf_basic_consistent(
        np.ascontiguousarray(t_grid.astype(np.float64)),
        float(q), float(n), float(tau), float(f_exp), float(tmin),
        source_t_sorted, source_ne_sorted,
        live_starts,
        live_stops
    )
#-----------------------------------------------------------------------------------------------------------------------
#this is basically just the other pdfs but put into one function, which is part of why it was easier to do all 3 seperately
#other than that there isn't much special happening here that hasn't been seen already.

def new_exp_additive_three_source_pdf(
    t_grid, s, n, tau, f_exp, tmin, c, d, q_weak, q_burst, k,
    pS2s_struct, source_like_struct, burst_source_struct, s1_times_ms, window_start_ms,
    window_stop_ms, model = 'extra_source'
):
    # Build common dead/live intervals
    pS2_time = pS2s_struct["time_since_start"].astype(np.float64)
    #actually im currently still debating whether the source terms should have a deadtime or not, so for now it's not included
    src_time = source_like_struct["time_since_start"].astype(np.float64)

    pS2_t = np.concatenate((pS2_time, src_time))
    pS2_t = np.sort(pS2_t)

    if s1_times_ms is not None and len(s1_times_ms) > 0:
        s1_sorted = np.sort(s1_times_ms.astype(np.float64))
    else:
        s1_sorted = np.zeros(0, dtype=np.float64)

    dead_intervals = build_dead_intervals(
        window_start_ms,
        window_stop_ms,
        np.sort(pS2_t),
        s1_sorted,
        tmin
    )

    live_intervals = build_live_intervals(
        window_start_ms,
        window_stop_ms,
        dead_intervals
    )

    live_starts = np.ascontiguousarray(
        np.array([x[0] for x in live_intervals], dtype=np.float64)
    )
    live_stops = np.ascontiguousarray(
        np.array([x[1] for x in live_intervals], dtype=np.float64)
    )

    live_time = np.sum(live_stops - live_starts)

    # pS2 term.
    # WARNING: new_exp_additive_pdf() has the capacity to build its own dead/live intervals.
    # if you're doing this while also allowing the source terms to have dead/live intervals,
    # then be sure to account for this, it is a functionality I have included depending on the model you choose.
    total_pS2, rate_pS2 = new_exp_additive_pdf(
        t_grid,
        s, n, tau, f_exp, tmin, c, d,0.0,
        pS2s_struct,
        s1_times_ms,
        live_starts,
        live_stops
    )

    # source-like term, using common live intervals
    total_weak, rate_weak = weak_source_like_pdf(
        t_grid,
        q_weak, n, tau, f_exp, tmin,
        source_like_struct,
        live_starts,
        live_stops
    )

    # large source term, also using common live intervals
    total_burst, rate_burst = burst_source_like_pdf(t_grid,
                                                    q_burst, n, tau, f_exp, tmin,
                                                    burst_source_struct,
                                                    live_starts,
                                                    live_stops)

    total_background = k * live_time

    total_rate = total_pS2 + total_weak + total_burst + total_background
    rate = rate_pS2 + rate_weak + rate_burst + k

    return total_rate, rate
#-----------------------------------------------------------------------------------------------------------------------
#literally just a wrapper, I have no idea why this is required but apparently Iminuit doesn't like not having this wrapper or something.
def multi_exp_additive_three_source_wrap(t, p, s2_roi, source_like_roi, burst_source_roi, s1_roi, window_start_ms, window_stop_ms):
    """
    Wrapper for extra source model.

    Expected parameter order:
        s, n, tau, f_exp, tmin, c, d, q_weak, q_burst, k
    """

    s, n, tau, f_exp, tmin, c, d, q_weak, q_burst, k = p

    return new_exp_additive_three_source_pdf(
        t,
        s, n, tau, f_exp, tmin, c, d, q_weak, q_burst, k,
        s2_roi,
        source_like_roi,
        burst_source_roi,
        s1_roi,
        window_start_ms,
        window_stop_ms,
        model = 'extra_source'
    )
#-----------------------------------------------------------------------------------------------------------------------
# This is what gets used in the cost function itself, why do we have a seperate function for this when all we are doing is returning an existing one?
# yeah good question idk either.
def to_fit_exp_additive_three_source(
    t, s, n, tau, f_exp, tmin, c, d, q_weak, q_burst, k,
    pS2_roi, source_like_roi, burst_source_roi, s1_roi,
    window_start_ms, window_stop_ms
):
    return new_exp_additive_three_source_pdf(
        t, s, n, tau, f_exp, tmin, c, d, q_weak, q_burst, k,
        pS2_roi, source_like_roi, burst_source_roi, s1_roi,
        window_start_ms, window_stop_ms
    )
#-----------------------------------------------------------------------------------------------------------------------
#This is the function that actually calculates all the stuff we want, that being the parameter values, BIC etc.
#Basically this actually tells us about our fit.
def cost_func_exp_additive_three_source(run_id, s2_roi, source_like_roi, burst_source_roi, se_roi, s1_roi,
                           seconds_range=None,
                           model= None):
    """
    Cost function for additive exponential + power-law delayed-electron fit with extra source rate.
    """
    if (model != 'exp_additive_ps2') and (model != 'exp_additive_ps2_weak') and (model != 'exp_additive_ps2_burst') and (model != 'exp_additive_ps2_burst_weak'):
        raise Exception(f"Model {model} not supported \nSupported models are: exp_additive_ps2, exp_additive_ps2_weak, exp_additive_ps2_burst, exp_additive_ps2_burst_weak")
    else:
        print(f"\nRunning the {model} cost function now")

    fdt = 2.3

    # 5*fdt seems to be the value that works the best, definitely it works better than 3*fdt
    tmin = 5 * fdt

    window_start_ms = seconds_range[0] * 1e3
    window_stop_ms = seconds_range[1] * 1e3

    se_times = se_roi['time_since_start']

    if hasattr(s1_roi, "dtype") and s1_roi.dtype.names is not None:
        s1_times = s1_roi["time_since_start"].astype(float)
    else:
        s1_times = np.asarray(s1_roi, dtype=float)
    s2_dead_time = np.concatenate((s2_roi["time_since_start"].astype(np.float64),
                                   source_like_roi["time_since_start"].astype(np.float64)))
    s2_dead_time = np.sort(s2_dead_time)

    dead_intervals = build_dead_intervals(
        window_start_ms,
        window_stop_ms,
        s2_dead_time,
        s1_times,
        tmin
    )

    live_intervals = build_live_intervals(
        window_start_ms,
        window_stop_ms,
        dead_intervals
    )

    live_mask = make_live_mask(se_times, live_intervals)
    se_times = se_times[live_mask]

    c1 = cost.ExtendedUnbinnedNLL(
        se_times,
        lambda t, s, n, tau, f_exp, tmin, c, d, q_weak, q_burst, k: to_fit_exp_additive_three_source(
            t, s, n, tau, f_exp, tmin, c, d, q_weak, q_burst, k,
            s2_roi, source_like_roi, burst_source_roi, s1_roi, window_start_ms, window_stop_ms
        )
    )
    m = Minuit(
        c1,
        s=1.5,
        n=1.43,
        tau=241,
        f_exp=0.27,
        tmin=tmin,
        c=0.7,
        d=1.5,
        q_weak= 0.01,
        q_burst= 0.09,
        k=0.01
    )

    # this is something that can be played around with, I think for example that the background k should just be 0,
    # but its good to check regardless.
    if model == 'exp_additive_ps2':
        m.values['q_weak'] = 0.0
        m.values['q_burst'] = 0.0
        m.fixed['q_weak'] = True
        m.fixed['q_burst'] = True
    elif model == 'exp_additive_ps2_weak':
        m.values['q_burst'] = 0.0
        m.fixed['q_burst'] = True
    elif model == 'exp_additive_ps2_burst':
        m.values['q_weak'] = 0.0
        m.fixed['q_weak'] = True

    m.limits['s'] = (0, None)
    m.limits['n'] = (1.2, 5.0)
    m.limits['tau'] = (0.2, 600.0)
    m.limits['f_exp'] = (0.1, 1.0)
    m.limits['c'] = (0.0, 5.0)
    m.limits['d'] = (-5.0, 5.0)
    m.limits['q_weak'] = (0.0, None)
    m.limits['q_burst'] = (0.0, None)
    m.limits['k'] = (0.0, 10.0)

    m.fixed['n'] = False
    m.fixed['tau'] = False
    m.fixed['f_exp'] = False
    m.fixed['c'] = False
    m.fixed['d'] = False
    m.fixed['k'] = False
    m.fixed['tmin'] = True

    def run_minimization(m, strategy=1, retries=0):
        m.strategy = strategy
        m.migrad(ncall=3000)

        if (not m.valid) and retries < 3:
            print(f"Minimization failed, retry #{retries + 1} with adjusted parameters")

            if retries == 0:
                m.values['f_exp'] = 0.05
                m.values['tau'] = 5.0
            elif retries == 1:
                m.values['f_exp'] = 0.5
                m.values['tau'] = 20.0
            elif retries == 2:
                m.values['s'] = 0.1
                strategy = 2

            return run_minimization(m, strategy=strategy, retries=retries + 1)

        return m

    start_3 = time.time()
    m = run_minimization(m)
    print(f"minimization takes {(time.time() - start_3):.4f} s")

    n_obs = len(se_times)
    n_free = sum(not m.fixed[p] for p in m.parameters)

    BIC = m.fval + (np.log(n_obs) * n_free)

    print(f"Minimisation Status: \n{m.fmin}")

    values, errors = m.values, m.errors

    fit_params = ['s', 'n', 'tau', 'f_exp', 'tmin', 'c', 'd', 'q_weak', 'q_burst', 'k']

    results_df = pd.DataFrame({
        "Parameter": fit_params,
        "Value": [values[p] for p in fit_params],
        "Error": [errors[p] for p in fit_params],
    })

    print("Fitted Parameters and Errors:")
    print(results_df)


    print(f"\nThe amount of single electrons used in the fit is: {len(se_times)}")
    print(f"The amount of single electrons before live-mask is: {len(se_roi)}")

    return values, errors, m.covariance, BIC


def time_fitting(run_id, s2s, source_like, burst_source, ses, s1_times=None, vetos=None, seconds_range=None,
                 time_range=None, history = 10.0, model = None):
    """
    This is the main fitting function. Put in the time range you want to fit over,
    it will return the minimised values etc., you can then put those into the cdf_plot function.

    Inputs:
    - run_id: run ID for the data, contains the name, metadata, etc.
    - s2s: structured array of primary S2s
    - ses: structured array of single/delayed electron signals
    - S1_times: array of S1 times (time since start of run in ms)
    - seconds_range: time since start of the run in seconds where you want to fit/plot over
    - time_range: time since epoch in nanoseconds where you want to fit/plot over

    Outputs:
    - values: the minimised values from the cost function: n, s, k etc.
    - covariance: the covariance matrix from the minimisation - honestly forget why I return this now
    - total_rate: the total expected rate from the model over the time range
    - differential_rate: the differential rate from the model over the time range
    - BIC: Bayesian Information Criterion value for the fit
            -- a measure of model quality, though only relevant when in comparison to others
    """
    print("\n" + "-" * 120 + "\n")  # Visual separator in the output

    run_start = run_id['start'].value

    print(f"Running model: {model}")
    model_list_1 = ['exp_additive_ps2', 'exp_additive_ps2_weak', 'exp_additive_ps2_burst', 'exp_additive_ps2_burst_weak']
    model_list_2 = [ "plaw_exp_ps2","plaw_exp_ps2_weak","plaw_exp_ps2_burst","plaw_exp_ps2_burst_weak"]

    # Some error-handling
    if (time_range is not None) and (seconds_range is not None):
        raise ValueError("Idiot error. Provide one or the other, not both")
    elif (time_range is None) and (seconds_range is None):
        raise ValueError("Idiot error. You need to provide one of time_range or seconds_range")

    if vetos is None:
        print("Running without DAQ vetos; cannot guarantee a clean fit.")  # This is a super minor effect though

    if s1_times is None:
        s1_times = np.empty(0, dtype=np.float64)
        print("Running without normalisation from S1 dead zones; some loss of accuracy expected.")

    fit_start_ms = seconds_range[0] * 1e3
    fit_stop_ms = seconds_range[1] * 1e3

    source_start_ms = max(0.0, fit_start_ms - history * 1e3)
    source_stop_ms = fit_stop_ms

    if seconds_range is not None:
        start_ms, end_ms = seconds_range[0] * int(1e3), seconds_range[1] * int(1e3)

        s2_region = s2s[(s2s['time_since_start'] >= source_start_ms) & (s2s['time_since_start'] <= source_stop_ms)]
        se_region = ses[(ses['time_since_start'] >= fit_start_ms) & (ses['time_since_start'] <= fit_stop_ms)]
        source_like_region = source_like[
            (source_like["time_since_start"] >= source_start_ms) & (source_like["time_since_start"] <= source_stop_ms)]
        burst_source_region = burst_source[
            (burst_source["time_since_start"] >= source_start_ms) & (burst_source["time_since_start"] <= source_stop_ms)]

        if s1_times is not None and len(s1_times) > 0:
            S1_region = s1_times[(s1_times >= start_ms) & (s1_times <= end_ms)]
        else:
            S1_region = np.empty(0, dtype=np.float64)

    elif time_range is not None:
        print("I think time_range stuff works, I've mostly just shoved it in to seconds_range, so just beware")
        start_ns, end_ns = time_range[0], time_range[1]
        start_ms, end_ms = (start_ns - run_start) / 1e6, (end_ns - run_start) / 1e6
        seconds_range = (start_ms / 1e3, end_ms / 1e3)

        s2_region = s2s[(s2s['time'] >= start_ns) & (s2s['time'] <= end_ns)]
        se_region = ses[(ses['time'] >= start_ns) & (ses['time'] <= end_ns)]

        if s1_times is not None and len(s1_times) > 0:
            S1_region = s1_times[(s1_times >= start_ms) & (s1_times <= end_ms)]
        else:
            S1_region = np.empty(0, dtype=np.float64)

    if model in model_list_2:
        values, errors, covariance, BIC = cost_func_plaw_exp_three_source(run_id, s2_region, source_like_region,
                                                                          burst_source_region, se_region,
                                                                          S1_region, seconds_range=(fit_start_ms/1e3, fit_stop_ms/1e3),
                                                                          model = model)
    elif model in model_list_1:
        values, errors, covariance, BIC = cost_func_exp_additive_three_source(run_id, s2_region, source_like_region,
                                                                          burst_source_region,
                                                                          se_region, S1_region,
                                                                          seconds_range=(fit_start_ms/1e3, fit_stop_ms/1e3),
                                                                          model=model)
    if model == 'exp_additive_ps2':
        print(f"These results are for the ps2 only source model, neither the weak nor burst component are included in this fit.")
        results_df = pd.DataFrame(
            {"Parameter": ['Run ID', 'Start Time (s)', 'End Time (s)', 's', 's_err', 'n', 'n_err', 'tau', 'tau_err',
                           'f_exp', 'f_exp_err', 'tmin', 'tmin_err', 'c', 'c_err', 'd', 'd_err',
                           'q_weak', 'q_weak_err', 'q_burst', 'q_burst_err', 'k', 'k_err', 'Num pS2s', 'Num SEs',
                           'BIC'],
             "Value": [run_id['name'], start_ms / 1e3, end_ms / 1e3, values[0], errors[0], values[1], errors[1],
                       values[2], errors[2], values[3], errors[3], values[4], errors[4],
                       values[5], errors[5], values[6], errors[6],  0, 0, 0, 0, values[9],
                       errors[9], len(s2_region), len(se_region),
                       BIC]})
    elif model == 'exp_additive_ps2_weak':
        print(f"These results are for the ps2 + weak source model, the burst component is not included in this fit.")
        results_df = pd.DataFrame(
            {"Parameter": ['Run ID', 'Start Time (s)', 'End Time (s)', 's', 's_err', 'n', 'n_err', 'tau', 'tau_err',
                           'f_exp', 'f_exp_err', 'tmin', 'tmin_err', 'c', 'c_err', 'd', 'd_err',
                           'q_weak', 'q_weak_err','q_burst', 'q_burst_err', 'k', 'k_err', 'Num pS2s', 'Num SEs', 'BIC'],
             "Value": [run_id['name'], start_ms / 1e3, end_ms / 1e3, values[0], errors[0], values[1], errors[1],
                       values[2], errors[2], values[3], errors[3], values[4], errors[4],
                       values[5], errors[5], values[6], errors[6], values[7], errors[7], 0, 0,
                       values[9], errors[9], len(s2_region), len(se_region),BIC]})
    elif model == 'exp_additive_ps2_burst':
        print(f"These results are for the ps2 + burst source model, the weak component is not included in this fit.")
        results_df = pd.DataFrame(
            {"Parameter": ['Run ID', 'Start Time (s)', 'End Time (s)', 's', 's_err', 'n', 'n_err', 'tau', 'tau_err',
                           'f_exp', 'f_exp_err', 'tmin', 'tmin_err', 'c', 'c_err', 'd', 'd_err',
                           'q_weak', 'q_weak_err', 'q_burst', 'q_burst_err', 'k', 'k_err', 'Num pS2s', 'Num SEs',
                           'BIC'],
             "Value": [run_id['name'], start_ms / 1e3, end_ms / 1e3, values[0], errors[0], values[1], errors[1],
                       values[2], errors[2], values[3], errors[3], values[4], errors[4],
                       values[5], errors[5], values[6], errors[6], 0, 0, values[8],
                       errors[8], values[9], errors[9], len(s2_region), len(se_region),
                       BIC]})
    elif model == 'exp_additive_ps2_burst_weak':
        print(f"These results are for the ps2 + weak + burst source model, all components are included in this fit.")
        results_df = pd.DataFrame(
            {"Parameter": ['Run ID', 'Start Time (s)', 'End Time (s)', 's', 's_err', 'n', 'n_err', 'tau', 'tau_err',
                           'f_exp', 'f_exp_err', 'tmin', 'tmin_err', 'c', 'c_err', 'd', 'd_err',
                           'q_weak', 'q_weak_err', 'q_burst', 'q_burst_err', 'k', 'k_err', 'Num pS2s', 'Num SEs',
                           'BIC'],
             "Value": [run_id['name'], start_ms / 1e3, end_ms / 1e3, values[0], errors[0], values[1], errors[1],
                       values[2], errors[2], values[3], errors[3], values[4], errors[4],
                       values[5], errors[5], values[6], errors[6], values[7], errors[7], values[8], errors[8], values[9],
                       errors[9], len(s2_region), len(se_region),
                       BIC]})
    if model == 'plaw_exp_ps2':
        print(
            "These results are for the plaw*exp ps2 only source model, neither the weak nor burst component are included in this fit.")

        results_df = pd.DataFrame(
            {
                "Parameter": [
                    'Run ID', 'Start Time (s)', 'End Time (s)',
                    's', 's_err',
                    'n', 'n_err',
                    'tau', 'tau_err',
                    'tmin', 'tmin_err',
                    'c', 'c_err',
                    'd', 'd_err',
                    'q_weak', 'q_weak_err',
                    'q_burst', 'q_burst_err',
                    'k', 'k_err',
                    'Num pS2s', 'Num SEs',
                    'BIC'
                ],
                "Value": [
                    run_id['name'], start_ms / 1e3, end_ms / 1e3,
                    values[0], errors[0],
                    values[1], errors[1],
                    values[2], errors[2],
                    values[3], errors[3],
                    values[4], errors[4],
                    values[5], errors[5],
                    0, 0,
                    0, 0,
                    values[8], errors[8],
                    len(s2_region), len(se_region),
                    BIC
                ]
            }
        )

    elif model == 'plaw_exp_ps2_weak':
        print(
            "These results are for the plaw*exp ps2 + weak source model, the burst component is not included in this fit.")

        results_df = pd.DataFrame(
            {
                "Parameter": [
                    'Run ID', 'Start Time (s)', 'End Time (s)','s', 's_err','n', 'n_err',
                    'tau', 'tau_err', 'tmin', 'tmin_err','c', 'c_err','d', 'd_err',
                    'q_weak', 'q_weak_err','q_burst', 'q_burst_err','k', 'k_err',
                    'Num pS2s', 'Num SEs','BIC'
                ],
                "Value": [
                    run_id['name'], start_ms / 1e3, end_ms / 1e3,
                    values[0], errors[0],values[1], errors[1],values[2], errors[2], values[3], errors[3],
                    values[4], errors[4],values[5], errors[5],values[6], errors[6],0, 0,
                    values[8], errors[8],len(s2_region), len(se_region),BIC
                ]
            }
        )

    elif model == 'plaw_exp_ps2_burst':
        print(
            "These results are for the plaw*exp ps2 + burst source model, the weak component is not included in this fit.")

        results_df = pd.DataFrame(
            {
                "Parameter": [
                    'Run ID', 'Start Time (s)', 'End Time (s)',
                    's', 's_err',
                    'n', 'n_err',
                    'tau', 'tau_err',
                    'tmin', 'tmin_err',
                    'c', 'c_err',
                    'd', 'd_err',
                    'q_weak', 'q_weak_err',
                    'q_burst', 'q_burst_err',
                    'k', 'k_err',
                    'Num pS2s', 'Num SEs',
                    'BIC'
                ],
                "Value": [
                    run_id['name'], start_ms / 1e3, end_ms / 1e3,
                    values[0], errors[0],
                    values[1], errors[1],
                    values[2], errors[2],
                    values[3], errors[3],
                    values[4], errors[4],
                    values[5], errors[5],
                    0, 0,
                    values[7], errors[7],
                    values[8], errors[8],
                    len(s2_region), len(se_region),
                    BIC
                ]
            }
        )

    elif model == 'plaw_exp_ps2_burst_weak':
        print(
            "These results are for the plaw*exp ps2 + weak + burst source model, all components are included in this fit.")

        results_df = pd.DataFrame(
            {
                "Parameter": [
                    'Run ID', 'Start Time (s)', 'End Time (s)',
                    's', 's_err',
                    'n', 'n_err',
                    'tau', 'tau_err',
                    'tmin', 'tmin_err',
                    'c', 'c_err',
                    'd', 'd_err',
                    'q_weak', 'q_weak_err',
                    'q_burst', 'q_burst_err',
                    'k', 'k_err',
                    'Num pS2s', 'Num SEs',
                    'BIC'
                ],
                "Value": [
                    run_id['name'], start_ms / 1e3, end_ms / 1e3,
                    values[0], errors[0],
                    values[1], errors[1],
                    values[2], errors[2],
                    values[3], errors[3],
                    values[4], errors[4],
                    values[5], errors[5],
                    values[6], errors[6],
                    values[7], errors[7],
                    values[8], errors[8],
                    len(s2_region), len(se_region),
                    BIC
                ]
            }
        )



    t = np.arange(se_region['time_since_start'][0], se_region['time_since_start'][-1], 0.5)

    window_start_ms = seconds_range[0] * 1e3
    window_stop_ms = seconds_range[1] * 1e3

    if model == 'exp_additive_ps2':
        total_rate, differential_rate = new_exp_additive_three_source_pdf(t, values[0], values[1], values[2], values[3],
                                                                          values[4],
                                                                          values[5], values[6], 0.0, 0.0,
                                                                          values[9],
                                                                          s2_region, source_like_region,
                                                                          burst_source_region,
                                                                          S1_region, window_start_ms, window_stop_ms,
                                                                          model=model)
    elif model == 'exp_additive_ps2_weak':
        total_rate, differential_rate = new_exp_additive_three_source_pdf(t, values[0], values[1], values[2], values[3],
                                                                          values[4],
                                                                          values[5], values[6], values[7],
                                                                          0.0, values[9],
                                                                          s2_region, source_like_region,
                                                                          burst_source_region,
                                                                          S1_region, window_start_ms, window_stop_ms,
                                                                          model=model
        )
    elif model == 'exp_additive_ps2_burst':
        total_rate, differential_rate = new_exp_additive_three_source_pdf(t, values[0], values[1], values[2], values[3],
                                                                          values[4],
                                                                          values[5], values[6],0.0,
                                                                          values[8], values[9],
                                                                          s2_region, source_like_region,
                                                                          burst_source_region,
                                                                          S1_region, window_start_ms, window_stop_ms,
                                                                          model=model
        )
    elif model == 'exp_additive_ps2_burst_weak':
        total_rate, differential_rate = new_exp_additive_three_source_pdf(t, values[0], values[1], values[2], values[3],
                                                                          values[4],
                                                                          values[5], values[6], values[7], values[8],
                                                                          values[9],
                                                                          s2_region, source_like_region,
                                                                          burst_source_region,
                                                                          S1_region, window_start_ms, window_stop_ms,
                                                                          model=model
        )
    if model == 'plaw_exp_ps2':
        total_rate, differential_rate = new_plaw_exp_three_source_pdf(t, values[0], values[1], values[2],
                                                                      values[3],
                                                                      values[4], values[5], 0.0, 0.0,
                                                                      values[8],
                                                                      s2_region, source_like_region,
                                                                      burst_source_region,
                                                                      S1_region, window_start_ms, window_stop_ms,
                                                                      model=model)

    elif model == 'plaw_exp_ps2_weak':
        total_rate, differential_rate = new_plaw_exp_three_source_pdf(t, values[0], values[1], values[2],
                                                                      values[3],
                                                                      values[4], values[5], values[6],
                                                                      0.0, values[8],
                                                                      s2_region, source_like_region,
                                                                      burst_source_region,
                                                                      S1_region, window_start_ms, window_stop_ms,
                                                                      model=model
                                                                      )

    elif model == 'plaw_exp_ps2_burst':
        total_rate, differential_rate = new_plaw_exp_three_source_pdf(t, values[0], values[1], values[2],
                                                                      values[3],
                                                                      values[4], values[5], 0.0,
                                                                      values[7], values[8],
                                                                      s2_region, source_like_region,
                                                                      burst_source_region,
                                                                      S1_region, window_start_ms, window_stop_ms,
                                                                      model=model
                                                                      )

    elif model == 'plaw_exp_ps2_burst_weak':
        total_rate, differential_rate = new_plaw_exp_three_source_pdf(t, values[0], values[1], values[2],
                                                                      values[3],
                                                                      values[4], values[5], values[6], values[7],
                                                                      values[8],
                                                                      s2_region, source_like_region,
                                                                      burst_source_region,
                                                                      S1_region, window_start_ms, window_stop_ms,
                                                                      model=model
                                                                      )
    return results_df, values, covariance, total_rate, differential_rate

#-----------------------------------------------------------------------------------------------------------------------
def cdf_plot(s2_roi, se_roi, source_like_roi, burst_source_roi, s1_roi, values, cov, model = None,
             seconds_range=None, ax=None, plot_zoom=(0, 0),
             label=None, color='r', extra_models=None, show_model_errors=False, show=True):
    """
    Main function to plot the cdf.

    The actual cdf is not particularly interesting, but does point to whether or not
    miniuit has done well, or totally messed up somehow (aside from the values it returns itself)

    More importantly can also point to quality of signal selection, such as if large S2s are not being picked up

    Inputs:
    - s2_roi: "region of interest" for the S2 peaks, determined by time_fitting function
    - se_roi: "" for se peaks ""
    - values: minimised values from the cost function
    - cov: covariance matrix from the cost function
    - model: which model was used in the fitting (try to match please otherwise idk what will happen - weird fits probably)
    - seconds_range: range of seconds since start of run
    - ax: deprecated, should remove but shan't
    - plot_zoom: (start_offset, width) in s to zoom in on a specific region of the plot - can only use this if plot=False in time_fitting function ofc

    Outputs:
    - Main fit plots we're interested in (red line + histogram stuff)
    """

    resolution_ms = 10  # histogram bin size

    if seconds_range is None:
        raise ValueError("seconds_range must be provided")

    window_start_ms = seconds_range[0] * 1e3
    window_stop_ms = seconds_range[1] * 1e3
    window_width_ms = window_stop_ms - window_start_ms

    # Getting the zoom parameters on the CDF plot was annoying
    if plot_zoom != (0, 0):
        zoom_start_rel_ms = plot_zoom[0] * 1e3  # relative to window start
        zoom_width_ms = plot_zoom[1] * 1e3
        zoom_end_rel_ms = zoom_start_rel_ms + zoom_width_ms

        zoom_end_rel_ms = min(zoom_end_rel_ms, window_width_ms)

        time_start_ms = window_start_ms + zoom_start_rel_ms
        time_stop_ms = window_start_ms + zoom_end_rel_ms

        plot_shift_ms = window_start_ms + zoom_start_rel_ms
        x_axis_left = 0
        x_axis_right = zoom_end_rel_ms - zoom_start_rel_ms  # = width_ms

    else:
        # No zoom = show whole window, shifted to 0
        zoom_start_rel_ms = 0
        time_start_ms = window_start_ms
        time_stop_ms = window_stop_ms
        plot_shift_ms = window_start_ms
        x_axis_left = 0
        x_axis_right = window_width_ms

    n_bins = max(1, int((time_stop_ms - time_start_ms) / resolution_ms))


    se_times = se_roi['time_since_start']

    mask_zoom = (se_times >= time_start_ms) & (se_times <= time_stop_ms)
    se_times_zoom = se_times[mask_zoom]

    se_times_plot = se_times_zoom - plot_shift_ms
    bin_edges_power_sum = np.linspace(
        x_axis_left,
        x_axis_right,
        n_bins + 1
    )
    hist_power_sum, bin_edges_power_sum = np.histogram(
        se_times_plot,
        bins=bin_edges_power_sum
    )
    bin_centers_plot = (bin_edges_power_sum[:-1] + bin_edges_power_sum[1:]) / 2
    bin_width = bin_edges_power_sum[1] - bin_edges_power_sum[0]
    t_model = bin_centers_plot + plot_shift_ms  # convert back to absolute ms

    wrap = model_wrap_for_name(model)

    model_rate, model_errors = propagate(
        lambda p: wrap(
            t_model, p,
            s2_roi, source_like_roi, burst_source_roi,
            s1_roi, window_start_ms, window_stop_ms,
            model=model
        )[1],
        values,
        cov
    )
    model_errors_prop = np.diag(model_errors) ** 0.5
    t_abs = np.arange(time_start_ms, time_stop_ms, 0.5)

    total_rate, p, rate_at_events = evaluate_model_for_plot(model, values, t_abs, s1_roi, s2_roi, source_like_roi,
                                                            burst_source_roi, se_times_zoom, window_start_ms, window_stop_ms)

    t_plot = t_abs - plot_shift_ms  # plot in relative zoom coordinates

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(15, 5))
        new_ax = True
    else:
        new_ax = False

    weights = np.ones_like(se_times_plot) / bin_width

    ax.hist(se_times_plot, bins=bin_edges_power_sum, weights=weights, color='k',
            histtype='step', label='Observed SE rate')

    # ax.errorbar(bin_centers_plot, hist_power_sum,
    #             yerr=np.sqrt(hist_power_sum), fmt='+', color='orange', markersize=0.4)

    for s2_time in s2_roi['time_since_start']:
        start = s2_time - plot_shift_ms
        stop = s2_time + 5 * 2.3 - plot_shift_ms

        if stop >= x_axis_left and start <= x_axis_right:
            ax.axvspan(
                max(start, x_axis_left),
                min(stop, x_axis_right),
                color='green',
                alpha=0.08,
                label='S2 dead zone' if 'S2 dead zone' not in ax.get_legend_handles_labels()[1] else ""
            )
    for s1_time in s1_roi:
        start = s1_time - plot_shift_ms
        stop = s1_time + 4.6 - plot_shift_ms

        if stop >= x_axis_left and start <= x_axis_right:
            ax.axvspan(
                max(start, x_axis_left),
                min(stop, x_axis_right),
                color='purple',
                alpha=0.10,
                label='S1 dead zone' if 'S1 dead zone' not in ax.get_legend_handles_labels()[1] else ""
            )

    p_plot = mask_dead_zones(p, t_abs, s1_roi, s2_roi)

    if label is None:
        label = f'{model} fit'

    if show_model_errors:
        ax.errorbar(
            bin_centers_plot,
            model_rate * bin_width,
            yerr=model_errors_prop * bin_width,
            fmt='none',
            ecolor='b',
            alpha=1.0,
            elinewidth=1.5,
            capsize=2,
            label="model uncertainty"
        )


    ax.plot(t_plot, p_plot, color='r', label=label)
    ax.vlines(
        se_times_plot,
        ymin=0,
        ymax=0.05 * np.nanmax(p_plot),
        color="k",
        alpha=0.5,
        label="Observed SE times"
    )
    # all the extra models stuff was added by me
    if extra_models is not None:
        for extra in extra_models:
            extra_model = extra["model"]
            extra_values = extra["values"]

            extra_label = extra.get("label", f"{extra_model} fit")
            extra_color = extra.get("color", None)
            extra_linestyle = extra.get("linestyle", "-")

            _, p_extra, _ = evaluate_model_for_plot(extra_model, extra_values, t_abs, s1_roi, s2_roi, source_like_roi, burst_source_roi,
                                                    se_times_zoom, window_start_ms, window_stop_ms)
            p_extra_plot = mask_dead_zones(p_extra, t_abs, s1_roi, s2_roi)

            ax.plot(
                t_plot,
                p_extra_plot,
                color=extra_color,
                linestyle=extra_linestyle,
                label=extra_label
            )
    ax.set_xlim(0, x_axis_right)
    ax.set_xlabel("Time since window start (ms)", fontsize=14)
    ax.set_ylabel("[SE/ms]", fontsize=14)
    ax.legend(fontsize='medium', loc='best')
    ax.ticklabel_format(useOffset=False, style='plain', axis='x')

    if new_ax:
        plt.show()

    return total_rate, p, t_abs, rate_at_events

def evaluate_model_for_plot(model_name, vals, t_abs, s1_roi, s2_roi, source_like_roi, burst_source_roi,
                            se_times_zoom, window_start_ms, window_stop_ms):

    wrap = model_wrap_for_name(model_name)

    total_rate, p = wrap(
        t_abs, vals,
        s2_roi, source_like_roi, burst_source_roi,
        s1_roi, window_start_ms, window_stop_ms,
        model=model_name
    )

    _, rate_at_events = wrap(
        se_times_zoom, vals,
        s2_roi, source_like_roi, burst_source_roi,
        s1_roi, window_start_ms, window_stop_ms,
        model=model_name
    )

    return total_rate, p, rate_at_events
def mask_dead_zones(p_in, t_abs, s1_roi, s2_roi):
    p_out = p_in.copy()

    for s1_time in s1_roi:
        dead = (t_abs >= s1_time) & (t_abs <= s1_time + 4.6)
        p_out[dead] = np.nan

    for s2_time in s2_roi["time_since_start"]:
        dead = (t_abs >= s2_time) & (t_abs <= s2_time + 5 * 2.3)
        p_out[dead] = np.nan

    return p_out

@njit(cache=False)
def _plaw_exp_kernel_raw(u, n, tau, tmin):
    """
    Unnormalized power-law * exponential-cutoff kernel.

    h(u) = (u/tmin)^(-n) * exp(-(u - tmin)/tau), for u > tmin.
    """
    if u <= tmin:
        return 0.0

    return (u / tmin) ** (-n) * np.exp(-(u - tmin) / tau)


@njit(cache=False)
def _plaw_exp_kernel_integral_raw(u0, u1, n, tau, tmin):
    """
    Numerical integral of the unnormalized kernel from u0 to u1.
    Uses fixed trapezoidal integration.
    """
    if u1 <= tmin:
        return 0.0

    a = max(u0, tmin)
    b = u1

    if b <= a:
        return 0.0

    n_steps = 80
    du = (b - a) / n_steps

    total = 0.0

    prev_u = a
    prev_y = _plaw_exp_kernel_raw(prev_u, n, tau, tmin)

    for i in range(1, n_steps + 1):
        u = a + i * du
        y = _plaw_exp_kernel_raw(u, n, tau, tmin)

        total += 0.5 * (prev_y + y) * du

        prev_u = u
        prev_y = y

    return total


@njit(cache=False)
def _plaw_exp_kernel_norm(n, tau, tmin):
    """
    Approximate normalization integral from tmin to infinity.

    The upper bound is chosen so the exponential cutoff has mostly died away.
    """
    if tau <= 0.0 or n <= 0.0 or tmin <= 0.0:
        return 0.0

    upper = tmin + max(50.0 * tau, 5000.0)

    norm = _plaw_exp_kernel_integral_raw(tmin, upper, n, tau, tmin)

    if norm <= 0.0:
        return 0.0

    return norm


@njit(cache=False)
def _plaw_exp_kernel_value_normed(u, n, tau, tmin, norm):
    if norm <= 0.0:
        return 0.0

    return _plaw_exp_kernel_raw(u, n, tau, tmin) / norm


@njit(cache=False)
def _plaw_exp_kernel_integral_normed(u0, u1, n, tau, tmin, norm):
    if norm <= 0.0:
        return 0.0

    return _plaw_exp_kernel_integral_raw(u0, u1, n, tau, tmin) / norm
@njit(cache=False, parallel=True, fastmath=True)
def _plaw_exp_ps2_pdf_basic(
    t_grid,
    s, n, tau, tmin, c, d,
    s2_t_sorted,
    s2_area_scaled,
    s2_width_scaled,
    live_starts,
    live_stops
):
    """
    pS2 source term using area/width scaling and plaw*exp kernel.
    Returns:
        total_rate, differential_rate
    """
    if tau <= 0.0 or n <= 0.0 or tmin <= 0.0:
        diff_bad = np.zeros(t_grid.size)
        return 0.0, diff_bad

    kernel_norm = _plaw_exp_kernel_norm(n, tau, tmin)

    if kernel_norm <= 0.0:
        diff_bad = np.zeros(t_grid.size)
        return 0.0, diff_bad

    norms = _compute_norms_basic(
        s,
        c,
        d,
        s2_area_scaled,
        s2_width_scaled
    )

    diff = np.zeros(t_grid.size)

    for i in prange(t_grid.size):
        ti = t_grid[i]
        acc = 0.0

        for j in range(s2_t_sorted.size):
            dt = ti - s2_t_sorted[j]

            if dt > tmin:
                acc += norms[j] * _plaw_exp_kernel_value_normed(
                    dt, n, tau, tmin, kernel_norm
                )

        diff[i] = acc

    total_rate = 0.0

    for j in range(s2_t_sorted.size):
        norm_j = norms[j]

        for ell in range(live_starts.size):
            u0 = live_starts[ell] - s2_t_sorted[j]
            u1 = live_stops[ell] - s2_t_sorted[j]

            total_rate += norm_j * _plaw_exp_kernel_integral_normed(
                u0, u1, n, tau, tmin, kernel_norm
            )

    return total_rate, diff


def plaw_exp_ps2_pdf(
    t_grid,
    s, n, tau, tmin, c, d,
    pS2s_struct,
    live_starts,
    live_stops
):
    """
    Python wrapper for clean pS2 term.
    """
    s2_t = pS2s_struct["time_since_start"].astype(np.float64)
    s2_area = pS2s_struct["area"].astype(np.float64)
    s2_width = (pS2s_struct["range_50p_area"] / 1e6).astype(np.float64)

    order = np.argsort(s2_t)

    s2_t_sorted = np.ascontiguousarray(s2_t[order])
    s2_area_sorted = np.ascontiguousarray(s2_area[order])
    s2_width_sorted = np.ascontiguousarray(s2_width[order])

    if len(s2_area_sorted) == 0:
        area_ref = 1.0
    else:
        area_ref = np.median(s2_area_sorted)

    if len(s2_width_sorted) == 0:
        width_ref = 1.0
    else:
        width_ref = np.median(s2_width_sorted)

    if (not np.isfinite(area_ref)) or area_ref <= 0.0:
        area_ref = 1.0

    if (not np.isfinite(width_ref)) or width_ref <= 0.0:
        width_ref = 1.0

    s2_area_scaled = np.ascontiguousarray(s2_area_sorted / area_ref)
    s2_width_scaled = np.ascontiguousarray(s2_width_sorted / width_ref)

    return _plaw_exp_ps2_pdf_basic(
        np.ascontiguousarray(t_grid.astype(np.float64)),
        float(s), float(n), float(tau), float(tmin), float(c), float(d),
        s2_t_sorted,
        s2_area_scaled,
        s2_width_scaled,
        np.ascontiguousarray(live_starts.astype(np.float64)),
        np.ascontiguousarray(live_stops.astype(np.float64)),
    )
@njit(cache=False, parallel=True, fastmath=True)
def _plaw_exp_ne_source_pdf_basic(
    t_grid,
    q, n, tau, tmin,
    source_t_sorted,
    source_ne_sorted,
    live_starts,
    live_stops
):
    """
    Generic n_electron_rec-scaled source term using plaw*exp kernel.
    """
    if tau <= 0.0 or n <= 0.0 or tmin <= 0.0 or q < 0.0:
        diff_bad = np.zeros(t_grid.size)
        return 0.0, diff_bad

    kernel_norm = _plaw_exp_kernel_norm(n, tau, tmin)

    if kernel_norm <= 0.0:
        diff_bad = np.zeros(t_grid.size)
        return 0.0, diff_bad

    norms = _compute_norms_source_ne(q, source_ne_sorted)

    diff = np.zeros(t_grid.size)

    for i in prange(t_grid.size):
        ti = t_grid[i]
        acc = 0.0

        for j in range(source_t_sorted.size):
            dt = ti - source_t_sorted[j]

            if dt > tmin:
                acc += norms[j] * _plaw_exp_kernel_value_normed(
                    dt, n, tau, tmin, kernel_norm
                )

        diff[i] = acc

    total_rate = 0.0

    for j in range(source_t_sorted.size):
        norm_j = norms[j]

        for ell in range(live_starts.size):
            u0 = live_starts[ell] - source_t_sorted[j]
            u1 = live_stops[ell] - source_t_sorted[j]

            total_rate += norm_j * _plaw_exp_kernel_integral_normed(
                u0, u1, n, tau, tmin, kernel_norm
            )

    return total_rate, diff


def plaw_exp_ne_source_pdf(
    t_grid,
    q, n, tau, tmin,
    source_struct,
    live_starts,
    live_stops
):
    """
    Python wrapper for any n_electron_rec-scaled source catalogue.
    """
    if source_struct is None or len(source_struct) == 0:
        return 0.0, np.zeros_like(t_grid, dtype=np.float64)

    source_t = source_struct["time_since_start"].astype(np.float64)

    if "n_electron_rec" not in source_struct.dtype.names:
        raise ValueError("source_struct must contain 'n_electron_rec'")

    source_ne = source_struct["n_electron_rec"].astype(np.float64)

    order = np.argsort(source_t)

    source_t_sorted = np.ascontiguousarray(source_t[order])
    source_ne_sorted = np.ascontiguousarray(source_ne[order])

    return _plaw_exp_ne_source_pdf_basic(
        np.ascontiguousarray(t_grid.astype(np.float64)),
        float(q), float(n), float(tau), float(tmin),
        source_t_sorted,
        source_ne_sorted,
        np.ascontiguousarray(live_starts.astype(np.float64)),
        np.ascontiguousarray(live_stops.astype(np.float64)),
    )
def new_plaw_exp_three_source_pdf(
    t_grid,
    s, n, tau, tmin, c, d, q_weak, q_burst, k,
    pS2s_struct,
    weak_source_struct,
    burst_source_struct,
    s1_times_ms,
    window_start_ms,
    window_stop_ms,
    model="plaw_exp_ps2_burst_weak"
):
    """
    Combined plaw*exp model.

    Parameters
    ----------
    s : clean pS2 amplitude
    n : power-law index
    tau : exponential cutoff scale [ms]
    tmin : prompt cutoff [ms]
    c, d : area/width exponents for clean pS2s
    q_weak : amplitude for weak/ne-scaled source
    q_burst : amplitude for burst/ne-scaled source
    k : constant background rate
    """

    # Convert S1 input
    if s1_times_ms is not None and len(s1_times_ms) > 0:
        s1_sorted = np.ascontiguousarray(np.sort(np.asarray(s1_times_ms, dtype=np.float64)))
    else:
        s1_sorted = np.zeros(0, dtype=np.float64)

    # Dead/live intervals from clean pS2s + S1s.
    # If you want unclean pS2s to also define prompt dead-time,
    # you should build a separate prompt catalogue and pass that here.
    pS2_time = pS2s_struct["time_since_start"].astype(np.float64)
    src_time = weak_source_struct["time_since_start"].astype(np.float64)
    pS2_t = np.concatenate((pS2_time, src_time))

    dead_intervals = build_dead_intervals(
        window_start_ms,
        window_stop_ms,
        np.sort(pS2_t),
        s1_sorted,
        tmin
    )

    live_intervals = build_live_intervals(
        window_start_ms,
        window_stop_ms,
        dead_intervals
    )

    live_starts = np.ascontiguousarray(
        np.array([x[0] for x in live_intervals], dtype=np.float64)
    )

    live_stops = np.ascontiguousarray(
        np.array([x[1] for x in live_intervals], dtype=np.float64)
    )

    live_time = np.sum(live_stops - live_starts)

    # Model-specific switches
    if model == "plaw_exp_ps2":
        q_weak_eff = 0.0
        q_burst_eff = 0.0

    elif model == "plaw_exp_ps2_weak":
        q_weak_eff = q_weak
        q_burst_eff = 0.0

    elif model == "plaw_exp_ps2_burst":
        q_weak_eff = 0.0
        q_burst_eff = q_burst

    elif model == "plaw_exp_ps2_burst_weak":
        q_weak_eff = q_weak
        q_burst_eff = q_burst

    else:
        raise ValueError(f"Unsupported model: {model}")

    # Clean pS2 term
    total_ps2, rate_ps2 = plaw_exp_ps2_pdf(
        t_grid,
        s, n, tau, tmin, c, d,
        pS2s_struct,
        live_starts,
        live_stops
    )

    # Weak/ne-scaled term
    total_weak, rate_weak = plaw_exp_ne_source_pdf(
        t_grid,
        q_weak_eff, n, tau, tmin,
        weak_source_struct,
        live_starts,
        live_stops
    )

    # Burst/ne-scaled term
    total_burst, rate_burst = plaw_exp_ne_source_pdf(
        t_grid,
        q_burst_eff, n, tau, tmin,
        burst_source_struct,
        live_starts,
        live_stops
    )

    total_background = k * live_time

    total_rate = total_ps2 + total_weak + total_burst + total_background
    rate = rate_ps2 + rate_weak + rate_burst + k

    return total_rate, rate
def to_fit_plaw_exp_three_source(
    t,
    s, n, tau, tmin, c, d, q_weak, q_burst, k,
    pS2_roi,
    weak_source_roi,
    burst_source_roi,
    s1_roi,
    window_start_ms,
    window_stop_ms,
    model="plaw_exp_ps2_burst_weak"
):
    return new_plaw_exp_three_source_pdf(
        t,
        s, n, tau, tmin, c, d, q_weak, q_burst, k,
        pS2_roi,
        weak_source_roi,
        burst_source_roi,
        s1_roi,
        window_start_ms,
        window_stop_ms,
        model=model
    )


def multi_plaw_exp_three_source_wrap(
    t,
    p,
    pS2_roi,
    weak_source_roi,
    burst_source_roi,
    s1_roi,
    window_start_ms,
    window_stop_ms,
    model="plaw_exp_ps2_burst_weak"
):
    """
    Parameter order:
        s, n, tau, tmin, c, d, q_weak, q_burst, k
    """
    s, n, tau, tmin, c, d, q_weak, q_burst, k = p

    return new_plaw_exp_three_source_pdf(
        t,
        s, n, tau, tmin, c, d, q_weak, q_burst, k,
        pS2_roi,
        weak_source_roi,
        burst_source_roi,
        s1_roi,
        window_start_ms,
        window_stop_ms,
        model=model
    )
def cost_func_plaw_exp_three_source(
    run_id,
    s2_roi,
    weak_source_roi,
    burst_source_roi,
    se_roi,
    s1_roi,
    seconds_range=None,
    model="plaw_exp_ps2_burst_weak",
    record_results=False,
    filename="fit_results_plaw_exp.csv"
):
    """
    Extended unbinned NLL for plaw*exp model.

    Supported models:
        plaw_exp_ps2
        plaw_exp_ps2_weak
        plaw_exp_ps2_burst
        plaw_exp_ps2_burst_weak
    """

    print(f"\nRunning the {model} cost function now")

    allowed = [
        "plaw_exp_ps2",
        "plaw_exp_ps2_weak",
        "plaw_exp_ps2_burst",
        "plaw_exp_ps2_burst_weak",
    ]

    if model not in allowed:
        raise ValueError(f"Unsupported model {model!r}. Supported models are {allowed}")

    fdt = 2.3
    tmin = 5 * fdt

    window_start_ms = seconds_range[0] * 1e3
    window_stop_ms = seconds_range[1] * 1e3

    se_times = se_roi["time_since_start"]

    # Convert S1 input
    if hasattr(s1_roi, "dtype") and s1_roi.dtype.names is not None:
        s1_times = s1_roi["time_since_start"].astype(float)
    else:
        s1_times = np.asarray(s1_roi, dtype=float)

    s2_dead_time = np.concatenate((s2_roi["time_since_start"], weak_source_roi["time_since_start"]))
    s2_dead_time = np.sort(s2_dead_time)

    # Same live mask as the model
    dead_intervals = build_dead_intervals(
        window_start_ms,
        window_stop_ms,
        s2_dead_time,
        s1_times,
        tmin
    )

    live_intervals = build_live_intervals(
        window_start_ms,
        window_stop_ms,
        dead_intervals
    )

    live_mask = make_live_mask(se_times, live_intervals)
    se_times = se_times[live_mask]

    c1 = cost.ExtendedUnbinnedNLL(
        se_times,
        lambda t, s, n, tau, tmin, c, d, q_weak, q_burst, k: to_fit_plaw_exp_three_source(
            t,
            s, n, tau, tmin, c, d, q_weak, q_burst, k,
            s2_roi,
            weak_source_roi,
            burst_source_roi,
            s1_times,
            window_start_ms,
            window_stop_ms,
            model=model
        )
    )

    m = Minuit(
        c1,
        s=1.2,
        n=1.2,
        tau=250.0,
        tmin=tmin,
        c=0.8,
        d=1.4,
        q_weak=2e-4,
        q_burst=5e-3,
        k=0.001,
    )

    m.limits["s"] = (0.0, None)
    m.limits["n"] = (0.2, 5.0)
    m.limits["tau"] = (10.0, 5000.0)
    m.limits["tmin"] = (tmin, tmin)
    m.limits["c"] = (0.0, 5.0)
    m.limits["d"] = (-5.0, 5.0)
    m.limits["q_weak"] = (0.0, None)
    m.limits["q_burst"] = (0.0, None)
    m.limits["k"] = (0.0, 10.0)

    m.fixed["tmin"] = True

    if model == "plaw_exp_ps2":
        m.values["q_weak"] = 0.0
        m.values["q_burst"] = 0.0
        m.fixed["q_weak"] = True
        m.fixed["q_burst"] = True

    elif model == "plaw_exp_ps2_weak":
        m.values["q_burst"] = 0.0
        m.fixed["q_burst"] = True

    elif model == "plaw_exp_ps2_burst":
        m.values["q_weak"] = 0.0
        m.fixed["q_weak"] = True

    elif model == "plaw_exp_ps2_burst_weak":
        pass

    def run_minimization(m, strategy=1, retries=0):
        m.strategy = strategy
        m.migrad(ncall=4000)

        if (not m.valid) and retries < 3:
            print(f"Minimization failed, retry #{retries + 1} with adjusted parameters")

            if retries == 0:
                m.values["tau"] = 100.0
                m.values["n"] = 1.0

            elif retries == 1:
                m.values["tau"] = 500.0
                m.values["n"] = 1.5

            elif retries == 2:
                m.values["s"] = 0.5
                strategy = 2

            return run_minimization(m, strategy=strategy, retries=retries + 1)

        return m

    start_3 = time.time()
    m = run_minimization(m)
    print(f"minimization takes {(time.time() - start_3):.4f} s")

    n_obs = len(se_times)
    n_free = sum(not m.fixed[p] for p in m.parameters)

    BIC = m.fval + np.log(n_obs) * n_free

    print(f"Minimisation Status: \n{m.fmin}")

    values, errors = m.values, m.errors

    fit_params = [
        "s", "n", "tau", "tmin",
        "c", "d", "q_weak", "q_burst", "k"
    ]

    results_df = pd.DataFrame({
        "Parameter": fit_params,
        "Value": [values[p] for p in fit_params],
        "Error": [errors[p] for p in fit_params],
    })

    print("Fitted Parameters and Errors:")
    print(results_df)

    print(f"\nThe amount of single electrons used in the fit is: {len(se_times)}")
    print(f"The amount of single electrons before live-mask is: {len(se_roi)}")

    return values, errors, m.covariance, BIC

def model_wrap_for_name(model):
    if model in [
        "exp_additive_ps2",
        "exp_additive_ps2_weak",
        "exp_additive_ps2_burst",
        "exp_additive_ps2_burst_weak",
    ]:
        return multi_exp_additive_three_source_wrap

    elif model in [
        "plaw_exp_ps2",
        "plaw_exp_ps2_weak",
        "plaw_exp_ps2_burst",
        "plaw_exp_ps2_burst_weak",
    ]:
        return multi_plaw_exp_three_source_wrap

    else:
        raise ValueError(f"Unknown model {model!r}")