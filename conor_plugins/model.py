
"""
This file contains the main functions used to fit the single electron time distribution
to the new power law + background model. 

You can pass either 'new', 'old', 'radial', or 'count' models to the time_fitting function (see my thesis for context)

The way to use this file and its functions are as follows:
1) Use time_fitting_buffer function to do the minimisation and plot the cdf
2) Use the returned values from time_fitting_buffer to plot the cdf with cdf_plot function
    (though time_fitting_buffer can also plot the cdf if you set plot = True)

Some of this might be a bit deprecated, e.g. the results_log function. 
I originally intended to use that to save my results, and for a bit I did, but then as the model changed a fair bit it
I stopped maintaining it as a function.
"""
import numpy as np
import matplotlib.pyplot as plt
from iminuit import cost, Minuit 
from jacobi import propagate
import pandas as pd
import os
# from scipy import stats
from numba import njit, prange
import time #TODO: Remove timings before end, since not necessary for others

#Functions to be used/called externally:

#Going to use this version of modelling to figure out new radius parameter for the power law

def time_fitting(run_id, s2s, source_like, burst_source, ses, s1_times = None, vetos = None, seconds_range = None, time_range = None,
                 plot = False, model = 'new', record_results = False, filename = "fit_results.csv"):
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
    print("\n" + "-" * 120 + "\n") #Visual separator in the output

    #This will be bad probably, oh well
    global run_start 
    run_start = run_id['start'].value

    print(f"Running model: {model}")

    #Some error-handling
    if (time_range is not None) and (seconds_range is not None):
        raise ValueError("Idiot error. Provide one or the other, not both")
    elif (time_range is None) and (seconds_range is None):
        raise ValueError("Idiot error. You need to provide one of time_range or seconds_range")

    if vetos is None:
        print("Running without DAQ vetos; cannot guarantee a clean fit.") #This is a super minor effect though

    if s1_times is None:
        s1_times = np.empty(0, dtype=np.float64)
        print("Running without normalisation from S1 dead zones; some loss of accuracy expected.")

    if seconds_range is not None:
        start_ms, end_ms = seconds_range[0] * int(1e3), seconds_range[1] * int(1e3)

        s2_region = s2s[(s2s['time_since_start'] >= start_ms) & (s2s['time_since_start'] <= end_ms)]
        se_region = ses[(ses['time_since_start'] >= start_ms) & (ses['time_since_start'] <= end_ms)]
        source_like_region = source_like[(source_like["time_since_start"] >= start_ms) & (source_like["time_since_start"] <= end_ms)]
        burst_source_region = burst_source[(burst_source["time_since_start"] >= start_ms) & (burst_source["time_since_start"] <= end_ms)]

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

    #Alternatively you may have done this outside the function, but just in case
    if model == 'radial':
        se_region = se_region[se_region['r'] <= 45]  # cm

    # print("\n" + "-" * 116 + "\n")
    print(f"\nThis selection will incorporate {len(s2_region)} pS2s")

    #Make sure whole region is considered
    #This section is only really here so repeating fits is easier
    print(f"Corresponding to the seconds range of: {start_ms/1e3:.0f} to {end_ms/1e3:.0f}")

    if model == 'radial':
        values, errors, covariance, BIC = cost_func_radial(run_id, s2_region, se_region, S1_region, 
                                                seconds_range = seconds_range,
                                                record_results = record_results, filename = filename)
    elif model == 'exp':
        values, errors, covariance, BIC = cost_func_exp_powerlaw(run_id, s2_region, se_region, S1_region,
                                                    seconds_range = seconds_range, model = model,
                                                    record_results = record_results, filename = filename)
    elif model == 'exp_additive':
        values, errors, covariance, BIC = cost_func_exp_additive(run_id, s2_region, se_region, S1_region,
                                                    seconds_range = seconds_range, model = model,
                                                    record_results = record_results, filename = filename)
    elif model == 'pure_exp':
        values, errors, covariance, BIC = cost_func_pure_exp(run_id, s2_region, se_region, S1_region,
                                                             seconds_range = seconds_range, model = model,
                                                             record_results = record_results, filename = filename)
    elif model == 'extra_source':
        values, errors, covariance, BIC = cost_func_exp_additive_three_source(run_id, s2_region, source_like_region, burst_source_region,
                                                            se_region, S1_region,
                                                            seconds_range = seconds_range, model = model,
                                                            record_results = record_results, filename = filename)
    else:
        values, errors, covariance, BIC = cost_func(run_id, s2_region, se_region, S1_region,
                                                    seconds_range = seconds_range, model = model,
                                                    record_results = record_results, filename = filename)
    
    if model == 'radial':
        results_df = pd.DataFrame({
            "Parameter": ['Run ID', 'Start Time (s)', 'End Time (s)', 's', 's_err', 'n', 'n_err', 'tmin', 'tmin_err',
                        'c', 'c_err', 'd', 'd_err', 'k', 'k_err', 'A', 'A_err', 'r0', 'r_p','Num pS2s', 'Num SEs', 'BIC'],
            "Value": [run_id['name'], start_ms/1e3, end_ms/1e3, values[0], errors[0], values[1], errors[1],
                    values[2], errors[2], values[3], errors[3], values[4], errors[4],
                    values[5], errors[5], values[6], errors[6], values[7], values[8], 
                    len(s2_region), len(se_region), BIC]})

    elif model == 'exp':
        results_df = pd.DataFrame({
            "Parameter": ['Run ID', 'Start Time (s)', 'End Time (s)', 's', 's_err', 'n', 'n_err', 'tau', 'tau_err',
                          't_switch', 't_switch_err', 'tmin', 'tmin_err', 'c', 'c_err','d', 'd_err', 'k', 'k_err', 'Num pS2s', 'Num SEs', 'BIC'],
            "Value": [run_id['name'], start_ms / 1e3, end_ms / 1e3, values[0], errors[0], values[1], errors[1],
                      values[2], errors[2], values[3], errors[3], values[4], errors[4],
                      values[5], errors[5], values[6], errors[6], values[7], errors[7], len(s2_region), len(se_region), BIC]})

    elif model == 'exp_additive':
        results_df = pd.DataFrame({"Parameter": ['Run ID', 'Start Time (s)', 'End Time (s)', 's', 's_err', 'n', 'n_err', 'tau', 'tau_err',
                          'f_exp', 'f_exp_err', 'tmin', 'tmin_err', 'c', 'c_err','d', 'd_err', 'k', 'k_err', 'Num pS2s', 'Num SEs', 'BIC'],
            "Value": [run_id['name'], start_ms / 1e3, end_ms / 1e3, values[0], errors[0], values[1], errors[1],
                      values[2], errors[2], values[3], errors[3], values[4], errors[4],
                      values[5], errors[5], values[6], errors[6], values[7], errors[7], len(s2_region), len(se_region), BIC]})
    elif model == 'pure_exp':
        results_df = pd.DataFrame(
            {"Parameter": ['Run ID', 'Start Time (s)', 'End Time (s)', 's', 's_err', 'tau', 'tau_err',
                           'tmin', 'tmin_err', 'c', 'c_err', 'd', 'd_err', 'k', 'k_err',
                           'Num pS2s', 'Num SEs', 'BIC'],
             "Value": [run_id['name'], start_ms / 1e3, end_ms / 1e3, values[0], errors[0], values[1], errors[1],
                       values[2], errors[2], values[3], errors[3], values[4], errors[4],
                       values[5], errors[5], len(s2_region), len(se_region),
                       BIC]})
    elif model == 'extra_source':
        results_df = pd.DataFrame(
            {"Parameter": ['Run ID', 'Start Time (s)', 'End Time (s)', 's', 's_err', 'n', 'n_err', 'tau', 'tau_err', 'f_exp',
                           'f_exp_err','tmin', 'tmin_err', 'c', 'c_err', 'd', 'd_err', 'q_weak', 'q_weak_err',
                           'q_burst', 'q_burst_err', 'k', 'k_err','Num pS2s', 'Num SEs', 'BIC'],
             "Value": [run_id['name'], start_ms / 1e3, end_ms / 1e3, values[0], errors[0], values[1], errors[1],
                       values[2], errors[2], values[3], errors[3], values[4], errors[4],
                       values[5], errors[5], values[6], errors[6], values[7], errors[7], values[8], errors[8], values[9],
                       errors[9], len(s2_region), len(se_region),
                       BIC]})


    else:
        results_df = pd.DataFrame({
            "Parameter": ['Run ID', 'Start Time (s)', 'End Time (s)', 's', 's_err', 'n', 'n_err', 'tmin', 'tmin_err',
                        'c', 'c_err', 'd', 'd_err', 'k', 'k_err', 'Num pS2s', 'Num SEs', 'BIC'],
            "Value": [run_id['name'], start_ms/1e3, end_ms/1e3, values[0], errors[0], values[1], errors[1],
                    values[2], errors[2], values[3], errors[3], values[4], errors[4],
                    values[5], errors[5], len(s2_region), len(se_region), BIC]
        })

    if plot:
        total_rate, differential_rate, absolute_time = cdf_plot(s2_region, se_region, S1_region, values, covariance, model = model,
                                 seconds_range = seconds_range)
        return results_df, covariance, total_rate, differential_rate, BIC

    else:
        t = np.arange(se_region['time_since_start'][0], se_region['time_since_start'][-1], 0.5)
        if model == 'radial':
            A = values[6]
            r0 = values[7]
            r_p = values[8]
        elif model =='exp':
            A = None
            r0 = None
            r_p = None
            window_start_ms = seconds_range[0] * 1e3
            window_stop_ms = seconds_range[1] * 1e3
            total_rate, differential_rate = exp_power_law_pdf(t, values[0], values[1], values[2], values[3], values[4],
                                                              values[5], values[6], values[7], s2_region, S1_region,
                                                              window_start_ms, window_stop_ms, A = A, r0 = r0, r_p = r_p,
                                                              model = model)
        elif model == 'exp_additive':
            A = None
            r0 = None
            r_p = None
            window_start_ms = seconds_range[0] * 1e3
            window_stop_ms = seconds_range[1] * 1e3
            total_rate, differential_rate = new_exp_additive_pdf(t, values[0], values[1], values[2], values[3], values[4],
                                                                 values[5], values[6], values[7], s2_region, S1_region,
                                                                  window_start_ms, window_stop_ms, A = A, r0 = r0, r_p = r_p,
                                                                  model = model)
        elif model == 'pure_exp':
            A = None
            r0 = None
            r_p = None
            window_start_ms = seconds_range[0]*1e3
            window_stop_ms = seconds_range[1]*1e3
            total_rate, differential_rate = pure_exp_pdf(t, values[0], values[1], values[2], values[3], values[4], values[5],
                                                         s2_region, S1_region, window_start_ms, window_stop_ms, A = A, r0 = r0, r_p = r_p,
                                                         model = model)
        elif model == 'extra_source':
            window_start_ms = seconds_range[0]*1e3
            window_stop_ms = seconds_range[1]*1e3
            total_rate, differential_rate = new_exp_additive_three_source_pdf(t, values[0], values[1], values[2], values[3], values[4],
                                                                              values[5], values[6], values[7], values[8], values[9],
                                                                              s2_region, source_like_region, burst_source_region,
                                                                              S1_region, window_start_ms, window_stop_ms, model = model)
        else: 
            A = None
            r0 = None
            r_p = None
            window_start_ms = seconds_range[0] * 1e3
            window_stop_ms = seconds_range[1] * 1e3
            total_rate, differential_rate = new_power_law_pdf(t, values[0], values[1], values[2], values[3],
                                      values[4], values[5], s2_region, S1_region, window_start_ms, window_stop_ms, A = A, r0 = r0, r_p = r_p, model = model)
        #TODO: If you call this with plot = False and then want to plot the CDF, you need to change results_df to values, or else return it also
        #Didn't do this yet because there were too many instances of calling this time_fitting function as it stands in my notebooks, sorry
        return results_df, values, covariance, total_rate, differential_rate, BIC

#------------------------------------------------------------------------------------------------------------

def cost_func(run_id, s2_roi, se_roi, s1_roi, seconds_range = None, model = 'new', record_results = False, filename = "fit_results.csv"):
    """
    CURRENTLY DEPRECATED SOZ, JUST PUT plot=True IN TIME_FITTING FUNCTION IF YOU WANT THE CDF PLOTTED

    Mainly calculates the cost function for the region of interest (roi)
    Also prints the outputs, sends them to be recorded in the csv file

    I don't anticipate this function being used outside of the time_fitting function, 
    so see that for description of the inputs here

    Outputs:
    - values: the minimised values from the cost function: n, s, k etc.
    - m.covariance: self explanatory. Later used to calculate error propagation
    """

    print(f"\nRunning the cost function now")

    if model == 'count':
        se_times = np.repeat(se_roi['time_since_start'], se_roi['n_electron_rec'])
    else:
        se_times = se_roi['time_since_start']

    fdt = 2.3
    if model == 'old':
        tmin = fdt * 3
    else:
        tmin = fdt * 5

    window_start_ms = seconds_range[0] * 1e3
    window_stop_ms = seconds_range[1] * 1e3

    se_times = se_roi['time_since_start']

    dead_intervals = build_dead_intervals(
        window_start_ms,
        window_stop_ms,
        s2_roi['time_since_start'],
        s1_roi,
        tmin
    )

    live_intervals = build_live_intervals(
        window_start_ms,
        window_stop_ms,
        dead_intervals
    )

    live_mask = make_live_mask(se_times, live_intervals)
    se_times = se_times[live_mask]

    c1 = cost.ExtendedUnbinnedNLL(se_times, 
                                  lambda t, s, n, tmin, c, d, k: to_fit(t, s, n, tmin, c,d, k, s2_roi, s1_roi,
                                                                        window_start_ms,window_stop_ms))
    
    m = Minuit(c1, s = 0.1, n = 1.5, tmin = tmin, c = 0.5, d = 0.5, k = 0.0)
    #I usually find changing s to like 0.5 or just something small like 20e-10 can help if things are going wrong
    
    m.limits['n'] = (1.0001, 5)
    m.limits['s'] = (0, None)
    m.limits['c'] = (0, 5)   # Tighter range
    m.limits['d'] = (-5, 5)  # Much tighter range - physical values should be around -1 to 1
    m.limits['k'] = (0, 10)
    m.fixed['k'] = False
    m.fixed['tmin'] = True

    # Just around in case minimisation fails the first time
    def run_minimization(m, strategy = 1, retries = 0):
        m.strategy = strategy
        m.migrad(ncall = 3000)

        if (not m.valid) and retries < 3:  # Maximum 3 retries
            print(f"Minimization failed, retry #{retries+1} with adjusted parameters")
    
            if retries == 0:
                m.values['d'] = -1
                return run_minimization(m, strategy = 1, retries = retries + 1)
            elif retries == 1:
                m.values['s'] = 20e-10  # Adjust s to a small value
                return run_minimization(m, strategy = 1, retries = retries + 1)
            elif retries == 2:
                m.values['s'] = 0.1
                return run_minimization(m, strategy = 2, retries = retries + 1)
            else:
                print("Minimization failed after 3 retries, soz")
                m.values['d'] = 1
                m.values['s'] = 20e-10
                m.hesse() #Cause why not
                #Technically this is a cheeky 4th retry just to cover the bases, still might not work
                return run_minimization(m, strategy = 2, retries = retries + 1)
        return m
    start_3 = time.time()
    m = run_minimization(m)
    print(f"minimization takes {(time.time() - start_3):.4f} s")

    n_obs = len(se_times)
    n_free = sum(not m.fixed[p] for p in m.parameters)

    BIC = (m.fval) + (np.log(n_obs) * n_free)
    #note: m.fval = −ln(L^)
    
    print(f"Minimisation Status: \n{m.fmin}")
    #Doing this has gotten rid of the colours that normally come with the printout,
    #but as long as one understands the terms you can tell if it's worked well or not

    values, errors = m.values, m.errors

    results_df = pd.DataFrame({
        "Parameter": ['s', 'n', 'tmin', 'c', 'd', 'k'],
        "Value": [values[p] for p in ['s', 'n', 'tmin', 'c', 'd', 'k']], #, 'r0', 'scaling'
        "Error": [errors[p] for p in ['s', 'n', 'tmin', 'c', 'd', 'k']],
    })

    # Print out the results
    print("Fitted Parameters and Errors:")
    print(results_df)

    if record_results:
        #TODO: Also put something in about if there's a valid minimum?
        results_log(run_id, s2_roi, values, errors, BIC, m.fmin.has_made_posdef_covar, 
                    seconds_range = seconds_range, filename = filename) #Also record to a file
    
    print(f"\nThe amount of single electrons in the region of interest is: {len(se_roi)}")
    print
    return values, errors, m.covariance, BIC

#------------------------------------------------------------------------------------------------------------

def cost_func_radial(run_id, s2_roi, se_roi, s1_roi, seconds_range = None, record_results = False, filename = "fit_results.csv"):
    """
    Mainly calculates the cost function for the region of interest (roi)
    Also prints the outputs, sends them to be recorded in the csv file

    I don't anticipate this function being used outside of the time_fitting function, 
    so see that for description of the inputs here

    Outputs:
    - values: the minimised values from the cost function: n, s, k etc.
    - m.covariance: self explanatory. Later used to calculate error propagation
    """

    print("Running the cost function now")

    se_times = se_roi['time_since_start']
    fdt = 2.3
    tmin = fdt * 5

    c1 = cost.ExtendedUnbinnedNLL(se_times, 
                                  lambda t, s, n, tmin, c, d, k, A, r0, r_p: to_fit_radial(t, s, n, tmin, c,
                                                                                     d, k, A, r0, r_p, s2_roi, s1_roi))

    m = Minuit(c1, s = 0.1, n = 1.5, tmin = tmin, c = 0.5, d = 0.5, k = 0.01, A = 2, r0 = 45, r_p = 11)
    #I usually find changing s to like 0.5 or just something small like 20e-10 can help if things are going wrong
    
    m.limits['n'] = (1.0001, 5)
    m.limits['s'] = (0, None)
    m.limits['c'] = (0, 5)   # Tighter range
    m.limits['d'] = (-5, 5)  # Much tighter range - physical values should be around -1 to 1
    m.limits['k'] = (0, 10)
    m.limits['A'] = (0, 5)
    # m.limits['r0'] = (35, 55) # Radius of fiducial volume, cm -- might put as fixed later
    # m.limits['r_p'] = (0, None) #Radius of position-correlated

    m.fixed['tmin'] = True
    m.fixed['r0'] = True
    m.fixed['r_p'] = True
    # m.fixed['A'] = True

    # Just around in case minimisation fails the first time
    def run_minimization(m, strategy = 1, retries = 0):
        m.strategy = strategy
        m.migrad(ncall = 3000)

        if (not m.valid) and retries < 3:  # Maximum 3 retries
            print(f"Minimization failed, retry #{retries+1} with adjusted parameters")
    
            if retries == 0:
                m.values['d'] = -1
                return run_minimization(m, strategy = 1, retries = retries + 1)
            elif retries == 1:
                m.values['s'] = 20e-10  # Adjust s to a small value
                return run_minimization(m, strategy = 1, retries = retries + 1)
            elif retries == 2:
                m.values['s'] = 0.1
                return run_minimization(m, strategy = 2, retries = retries + 1)
            else:
                print("Minimization failed after 3 retries, soz")
                m.values['d'] = 1
                m.values['s'] = 20e-10
                m.hesse() #Cause why not
                #Technically this is a cheeky 4th retry just to cover the bases, still might not work
                return run_minimization(m, strategy = 2, retries = retries + 1)
        return m
    start_3 = time.time()
    m = run_minimization(m)
    print(f"minimization takes {(time.time() - start_3):.4f} s")

    n_obs = len(se_times)
    n_free = sum(not m.fixed[p] for p in m.parameters)

    BIC = (m.fval) + (np.log(n_obs) * n_free)

    # print("\n Covariance Matrix Status: \n")
    # print(f" - Positive Definite: {m.fmin.has_posdef_covar} \n")
    # print(f" - Forced Positive Definite: {m.fmin.has_made_posdef_covar}\n")
    
    print(f"Minimisation Status: \n{m.fmin}")
    #Doing this has gotten rid of the colours that normally come with the printout,
    #but as long as one understands the terms you can tell if it's worked well or not

    values, errors = m.values, m.errors

    results_df = pd.DataFrame({
        "Parameter": ['s', 'n', 'tmin', 'c', 'd', 'k', 'A', 'r0', 'r_p'],
        "Value": [values[p] for p in ['s', 'n', 'tmin', 'c', 'd', 'k', 'A', 'r0', 'r_p']],
        "Error": [errors[p] for p in ['s', 'n', 'tmin', 'c', 'd', 'k', 'A', 'r0', 'r_p']],
    })

    # Print out the results
    print("Fitted Parameters and Errors:")
    print(results_df)

    if record_results:
        #TODO: fix if I care? (I don't)
        results_log(run_id, s2_roi, values, errors, BIC, m.fmin.has_made_posdef_covar, 
                    seconds_range = seconds_range, filename = filename) #Also record to a file
    
    print(f"The amount of single electrons in the region of interest is: {len(se_roi)}")
    return values, errors, m.covariance, BIC

#------------------------------------------------------------------------------------------------------------

def cost_func_exp_powerlaw(run_id, s2_roi, se_roi, s1_roi,
                           seconds_range=None,
                           model='exp',
                           record_results=False,
                           filename="fit_results_exp_powerlaw.csv"):
    """
    Separate cost function for exponential-to-power-law delayed electron fit.

    Does not modify or interfere with cost_func.
    """

    print(f"\nRunning the exponential-to-power-law cost function now")
    window_start_ms = seconds_range[0] * 1e3
    window_stop_ms = seconds_range[1] * 1e3
    if model == 'count':
        se_times = np.repeat(se_roi['time_since_start'], se_roi['n_electron_rec'])
    else:
        se_times = se_roi['time_since_start']

    fdt = 2.3

    tmin = fdt * 5
    t_switch = 10*tmin

    se_times = se_roi['time_since_start']

    dead_intervals = build_dead_intervals(
        window_start_ms,
        window_stop_ms,
        s2_roi['time_since_start'],
        s1_roi,
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
        lambda t, s, n, tau, t_switch, tmin, c, d, k: to_fit_exp(
            t, s, n, tau, t_switch, tmin, c, d, k,
            s2_roi, s1_roi,
            window_start_ms,
            window_stop_ms
        )
    )

    m = Minuit(c1,s=0.1,n=1.5,tau=10.0,t_switch=t_switch,tmin=tmin,c=0.5,d=0.5,k=0.0)

    m.limits['n'] = (1.0001, 5)
    m.limits['s'] = (0, None)
    m.limits['tau'] = (0.5, None)

    # Important: t_switch must be larger than tmin.
    m.limits['t_switch'] = (tmin * 1.01, None)

    m.limits['c'] = (0, 5)
    m.limits['d'] = (-5, 5)
    m.limits['k'] = (0, 10)

    m.fixed['k'] = False
    m.fixed['tmin'] = True
    #m.fixed['c'] = True
    #m.fixed['d'] = True


    # Recommended at first: fix the transition time.
    # Once the model behaves sensibly, you can comment this out.
    #m.fixed['t_switch'] = True

    def run_minimization(m, strategy=1, retries=0):
        m.strategy = strategy
        m.migrad(ncall=3000)

        if (not m.valid) and retries < 3:
            print(f"Minimization failed, retry #{retries + 1} with adjusted parameters")

            if retries == 0:
                m.values['d'] = -1
            elif retries == 1:
                m.values['s'] = 20e-10
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

    BIC = (m.fval) + (np.log(n_obs) * n_free)

    print(f"Minimisation Status: \n{m.fmin}")

    values, errors = m.values, m.errors

    fit_params = ['s', 'n', 'tau', 't_switch', 'tmin', 'c', 'd', 'k']

    results_df = pd.DataFrame({
        "Parameter": fit_params,
        "Value": [values[p] for p in fit_params],
        "Error": [errors[p] for p in fit_params],
    })

    print("Fitted Parameters and Errors:")
    print(results_df)

    if record_results:
        results_log(
            run_id,
            s2_roi,
            values,
            errors,
            BIC,
            m.fmin.has_made_posdef_covar,
            seconds_range=seconds_range,
            filename=filename
        )
    print(f"\nThe amount of single electrons in the region of interest is: {len(se_roi)}")

    return values, errors, m.covariance, BIC

#------------------------------------------------------------------------------------------------------------

def cost_func_exp_additive(run_id, s2_roi, se_roi, s1_roi,
                           seconds_range=None,
                           model='exp_additive',
                           record_results=False,
                           filename="fit_results_exp_additive.csv"):
    """
    Separate cost function for additive exponential + power-law delayed-electron fit.
    """

    print(f"\nRunning the additive exponential + power-law cost function now")

    if model != 'exp_additive':
        raise ValueError(
            f"cost_func_exp_additive is only for model='exp_additive', got {model!r}"
        )

    fdt = 2.3

    # For fair comparison with your old 'new' model, start with 5*fdt.
    # You can later test 3*fdt.
    tmin = 5 * fdt

    window_start_ms = seconds_range[0] * 1e3
    window_stop_ms = seconds_range[1] * 1e3

    se_times = se_roi['time_since_start']

    dead_intervals = build_dead_intervals(
        window_start_ms,
        window_stop_ms,
        s2_roi['time_since_start'],
        s1_roi,
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
        lambda t, s, n, tau, f_exp, tmin, c, d, k: to_fit_exp_additive(
            t, s, n, tau, f_exp, tmin, c, d, k,
            s2_roi, s1_roi, window_start_ms, window_stop_ms
        )
    )

    m = Minuit(
        c1,
        s=1.5,
        n=1.35,
        tau=45.0,
        f_exp=0.3,
        tmin=tmin,
        c=0.8,
        d=1.4,
        k=0.01
    )

    m.limits['s'] = (0, None)
    m.limits['n'] = (1.2, 5.0)
    m.limits['tau'] = (0.2, 300.0)
    m.limits['f_exp'] = (0.0, 1.0)
    m.limits['c'] = (0.0, 5.0)
    m.limits['d'] = (-5.0, 5.0)
    m.limits['k'] = (0.0, 10.0)

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

    BIC = (m.fval) + (np.log(n_obs) * n_free)

    print(f"Minimisation Status: \n{m.fmin}")

    values, errors = m.values, m.errors

    fit_params = ['s', 'n', 'tau', 'f_exp', 'tmin', 'c', 'd', 'k']

    results_df = pd.DataFrame({
        "Parameter": fit_params,
        "Value": [values[p] for p in fit_params],
        "Error": [errors[p] for p in fit_params],
    })

    print("Fitted Parameters and Errors:")
    print(results_df)

    if record_results:
        results_log(
            run_id,
            s2_roi,
            values,
            errors,
            BIC,
            m.fmin.has_made_posdef_covar,
            seconds_range=seconds_range,
            filename=filename
        )

    print(f"\nThe amount of single electrons used in the fit is: {len(se_times)}")
    print(f"The amount of single electrons before live-mask is: {len(se_roi)}")

    return values, errors, m.covariance, BIC

#------------------------------------------------------------------------------------------------------------
def cost_func_pure_exp(run_id, s2_roi, se_roi, s1_roi,
                           seconds_range=None,
                           model='pure_exp',
                           record_results=False,
                           filename="fit_results_pure_exp.csv"):
    """
    Separate cost function for pure exponential delayed electron fit.
    """

    print(f"\nRunning the pure exponential cost function now")

    if model != 'pure_exp':
        raise ValueError(
            f"cost_func_pure_exp is only for model='pure_exp', got {model!r}"
        )

    fdt = 2.3

    # For fair comparison with your old 'new' model, start with 5*fdt.
    # You can later test 3*fdt.
    tmin = 5 * fdt

    window_start_ms = seconds_range[0] * 1e3
    window_stop_ms = seconds_range[1] * 1e3

    se_times = se_roi['time_since_start']

    dead_intervals = build_dead_intervals(
        window_start_ms,
        window_stop_ms,
        s2_roi['time_since_start'],
        s1_roi,
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
        lambda t, s, tau, tmin, c, d, k: to_fit_pure_exp(
            t, s, tau, tmin, c, d, k,
            s2_roi, s1_roi, window_start_ms, window_stop_ms
        )
    )

    m = Minuit(
        c1,
        s=1.5,
        tau=45.0,
        tmin=tmin,
        c=0.8,
        d=1.4,
        k=0.01
    )

    m.limits['s'] = (0, None)
    m.limits['tau'] = (0.2, 300.0)
    m.limits['c'] = (0.0, 5.0)
    m.limits['d'] = (-5.0, 5.0)
    m.limits['k'] = (0.0, 10.0)
    m.fixed['k'] = False
    m.fixed['tmin'] = True

    def run_minimization(m, strategy=1, retries=0):
        m.strategy = strategy
        m.migrad(ncall=3000)

        if (not m.valid) and retries < 3:
            print(f"Minimization failed, retry #{retries + 1} with adjusted parameters")

            if retries == 0:
                m.values['tau'] = 5.0
            elif retries == 1:
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

    fit_params = ['s', 'tau', 'tmin', 'c', 'd', 'k']

    results_df = pd.DataFrame({
        "Parameter": fit_params,
        "Value": [values[p] for p in fit_params],
        "Error": [errors[p] for p in fit_params],
    })

    print("Fitted Parameters and Errors:")
    print(results_df)

    if record_results:
        results_log(
            run_id,
            s2_roi,
            values,
            errors,
            BIC,
            m.fmin.has_made_posdef_covar,
            seconds_range=seconds_range,
            filename=filename
        )

    print(f"\nThe amount of single electrons in the region of interest is: {len(se_roi)}")

    return values, errors, m.covariance, BIC
#------------------------------------------------------------------------------------------------------------
def cdf_plot(s2_roi, se_roi, s1_roi, values, cov, model = 'new',
             seconds_range = None, ax = None, plot_zoom = (0, 0),
             label=None, color='r',extra_models=None, show_model_errors = False, show=True):
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

    resolution_ms = 10 # histogram bin size

    if seconds_range is None:
        raise ValueError("seconds_range must be provided")

    window_start_ms = seconds_range[0] * 1e3
    window_stop_ms  = seconds_range[1] * 1e3
    window_width_ms = window_stop_ms - window_start_ms

    #Getting the zoom parameters on the CDF plot was annoying
    if plot_zoom != (0, 0):
        zoom_start_rel_ms = plot_zoom[0] * 1e3   # relative to window start
        zoom_width_ms     = plot_zoom[1] * 1e3
        zoom_end_rel_ms   = zoom_start_rel_ms + zoom_width_ms

        zoom_end_rel_ms = min(zoom_end_rel_ms, window_width_ms)

        time_start_ms = window_start_ms + zoom_start_rel_ms
        time_stop_ms  = window_start_ms + zoom_end_rel_ms

        plot_shift_ms = window_start_ms + zoom_start_rel_ms
        x_axis_left   = 0
        x_axis_right  = zoom_end_rel_ms - zoom_start_rel_ms   # = width_ms

    else:
        # No zoom = show whole window, shifted to 0
        zoom_start_rel_ms = 0
        time_start_ms = window_start_ms
        time_stop_ms  = window_stop_ms
        plot_shift_ms = window_start_ms
        x_axis_left   = 0
        x_axis_right  = window_width_ms

    n_bins = max(1, int((time_stop_ms - time_start_ms) / resolution_ms))

    if model == 'count':
        se_times = np.repeat(se_roi['time_since_start'], se_roi['n_electron_rec'])
    else:
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

    if model == 'radial':
        model_rate, model_errors = propagate(
            lambda p: multi_powerlaw_wrap_radial(t_model, p, s2_roi, s1_roi)[1],
            values, cov
        )
    elif model == 'exp':
        model_rate, model_errors = propagate(
            lambda p: multi_exp_powerlaw_wrap(t_model, p, s2_roi, s1_roi, window_start_ms, window_stop_ms)[1],
            values, cov
        )
        model_errors_prop = np.sqrt(np.diag(model_errors))
    elif model == 'exp_additive':
        model_rate, model_errors = propagate(
            lambda p: multi_exp_additive_wrap(t_model, p, s2_roi, s1_roi, window_start_ms, window_stop_ms)[1],
            values, cov
        )
        model_errors_prop = np.sqrt(np.diag(model_errors))
    else:
        model_rate, model_errors = propagate(
            lambda p: multi_powerlaw_wrap(t_model, p, s2_roi, s1_roi, window_start_ms, window_stop_ms)[1],
            values, cov
        )
        model_errors_prop = np.diag(model_errors)**0.5
    t_abs = np.arange(time_start_ms, time_stop_ms, 0.5)
    # if model == 'exp':
    #     total_rate, p = multi_exp_powerlaw_wrap(
    #         t_abs,
    #         values,
    #         s2_roi,
    #         s1_roi,
    #         window_start_ms,
    #         window_stop_ms
    #     )
    #     _, rate_at_events = multi_exp_powerlaw_wrap(
    #         se_times_zoom,
    #         values,
    #         s2_roi,
    #         s1_roi,
    #         window_start_ms,
    #         window_stop_ms
    #     )
    # elif model == 'exp_additive':
    #     total_rate, p = multi_exp_additive_wrap(
    #         t_abs,
    #         values,
    #         s2_roi,
    #         s1_roi,
    #         window_start_ms,
    #         window_stop_ms
    #     )
    #     _, rate_at_events = multi_exp_additive_wrap(
    #         se_times_zoom,
    #         values,
    #         s2_roi,
    #         s1_roi,
    #         window_start_ms,
    #         window_stop_ms
    #     )
    # else:
    #     total_rate, p = new_power_law_pdf(
    #         t_abs, values[0], values[1], values[2], values[3],
    #         values[4], values[5], s2_roi, s1_roi, window_start_ms, window_stop_ms
    #     )
    #     _, rate_at_events = multi_powerlaw_wrap(
    #         se_times_zoom,
    #         values,
    #         s2_roi,
    #         s1_roi,
    #         window_start_ms,
    #         window_stop_ms
    #     )

    total_rate, p, rate_at_events = evaluate_model_for_plot(model, values, t_abs, s1_roi, s2_roi, se_times_zoom,
                                                            window_start_ms, window_stop_ms)

    t_plot = t_abs - plot_shift_ms  # plot in relative zoom coordinates

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(15, 5))
        new_ax = True
    else:
        new_ax = False

    weights = np.ones_like(se_times_plot) / bin_width

    ax.hist(se_times_plot, bins=bin_edges_power_sum, weights = weights, color='k',
            histtype='step', label='Observed SE rate')

    # ax.errorbar(bin_centers_plot, hist_power_sum,
    #             yerr=np.sqrt(hist_power_sum), fmt='+', color='orange', markersize=0.4)

    for s2_time in s2_roi['time_since_start']:
        start = s2_time - plot_shift_ms
        stop = s2_time + 5*2.3 - plot_shift_ms

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
    # p_plot = p.copy()
    #
    # for s1_time in s1_roi:
    #     mask = (t_abs >= s1_time) & (t_abs <= s1_time + 4.6)
    #     p_plot[mask] = np.nan
    #
    # for s2_time in s2_roi["time_since_start"]:
    #     mask = (t_abs >= s2_time) & (t_abs <= s2_time + 5*2.3)
    #     p_plot[mask] = np.nan

    p_plot = mask_dead_zones(p, t_abs, s1_roi, s2_roi, window_start_ms, window_stop_ms)

    if label is None:
        if model == 'exp_additive':
            label = 'Exp + power-law fit'
        elif model == 'new':
            label = 'Power-law fit'
        elif model == 'exp':
            label = 'Exp-to-power-law fit'
        else:
            label = f'{model} fit'
    if show_model_errors:
        ax.errorbar(
            bin_centers_plot,
            model_rate * bin_width,
            yerr=model_errors_prop * bin_width,
            fmt='none',
            ecolor = 'b',
            alpha = 1.0,
            elinewidth = 1.5,
            capsize=2,
            label = "model uncertainty"
        )

    #p_plot = p.copy()
    #p_plot[p_plot <= 0] = np.nan

    ax.plot(t_plot, p_plot, color='r', label = label)
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

            _, p_extra, _ = evaluate_model_for_plot(extra_model, extra_values, t_abs, s1_roi, s2_roi, se_times_zoom,
                                                    window_start_ms, window_stop_ms)
            p_extra_plot = mask_dead_zones(p_extra, t_abs, s1_roi, s2_roi, window_start_ms, window_stop_ms)

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
# ----------------------------------------------------------------------------------------------------------
# helpers for cdf_plot
def evaluate_model_for_plot(model_name, vals, t_abs, s1_roi, s2_roi, se_times_zoom, window_start_ms, window_stop_ms):
    if model_name == 'exp':
        total_rate, p = multi_exp_powerlaw_wrap(
            t_abs,
            vals,
            s2_roi,
            s1_roi,
            window_start_ms,
            window_stop_ms
        )
        _, rate_at_events = multi_exp_powerlaw_wrap(
            se_times_zoom,
            vals,
            s2_roi,
            s1_roi,
            window_start_ms,
            window_stop_ms
        )

    elif model_name == 'exp_additive':
        total_rate, p = multi_exp_additive_wrap(
            t_abs,
            vals,
            s2_roi,
            s1_roi,
            window_start_ms,
            window_stop_ms
        )
        _, rate_at_events = multi_exp_additive_wrap(
            se_times_zoom,
            vals,
            s2_roi,
            s1_roi,
            window_start_ms,
            window_stop_ms
        )

    else:
        total_rate, p = new_power_law_pdf(
            t_abs,
            vals[0], vals[1], vals[2], vals[3],
            vals[4], vals[5],
            s2_roi,
            s1_roi,
            window_start_ms,
            window_stop_ms
        )
        _, rate_at_events = multi_powerlaw_wrap(
            se_times_zoom,
            vals,
            s2_roi,
            s1_roi,
            window_start_ms,
            window_stop_ms
        )

    return total_rate, p, rate_at_events
def mask_dead_zones(p_in, t_abs, s1_roi, s2_roi, window_start_ms, window_stop_ms):
    p_out = p_in.copy()

    for s1_time in s1_roi:
        dead = (t_abs >= s1_time) & (t_abs <= s1_time + 4.6)
        p_out[dead] = np.nan

    for s2_time in s2_roi["time_since_start"]:
        dead = (t_abs >= s2_time) & (t_abs <= s2_time + 5 * 2.3)
        p_out[dead] = np.nan

    return p_out

#-----------------------------------------------------------------------------------------------------------

#Functions below here are internal, not really intended to be called directly

#These functions here just kind of as wrappers for other stuff
#cost.Extended... needs a function to be passed in to work, but it's finicky
#Kind of the same deal for using 'propagate', just a slightly different form.

def to_fit(t, s, n, tmin, c, d, k, s2_roi, s1_roi, window_start_ms, window_stop_ms, model = 'new'):
    return new_power_law_pdf(t, s, n, tmin, c, d, k, s2_roi, s1_roi, window_start_ms, window_stop_ms, model = model)
#-----------------------------------------------------------------------------------------------------------

def multi_powerlaw_wrap(t, p, s2_roi, s1_roi,window_start_ms, window_stop_ms, model = 'new'):
    s, n, tmin, c, d, k = p
    return new_power_law_pdf(t, s, n, tmin, c, d, k, s2_roi, s1_roi, window_start_ms, window_stop_ms, model = model)

#-----------------------------------------------------------------------------------------------------------

def to_fit_radial(t, s, n, tmin, c, d, k, A, r0, r_p, s2_roi, s1_roi):
    return new_power_law_pdf(t, s, n, tmin, c, d, k, s2_roi, s1_roi, A, r0, r_p, model = 'radial')

#-----------------------------------------------------------------------------------------------------------

def multi_powerlaw_wrap_radial(t, p, s2_roi, s1_roi):
    s, n, tmin, c, d, k, A, r0, r_p = p
    return new_power_law_pdf(t, s, n, tmin, c, d, k, s2_roi, s1_roi, A, r0, r_p, model = 'radial')

# -----------------------------------------------------------------------------------------------------------

def to_fit_exp(t, s, n, tau, t_switch, tmin, c, d, k, s2_roi, s1_roi, window_start_ms, window_stop_ms):
    return exp_power_law_pdf(t, s, n, tau, t_switch, tmin, c, d, k, s2_roi, s1_roi, window_start_ms, window_stop_ms, model = 'exp')

# -----------------------------------------------------------------------------------------------------------
def multi_exp_powerlaw_wrap(t, p, s2_roi, s1_roi, window_start_ms, window_stop_ms):
    """
    Wrapper for the exponential-to-power-law model.

    Expected parameter order:
        s, n, tau, t_switch, tmin, c, d, k
    """

    s, n, tau, t_switch, tmin, c, d, k = p

    return exp_power_law_pdf(t,s, n, tau, t_switch, tmin, c, d, k,s2_roi,s1_roi,window_start_ms,window_stop_ms)

# -----------------------------------------------------------------------------------------------------------

def to_fit_exp_additive(t, s, n, tau, f_exp, tmin, c, d, k, s2_roi, s1_roi, window_start_ms, window_stop_ms):
    return new_exp_additive_pdf(
        t, s, n, tau, f_exp, tmin, c, d, k,
        s2_roi, s1_roi, window_start_ms, window_stop_ms
    )

# -----------------------------------------------------------------------------------------------------------

def multi_exp_additive_wrap(t, p, s2_roi, s1_roi, window_start_ms, window_stop_ms):
    """
    Wrapper for additive exponential + power-law model.

    Expected parameter order:
        s, n, tau, f_exp, tmin, c, d, k
    """

    s, n, tau, f_exp, tmin, c, d, k = p

    return new_exp_additive_pdf(
        t,
        s, n, tau, f_exp, tmin, c, d, k,
        s2_roi,
        s1_roi,
        window_start_ms,
        window_stop_ms,
        model = 'exp_additive'
    )
# -----------------------------------------------------------------------------------------------------------
def to_fit_pure_exp(t, s, tau, tmin, c, d, k, s2_roi, s1_roi, window_start_ms, window_stop_ms):
    return pure_exp_pdf(
        t, s, tau, tmin, c, d, k,
        s2_roi, s1_roi, window_start_ms, window_stop_ms
    )
# -----------------------------------------------------------------------------------------------------------
def multi_pure_exp_wrap(t, p, s2_roi, s1_roi, window_start_ms, window_stop_ms):
    """
    Wrapper for pure exponential pdf

    Expected parameter order:
        s, tau, tmin, c, d, k
    """

    s, tau, tmin, c, d, k = p

    return pure_exp_pdf(
        t,
        s, tau, tmin, c, d, k,
        s2_roi,
        s1_roi,
        window_start_ms,
        window_stop_ms,
        model = 'pure_exp'
    )
# -----------------------------------------------------------------------------------------------------------
#Buncha numba functions 
@njit(cache=False)
def _compute_norms_basic(s, c, d, areas, ranges):
    # areas in phe, ranges in ms (you already divide by 1e6 for range_50p_area upstream)
    return s * (areas ** c) * (ranges ** d)

@njit
def _compute_norms_radial(s, c, d, A, r0, r_p, areas, ranges, r): #Computing the norms for each pS2 can be a big slow-down, so this helps
    return s * (areas ** c) * (ranges ** d) * (1 / (1 + np.exp(A*(r - r0) / r_p)))

@njit(cache=False)
def _cdf_scalar(x, tmin, n):
    # CDF of the single-pS2 power law at 'x' (scalar); used for the "cut" term
    return 0.0 if x < tmin else 1.0 - (tmin / x) ** (n - 1.0)
#I added the powerlaw kernel to try to make a fair comparison with my hybrid model
@njit(cache=False)
def _powerlaw_kernel_integral_normed(u0, u1, tmin, n):
    """
    Integral of the normalized power-law kernel over [u0, u1].

    h(u) = (n - 1)/tmin * (u/tmin)^(-n), for u > tmin.
    """

    if n <= 1.0:
        return 0.0

    if u1 <= u0:
        return 0.0

    if u1 <= tmin:
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
def _last_leq(arr, x):
    """Index of last element <= x in sorted arr, or -1 if none."""
    lo = 0
    hi = arr.size - 1
    idx = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] <= x:
            idx = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return idx

@njit(cache=False)
def _in_dead_s2(t, s2_t, tmin):
    """True if time 't' lies in any S2 dead-zone [s2, s2+tmin].  s2_t must be sorted."""
    if s2_t.size == 0:
        return False
    idx = _last_leq(s2_t, t)
    if idx == -1:
        return False
    return (t - s2_t[idx]) <= tmin

@njit(cache=False)
def _in_dead_s1(t, s1_t_sorted):
    """True if time 't' lies in any S1 dead-zone [s1, s1+4.6 ms].  s1_t_sorted must be sorted."""
    if s1_t_sorted.size == 0:
        return False
    idx = _last_leq(s1_t_sorted, t)
    if idx == -1:
        return False
    return (t - s1_t_sorted[idx]) <= 4.6


def merge_intervals(intervals):
    """
    Merge overlapping intervals.

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

def build_dead_intervals(window_start_ms, window_stop_ms, s2_t_sorted, s1_sorted, tmin):
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

def build_live_intervals(window_start_ms, window_stop_ms, dead_intervals):
    """returns the livetime intervals having removed the dead intervals, i.e. areas directly after pS2's or S1's"""
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


@njit(cache=False, parallel=True, fastmath=True)
def _powerlaw_pdf_basic_consistent(t_grid, s, n, tmin, c, d, k,
                                   A, r0, r_p, s2_t_sorted, s2_area_sorted, s2_rng_sorted, 
                                   s2_r_sorted, s1_t_sorted, model_flag, live_starts, live_stops):
    """
    Core Numba evaluator for both new/old and radial models.
    """

    # Compute norms (pre-factor for each pS2)
    if model_flag == 1:
        norms = _compute_norms_radial(s, c, d, A, r0, r_p,
                                      s2_area_sorted, s2_rng_sorted, s2_r_sorted)
    else:
        norms = _compute_norms_basic(s, c, d, s2_area_sorted, s2_rng_sorted)

    diff = np.full(t_grid.size, k)
    scale = (n - 1.0) / tmin

    # PDF evaluation
    for i in prange(t_grid.size):
        ti = t_grid[i]

        # Dead zone → zero rate
        # if _in_dead_s2(ti, s2_t_sorted, tmin) or _in_dead_s1(ti, s1_t_sorted):
        #     diff[i] = 0.0
        #     continue

        acc = 0.0
        for j in range(s2_t_sorted.size):
            dt = ti - s2_t_sorted[j]
            if dt > tmin:
                acc += norms[j] * scale * (dt / tmin)**(-n)

        diff[i] += acc

    # Correction term (as in GOF), this is how Conor did it, below these comments is my version
    # cut = np.zeros(s2_t_sorted.size)
    # for j in range(s2_t_sorted.size):
    #     up = _cdf_scalar(s2_t_sorted[j] + tmin, tmin, n)
    #     lo = _cdf_scalar(s2_t_sorted[j],         tmin, n)
    #     cut[j] = norms[j] * (up - lo)
    #
    # # Live time (same approximation as GOF)
    # total_time = t_grid[-1] - t_grid[0]
    # live = total_time - s2_t_sorted.size * tmin - s1_t_sorted.size * 4.6
    # if live < 0.0:
    #     live = 0.0
    #
    # total_rate = norms.sum() - cut.sum() + k * live

    total_rate = 0.0

    # Background contribution over live intervals
    for ell in range(live_starts.size):
        total_rate += k * (live_stops[ell] - live_starts[ell])

    # S2-correlated contribution over live intervals
    for j in range(s2_t_sorted.size):
        for ell in range(live_starts.size):
            u0 = live_starts[ell] - s2_t_sorted[j]
            u1 = live_stops[ell] - s2_t_sorted[j]

            total_rate += norms[j] * _powerlaw_kernel_integral_normed(u0,u1,tmin,n)
    return total_rate, diff

def new_power_law_pdf(t_grid, s, n, tmin, c, d, k,
                      pS2s_struct, s1_times_ms,
                      window_start_ms, window_stop_ms,
                      A=None, r0=None, r_p=None,
                      model='new'):
    """
    Wrapper for the Numba-compiled power-law evaluator.
    """

    # Extract S2 fields
    s2_t = pS2s_struct['time_since_start'].astype(np.float64)
    s2_area = pS2s_struct['area'].astype(np.float64)
    s2_rng  = (pS2s_struct['range_50p_area'] / 1e6).astype(np.float64)

    # Ensure radius exists
    if 'r' in pS2s_struct.dtype.names:
        s2_r = pS2s_struct['r'].astype(np.float64)
    else:
        s2_r = np.zeros_like(s2_t)

    # Sort S2s consistently
    order = np.argsort(s2_t)
    s2_t_sorted    = np.ascontiguousarray(s2_t[order])
    s2_area_sorted = np.ascontiguousarray(s2_area[order])
    s2_rng_sorted  = np.ascontiguousarray(s2_rng[order])
    s2_r_sorted    = np.ascontiguousarray(s2_r[order])

    # Scale area and width/range to dimensionless values, added by me was not in Conors master thesis code
    area_ref = np.median(s2_area_sorted)
    width_ref = np.median(s2_rng_sorted)

    if area_ref <= 0:
        area_ref = 1.0

    if width_ref <= 0:
        width_ref = 1.0

    s2_area_scaled = np.ascontiguousarray(s2_area_sorted / area_ref)
    s2_rng_scaled = np.ascontiguousarray(s2_rng_sorted / width_ref)

    # Sort S1 times
    if s1_times_ms is not None and len(s1_times_ms) > 0:
        s1_sorted = np.ascontiguousarray(np.sort(s1_times_ms.astype(np.float64)))
    else:
        s1_sorted = np.zeros(0, dtype=np.float64)

#added by me, wasn't in Conor's code
    dead_intervals = build_dead_intervals(window_start_ms, window_stop_ms, s2_t_sorted, s1_sorted, tmin)

    live_intervals = build_live_intervals(window_start_ms,window_stop_ms,dead_intervals)

    live_starts = np.ascontiguousarray(np.array([x[0] for x in live_intervals], dtype=np.float64))

    live_stops = np.ascontiguousarray(np.array([x[1] for x in live_intervals], dtype=np.float64))

    #Numba doesn't like 'radial' or whatever
    model_flag = 1 if model == 'radial' else 0
    if model_flag == 0: #Don't actually need these just think numba needs some values or whatever
        A = 0.0
        r0 = 0.0
        r_p = 1.0

    return _powerlaw_pdf_basic_consistent(
        np.ascontiguousarray(t_grid.astype(np.float64)),
        float(s), float(n), float(tmin), float(c), float(d), float(k),
        float(A), float(r0), float(r_p),
        s2_t_sorted, s2_area_scaled, s2_rng_scaled, s2_r_sorted,
        s1_sorted,
        model_flag,
        live_starts, live_stops
    )
@njit(cache=False, parallel=True, fastmath=True)
def _exp_powerlaw_pdf_basic_consistent(
    t_grid,
    s, n, tau, t_switch, tmin, c, d, k,
    A, r0, r_p,
    s2_t_sorted, s2_area_sorted, s2_rng_sorted, s2_r_sorted,
    s1_t_sorted,
    model_flag,
    live_starts,
    live_stops
):
    """
     Exponential-near-start + power-law-late model.

     λ(t) = k + Σ_p norm_p * g(t - t_p)
     """

    if model_flag == 1:
        norms = _compute_norms_radial(
            s, c, d, A, r0, r_p,
            s2_area_sorted, s2_rng_sorted, s2_r_sorted
        )
    else:
        norms = _compute_norms_basic(
            s, c, d,
            s2_area_sorted, s2_rng_sorted
        )

    diff = np.full(t_grid.size, k)

    # Pointwise rate
    for i in prange(t_grid.size):
        ti = t_grid[i]

        # if _in_dead_s2(ti, s2_t_sorted, tmin) or _in_dead_s1(ti, s1_t_sorted):
        #     diff[i] = 0.0
        #     continue

        acc = 0.0

        for j in range(s2_t_sorted.size):
            dt = ti - s2_t_sorted[j]

            if dt > tmin:
                acc += norms[j] * _hybrid_kernel_value_normed(dt,n,tau,tmin,t_switch)

        diff[i] += acc

    # Integrated expected event count over the fit window
    total_rate = 0.0

    #background term
    for ell in range(live_starts.size):
        total_rate += k*(live_stops[ell] - live_starts[ell])

    #pS2 term
    for j in range(s2_t_sorted.size):
        for ell in range(live_starts.size):
            u0 = live_starts[ell] - s2_t_sorted[j]
            u1 = live_stops[ell] - s2_t_sorted[j]

            total_rate += norms[j] * _hybrid_kernel_integral_normed(u0,u1,n,tau,tmin,t_switch)

    return total_rate, diff
def exp_power_law_pdf(t_grid, s, n, tau, t_switch, tmin, c, d, k, pS2s_struct, s1_times_ms,window_start_ms, window_stop_ms,
                      A = None, r0 = None,
                      r_p = None, model = 'exp'):
    """Wrapper for the Numba-compiled exponential-to-power-law evaluator."""

    # Extract S2 fields
    s2_t = pS2s_struct['time_since_start'].astype(np.float64)
    s2_area = pS2s_struct['area'].astype(np.float64)
    s2_rng  = (pS2s_struct['range_50p_area'] / 1e6).astype(np.float64)

    # Ensure radius exists
    if 'r' in pS2s_struct.dtype.names:
        s2_r = pS2s_struct['r'].astype(np.float64)
    else:
        s2_r = np.zeros_like(s2_t)

    # Sort S2s consistently
    order = np.argsort(s2_t)
    s2_t_sorted    = np.ascontiguousarray(s2_t[order])
    s2_area_sorted = np.ascontiguousarray(s2_area[order])
    s2_rng_sorted  = np.ascontiguousarray(s2_rng[order])
    s2_r_sorted    = np.ascontiguousarray(s2_r[order])
    # Scale area and width/range to dimensionless values
    area_ref = np.median(s2_area_sorted)
    width_ref = np.median(s2_rng_sorted)

    if area_ref <= 0:
        area_ref = 1.0

    if width_ref <= 0:
        width_ref = 1.0

    s2_area_scaled = np.ascontiguousarray(s2_area_sorted / area_ref)
    s2_rng_scaled = np.ascontiguousarray(s2_rng_sorted / width_ref)

    # Sort S1 times
    if s1_times_ms is not None and len(s1_times_ms) > 0:
        s1_sorted = np.ascontiguousarray(np.sort(s1_times_ms.astype(np.float64)))
    else:
        s1_sorted = np.zeros(0, dtype=np.float64)

    dead_intervals = build_dead_intervals(window_start_ms, window_stop_ms, s2_t_sorted, s1_sorted, tmin)

    live_intervals = build_live_intervals(window_start_ms, window_stop_ms, dead_intervals)

    live_starts = np.ascontiguousarray(np.array([x[0] for x in live_intervals]), dtype = np.float64)

    live_stops = np.ascontiguousarray(np.array([x[1] for x in live_intervals]), dtype = np.float64)

    # Numba model flag
    model_flag = 1 if model == 'radial' else 0

    if model_flag == 0:
        A = 0.0
        r0 = 0.0
        r_p = 1.0

    return _exp_powerlaw_pdf_basic_consistent(
        np.ascontiguousarray(t_grid.astype(np.float64)),
        float(s), float(n), float(tau), float(t_switch), float(tmin),
        float(c), float(d), float(k),
        float(A), float(r0), float(r_p),
        s2_t_sorted, s2_area_scaled, s2_rng_scaled, s2_r_sorted,
        s1_sorted,
        model_flag,
        live_starts,
        live_stops
    )
# defining how we go about integrating for a hybrid exponential+power law

@njit(cache=False)
def _hybrid_kernel_total_integral(n, tau, tmin, t_switch):
    """
    Integral of the unnormalized hybrid kernel over [tmin, infinity).
    """

    if n <= 1.0 or tau <= 0.0 or tmin <= 0.0 or t_switch <= tmin:
        return 0.0

    M = np.exp((t_switch - tmin) / tau)

    # exponential part: tmin to t_switch
    exp_int = M * tau * (
        1.0 - np.exp(-(t_switch - tmin) / tau)
    )

    # power-law part: t_switch to infinity
    pl_int = t_switch / (n - 1.0)

    return exp_int + pl_int

@njit(cache=False)
def _hybrid_kernel_value_normed(dt, n, tau, tmin, t_switch):
    if dt <= tmin:
        return 0.0

    I = _hybrid_kernel_total_integral(n, tau, tmin, t_switch)
    if I <= 0.0:
        return 0.0

    M = np.exp((t_switch - tmin) / tau)

    if dt < t_switch:
        g = M * np.exp(-(dt - tmin) / tau)
    else:
        g = (dt/t_switch) ** (-n)

    return g / I
@njit(cache=False)
def _hybrid_kernel_integral(u0, u1, n, tau, tmin, t_switch):
    """
    Integral of the hybrid exponential-to-power-law kernel over delay interval [u0, u1].

    g(u) = 0 for u <= tmin
    g(u) = M exp(-(u - tmin)/tau) for tmin < u < t_switch
    g(u) = u^(-n) for u >= t_switch

    The exponential is matched continuously to the power law at t_switch.
    """

    if u1 <= tmin:
        return 0.0

    lo = u0
    hi = u1

    if lo < tmin:
        lo = tmin

    if hi <= lo:
        return 0.0

    total = 0.0

    # Matching factor
    M = np.exp((t_switch - tmin) / tau)

    # Exponential section
    if lo < t_switch:
        exp_lo = lo
        exp_hi = hi

        if exp_hi > t_switch:
            exp_hi = t_switch

        if exp_hi > exp_lo:
            total += M * tau * (
                np.exp(-(exp_lo - tmin) / tau)
                - np.exp(-(exp_hi - tmin) / tau)
            )

    # Power-law section
    if hi > t_switch:
        pl_lo = lo
        pl_hi = hi

        if pl_lo < t_switch:
            pl_lo = t_switch

        if pl_hi > pl_lo:
            # For n > 1:
            total += (t_switch ** n) * (
                pl_lo ** (1.0 - n) - pl_hi ** (1.0 - n)
            ) / (n - 1.0)

    return total

@njit(cache=False)
def _hybrid_kernel_integral_normed(u0, u1, n, tau, tmin, t_switch):
    raw = _hybrid_kernel_integral(u0, u1, n, tau, tmin, t_switch)

    I = _hybrid_kernel_total_integral(n, tau, tmin, t_switch)
    if I <= 0.0:
        return 0.0

    return raw / I
@njit(cache=False, parallel=True, fastmath=True)
def _pure_exp_pdf_basic_consistent(t_grid, s, tau, tmin,
                                  c,d,k,
                                  A, r0, r_p,
                                  s2_t_sorted, s2_area_sorted, s2_rng_sorted,
                                  s2_r_sorted, s1_t_sorted, model_flag,
                                  live_starts, live_stops):
    """
    Core pure exponential evaluator.
    Pointwise rate:
        lambda(t) = k* sum_p N_p h_exp(t-t_p)
    Expected count:
        Lambda = k*T_live
                + sum_p N_p sum_l int_{L_(l,0)-t_p} {L_(l,1)-t_p} h_exp(u)du
    where
        h_exp(u) = (1/tau) exp(-(u - tmin)/tau)
    is the normalized exponential kernel
    """

    # Guard invalid parameters
    if tau <= 0.0 or tmin <= 0.0:
        diff_bad = np.zeros(t_grid.size)
        return 0.0, diff_bad

    # Compute S2-dependent norms
    if model_flag == 1:
        norms = _compute_norms_radial(
            s, c, d, A, r0, r_p,
            s2_area_sorted, s2_rng_sorted, s2_r_sorted
        )
    else:
        norms = _compute_norms_basic(
            s, c, d,
            s2_area_sorted, s2_rng_sorted
        )

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
                acc += norms[j] * _exp_kernel_value_normed(
                    dt, tau, tmin
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

            total_rate += norm_j * _exp_kernel_integral_normed(
                u0, u1, tau, tmin
            )

    return total_rate, diff

def pure_exp_pdf(t_grid, s, tau, tmin, c, d, k,
                     pS2s_struct, s1_times_ms,
                     window_start_ms, window_stop_ms,
                     A=None, r0=None, r_p=None,
                     model='pure exp'):
    """
       Pure exponential model.

       For dt > tmin:

           h_exp(dt) =
               (1/tau) exp(-(dt - tmin)/tau)

       Kernel is normalized, so the S2 norm remains an expected yield.
       """

    # Extract S2 fields
    s2_t = pS2s_struct['time_since_start'].astype(np.float64)
    s2_area = pS2s_struct['area'].astype(np.float64)
    s2_rng = (pS2s_struct['range_50p_area'] / 1e6).astype(np.float64)

    if 'r' in pS2s_struct.dtype.names:
        s2_r = pS2s_struct['r'].astype(np.float64)
    else:
        s2_r = np.zeros_like(s2_t)

    # Sort S2s consistently
    order = np.argsort(s2_t)
    s2_t_sorted = np.ascontiguousarray(s2_t[order])
    s2_area_sorted = np.ascontiguousarray(s2_area[order])
    s2_rng_sorted = np.ascontiguousarray(s2_rng[order])
    s2_r_sorted = np.ascontiguousarray(s2_r[order])

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

    # Build dead intervals matching _in_dead_s2/_in_dead_s1
    dead_intervals = build_dead_intervals(window_start_ms, window_stop_ms, s2_t_sorted, s1_sorted, tmin)

    # Convert dead intervals to live intervals
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

    model_flag = 1 if model == 'radial' else 0

    if model_flag == 0:
        A = 0.0
        r0 = 0.0
        r_p = 1.0

    return _pure_exp_pdf_basic_consistent(
        np.ascontiguousarray(t_grid.astype(np.float64)),
        float(s), float(tau), float(tmin),
        float(c), float(d), float(k),
        float(A), float(r0), float(r_p),
        s2_t_sorted,
        s2_area_scaled,
        s2_rng_scaled,
        s2_r_sorted,
        s1_sorted,
        model_flag,
        live_starts,
        live_stops
    )

@njit(cache=False, parallel=True, fastmath=True)
def _exp_additive_pdf_basic_consistent(t_grid, s, n, tau, f_exp, tmin,
                                       c, d, k,
                                       A, r0, r_p,
                                       s2_t_sorted, s2_area_sorted, s2_rng_sorted,
                                       s2_r_sorted, s1_t_sorted, model_flag,
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
    if model_flag == 1:
        norms = _compute_norms_radial(
            s, c, d, A, r0, r_p,
            s2_area_sorted, s2_rng_sorted, s2_r_sorted
        )
    else:
        norms = _compute_norms_basic(
            s, c, d,
            s2_area_sorted, s2_rng_sorted
        )

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

def new_exp_additive_pdf(t_grid, s, n, tau, f_exp, tmin, c, d, k,
                         pS2s_struct, s1_times_ms,
                         window_start_ms, window_stop_ms,
                         A=None, r0=None, r_p=None,
                         model='exp_additive'):
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
    s2_t = pS2s_struct['time_since_start'].astype(np.float64)
    s2_area = pS2s_struct['area'].astype(np.float64)
    s2_rng = (pS2s_struct['range_50p_area'] / 1e6).astype(np.float64)

    if 'r' in pS2s_struct.dtype.names:
        s2_r = pS2s_struct['r'].astype(np.float64)
    else:
        s2_r = np.zeros_like(s2_t)

    # Sort S2s consistently
    order = np.argsort(s2_t)
    s2_t_sorted = np.ascontiguousarray(s2_t[order])
    s2_area_sorted = np.ascontiguousarray(s2_area[order])
    s2_rng_sorted = np.ascontiguousarray(s2_rng[order])
    s2_r_sorted = np.ascontiguousarray(s2_r[order])

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

    # Build dead intervals matching _in_dead_s2/_in_dead_s1
    if model == 'exp_additive':
        dead_intervals = build_dead_intervals(window_start_ms, window_stop_ms, s2_t_sorted, s1_sorted, tmin)

        # Convert dead intervals to live intervals
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
    else:
        live_starts = window_start_ms
        live_stops = window_stop_ms

    model_flag = 1 if model == 'radial' else 0

    if model_flag == 0:
        A = 0.0
        r0 = 0.0
        r_p = 1.0

    return _exp_additive_pdf_basic_consistent(
        np.ascontiguousarray(t_grid.astype(np.float64)),
        float(s), float(n), float(tau), float(f_exp), float(tmin),
        float(c), float(d), float(k),
        float(A), float(r0), float(r_p),
        s2_t_sorted,
        s2_area_scaled,
        s2_rng_scaled,
        s2_r_sorted,
        s1_sorted,
        model_flag,
        live_starts,
        live_stops
    )
@njit(cache=False)
def _exp_kernel_value_normed(dt, tau, tmin):
    if tau <= 0.0:
        return 0.0

    if dt <= tmin:
        return 0.0

    return (1.0 / tau) * np.exp(-(dt - tmin) / tau)


@njit(cache=False)
def _exp_cdf_scalar(x, tau, tmin):
    if tau <= 0.0:
        return 0.0

    if x <= tmin:
        return 0.0

    return 1.0 - np.exp(-(x - tmin) / tau)


@njit(cache=False)
def _exp_kernel_integral_normed(u0, u1, tau, tmin):
    if tau <= 0.0:
        return 0.0

    if u1 <= u0:
        return 0.0

    if u1 <= tmin:
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
    if f_exp < 0.0 or f_exp > 1.0:
        return 0.0

    h_exp = _exp_kernel_value_normed(dt, tau, tmin)

    h_pl = 0.0
    if n > 1.0 and dt > tmin:
        scale = (n - 1.0) / tmin
        h_pl = scale * (dt / tmin) ** (-n)

    return f_exp * h_exp + (1.0 - f_exp) * h_pl


@njit(cache=False)
def _additive_kernel_integral_normed(u0, u1, n, tau, f_exp, tmin):
    if f_exp < 0.0 or f_exp > 1.0:
        return 0.0

    exp_int = _exp_kernel_integral_normed(u0, u1, tau, tmin)
    pl_int = _powerlaw_kernel_integral_normed(u0, u1, tmin, n)

    return f_exp * exp_int + (1.0 - f_exp) * pl_int

# ------------------------------------------------------------------------------------------------------------------
#I'm gonna define an entirely new thing to take as an input some extra sources which are distinc from pS2's but may have
#an impact on the actual fit in a positive way.

@njit(cache=False)
def _compute_norms_source_ne(q, n_electron_rec):
    return q * n_electron_rec

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

def new_exp_additive_three_source_pdf(
    t_grid, s, n, tau, f_exp, tmin, c, d, q_weak, q_burst, k,
    pS2s_struct, source_like_struct, burst_source_struct, s1_times_ms, window_start_ms,
    window_stop_ms, model = 'extra_source'
):
    # Build common dead/live intervals
    pS2_t = pS2s_struct["time_since_start"].astype(np.float64)
    src_t = source_like_struct["time_since_start"].astype(np.float64)

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
    # WARNING: your existing new_exp_additive_pdf builds its own live intervals.
    # This is acceptable for a temporary test, but not fully consistent.
    total_pS2, rate_pS2 = new_exp_additive_pdf(
        t_grid,
        s, n, tau, f_exp, tmin, c, d, 0.0,
        pS2s_struct,
        s1_times_ms,
        live_starts,
        live_stops, model = model
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
def cost_func_exp_additive_three_source(run_id, s2_roi, source_like_roi, burst_source_roi, se_roi, s1_roi,
                           seconds_range=None,
                           model='extra_source',
                           record_results=False,
                           filename="fit_results_extra_source.csv"):
    """
    Separate cost function for additive exponential + power-law delayed-electron fit with extra source rate.
    """

    print(f"\nRunning the extra_source cost function now")

    if model != 'extra_source':
        raise ValueError(
            f"cost_func_exp_additive is only for model='extra_source', got {model!r}"
        )

    fdt = 2.3

    # For fair comparison with your old 'new' model, start with 5*fdt.
    # You can later test 3*fdt.
    tmin = 5 * fdt

    window_start_ms = seconds_range[0] * 1e3
    window_stop_ms = seconds_range[1] * 1e3

    se_times = se_roi['time_since_start']

    if hasattr(s1_roi, "dtype") and s1_roi.dtype.names is not None:
        s1_times = s1_roi["time_since_start"].astype(float)
    else:
        s1_times = np.asarray(s1_roi, dtype=float)

    dead_intervals = build_dead_intervals(
        window_start_ms,
        window_stop_ms,
        s2_roi["time_since_start"],
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
        n=1.35,
        tau=45.0,
        f_exp=0.3,
        tmin=tmin,
        c=0.8,
        d=1.4,
        q_weak= 0.01,
        q_burst= 0.09,
        k=0.01
    )

    m.limits['s'] = (0, None)
    m.limits['n'] = (1.2, 5.0)
    m.limits['tau'] = (0.2, 300.0)
    m.limits['f_exp'] = (0.0, 1.0)
    m.limits['c'] = (0.0, 5.0)
    m.limits['d'] = (-5.0, 5.0)
    m.limits['q_weak'] = (0.0, None)
    m.limits['q_burst'] = (0.0, None)
    m.limits['k'] = (0.0, 10.0)

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

    BIC = (m.fval) + (np.log(n_obs) * n_free)

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

    if record_results:
        results_log(
            run_id,
            s2_roi,
            values,
            errors,
            BIC,
            m.fmin.has_made_posdef_covar,
            seconds_range=seconds_range,
            filename=filename
        )

    print(f"\nThe amount of single electrons used in the fit is: {len(se_times)}")
    print(f"The amount of single electrons before live-mask is: {len(se_roi)}")

    return values, errors, m.covariance, BIC
#--------------------------------------------------------------------------------------------------------------------

def results_log(run_id, s2_roi, values, errors, bic_val, forced, 
                seconds_range = None, filename = "fit_results.csv"):
    """
    #TODO: Probably broke this function, sorry
    Appends fit results to a .csv file

    Inputs:
    - run_id: Identifier for the run (currently a global variable, uh oh)
    - s2_roi: s2 "region of interest", whatever we did the fit over
    - values: Minuit fit values   
    - errors: Minuit error values
    - bic_val: Bayesian Information Criterion value
    - forced: boolean, whether or not Minuit forced the covariance to be positive definite

    Outputs:
    - fit_results.csv I guess? + whatever is written to that file
    """
        
    if seconds_range is not None:
        start, end = ((seconds_range[0] * 1e9) + run_start), ((seconds_range[-1] * 1e9) + run_start)
        seconds_start, seconds_end = seconds_range[0], seconds_range[1]

    else:
        print("How did you get here")
    data = {
        'Run ID': run_id['name'],
        'Start Time (ns since epoch)': start,
        'End Time (ns since epoch)': end,
        'Start Time (s since run start)': seconds_start,
        'End time (s since run start)': seconds_end,
        'Number of pS2s included': len(s2_roi),
        's': values['s'], 'n': values['n'], 'tmin': values['tmin'], 'c': values['c'], 
        'd': values['d'], 'k': values['k'], #, 'r0': values['r0'], 'scaling': values['scaling']
        's_error': errors['s'], 'n_error': errors['n'], 'tmin_error': errors['tmin'], 'c_error': errors['c'],
        'd_error': errors['d'], 'k_error': errors['k'], #, 'r0_error': errors['r0'], 'scaling_error': errors['scaling']
        'forced pos. def. covariance': forced,
        'BIC': bic_val
    }

    #Have stored whether minuit has forced the covariance to be positive definite.
    #It's not necessarily a bad thing if so, but sometimes the n value or the errors can be weird.
    #Just good to have some way to filter them out later if necessary.
    
    results_row = pd.DataFrame([data])

    # Check if file exists -- write headers if not
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline = '') as f:
        results_row.to_csv(f, header = not file_exists, index = False)

    #TODO: Right now this allows duplicates to be written, 
    # currently just dealing with this by manual selection in other places when needed, so not actually a huge issue
    
    print(f"\n Results logged to {filename} \n")


#-----------------------------------------------------------------------------------------------------------