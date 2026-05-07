
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

def time_fitting(run_id, s2s, ses, s1_times = None, vetos = None, seconds_range = None, time_range = None, 
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

    else:
        results_df = pd.DataFrame({
            "Parameter": ['Run ID', 'Start Time (s)', 'End Time (s)', 's', 's_err', 'n', 'n_err', 'tmin', 'tmin_err',
                        'c', 'c_err', 'd', 'd_err', 'k', 'k_err', 'Num pS2s', 'Num SEs', 'BIC'],
            "Value": [run_id['name'], start_ms/1e3, end_ms/1e3, values[0], errors[0], values[1], errors[1],
                    values[2], errors[2], values[3], errors[3], values[4], errors[4],
                    values[5], errors[5], len(s2_region), len(se_region), BIC]
        })

    if plot:
        total_rate, differential_rate = cdf_plot(s2_region, se_region, S1_region, values, covariance, model = model,
                                 seconds_range = seconds_range)
        return results_df, covariance, total_rate, differential_rate, BIC

    else:
        t = np.linspace(se_region['time_since_start'][0], se_region['time_since_start'][-1], len(se_region) * 10)
        if model == 'radial':
            A = values[6]
            r0 = values[7]
            r_p = values[8]
        else: 
            A = None
            r0 = None
            r_p = None 
        total_rate, differential_rate = new_power_law_pdf(t, values[0], values[1], values[2], values[3], 
                                      values[4], values[5], s2_region, S1_region, A = A, r0 = r0, r_p = r_p, model = model) 
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

    c1 = cost.ExtendedUnbinnedNLL(se_times, 
                                  lambda t, s, n, tmin, c, d, k: to_fit(t, s, n, tmin, c, #, r0, scaling
                                                                                     d, k, s2_roi, s1_roi))
    
    m = Minuit(c1, s = 0.1, n = 1.5, tmin = tmin, c = 0.5, d = 0.5, k = 0.01)
    #I usually find changing s to like 0.5 or just something small like 20e-10 can help if things are going wrong
    
    m.limits['n'] = (1.0001, 5)
    m.limits['s'] = (0, None)
    m.limits['c'] = (0, 5)   # Tighter range
    m.limits['d'] = (-5, 5)  # Much tighter range - physical values should be around -1 to 1
    m.limits['k'] = (0, 10)
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

    BIC = (2 * m.fval) + (np.log(len(se_roi)) * m.nfit) #note: m.fval = −ln(L^)
    
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

    #AIC = (2 * m.fval) + (2 * m.nfit)
    BIC = (2 * m.fval) + (np.log(len(se_roi)) * m.nfit)

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

    c1 = cost.ExtendedUnbinnedNLL(
        se_times,
        lambda t, s, n, tau, t_switch, tmin, c, d, k: to_fit_exp(
            t, s, n, tau, t_switch, tmin, c, d, k,
            s2_roi, s1_roi,
            window_start_ms,
            window_stop_ms
        )
    )
    m = Minuit(c1,s=0.1,n=1.5,tau=10.0,t_switch=t_switch,tmin=tmin,c=0.5,d=0.5,k=0.01)

    m.limits['n'] = (1.0001, 5)
    m.limits['s'] = (0, None)
    m.limits['tau'] = (0.5, None)

    # Important: t_switch must be larger than tmin.
    m.limits['t_switch'] = (tmin * 1.01, None)

    m.limits['c'] = (0, 5)
    m.limits['d'] = (-5, 5)
    m.limits['k'] = (0, 10)

    m.fixed['tmin'] = True
    #m.fixed['c'] = True
    #m.fixed['d'] = True


    # Recommended at first: fix the transition time.
    # Once the model behaves sensibly, you can comment this out.
    m.fixed['t_switch'] = True

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

    BIC = (2 * m.fval) + (np.log(len(se_roi)) * m.nfit)

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

    se_times = se_roi['time_since_start']

    fdt = 2.3

    # For fair comparison with your old 'new' model, start with 5*fdt.
    # You can later test 3*fdt.
    tmin = 5 * fdt

    c1 = cost.ExtendedUnbinnedNLL(
        se_times,
        lambda t, s, n, tau, f_exp, tmin, c, d, k: to_fit_exp_additive(
            t, s, n, tau, f_exp, tmin, c, d, k,
            s2_roi, s1_roi
        )
    )

    m = Minuit(
        c1,
        s=0.1,
        n=1.7,
        tau=10.0,
        f_exp=0.2,
        tmin=tmin,
        c=0.8,
        d=1.5,
        k=0.02
    )

    m.limits['s'] = (0, None)
    m.limits['n'] = (1.2, 5.0)
    m.limits['tau'] = (0.2, 50.0)
    m.limits['f_exp'] = (0.0, 0.5)
    m.limits['c'] = (0.0, 5.0)
    m.limits['d'] = (-5.0, 5.0)
    m.limits['k'] = (0.0, 10.0)

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

    BIC = (2 * m.fval) + (np.log(len(se_roi)) * m.nfit)

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

    print(f"\nThe amount of single electrons in the region of interest is: {len(se_roi)}")

    return values, errors, m.covariance, BIC

#------------------------------------------------------------------------------------------------------------

def cdf_plot(s2_roi, se_roi, s1_roi, values, cov, model = 'new',
             seconds_range = None, ax = None, plot_zoom = (0, 0)):
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
            lambda p: multi_exp_additive_wrap(t_model, p, s2_roi, s1_roi)[1],
            values, cov
        )
        model_errors_prop = np.sqrt(np.diag(model_errors))
    else:
        model_rate, model_errors = propagate(
            lambda p: multi_powerlaw_wrap(t_model, p, s2_roi, s1_roi)[1],
            values, cov
        )
        model_errors_prop = np.diag(model_errors)**0.5
    t_abs = np.linspace(time_start_ms, time_stop_ms, len(se_times_zoom) * 10)
    if model == 'exp':
        total_rate, p = multi_exp_powerlaw_wrap(
            t_abs,
            values,
            s2_roi,
            s1_roi,
            window_start_ms,
            window_stop_ms
        )
    elif model == 'exp_additive':
        total_rate, p = multi_exp_additive_wrap(
            t_abs,
            values,
            s2_roi,
            s1_roi
        )
    else:
        total_rate, p = new_power_law_pdf(
            t_abs, values[0], values[1], values[2], values[3],
            values[4], values[5], s2_roi, s1_roi
        )

    t_plot = t_abs - plot_shift_ms  # plot in relative zoom coordinates

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(15, 5))
        new_ax = True
    else:
        new_ax = False

    ax.hist(se_times_plot, bins=bin_edges_power_sum, color='k',
            histtype='step', label='Observed SEs')

    ax.errorbar(bin_centers_plot, hist_power_sum,
                yerr=np.sqrt(hist_power_sum), fmt='ko', markersize=0.4)

    for s2_time in s2_roi['time_since_start']:
        s2_plot = s2_time - plot_shift_ms
        if x_axis_left <= s2_plot <= x_axis_right:
            ax.axvline(s2_plot, color='g', linestyle='--', alpha=0.5,
                       label='S2 peak' if 'S2 peak' not in ax.get_legend_handles_labels()[1] else "")

    ax.errorbar(
        bin_centers_plot,
        model_rate * bin_width,
        yerr=model_errors_prop * bin_width,
        fmt='ro',
        markersize=0.5,
        label='Exp-to-power-law fit' if model == 'exp' else 'Power-law fit'
    )

    ax.plot(t_plot, p * bin_width, color='r')

    ax.set_xlim(0, x_axis_right)
    ax.set_xlabel("Time since window start (ms)", fontsize=14)
    ax.set_ylabel("Counts", fontsize=14)
    ax.legend(fontsize='medium', loc='best')
    ax.ticklabel_format(useOffset=False, style='plain', axis='x')

    if new_ax:
        plt.show()

    return total_rate, p

#-----------------------------------------------------------------------------------------------------------

#Functions below here are internal, not really intended to be called directly

#These functions here just kind of as wrappers for other stuff
#cost.Extended... needs a function to be passed in to work, but it's finicky
#Kind of the same deal for using 'propagate', just a slightly different form.

def to_fit(t, s, n, tmin, c, d, k, s2_roi, s1_roi, model = 'new'):
    return new_power_law_pdf(t, s, n, tmin, c, d, k, s2_roi, s1_roi, model = model)
#-----------------------------------------------------------------------------------------------------------

def multi_powerlaw_wrap(t, p, s2_roi, s1_roi, model = 'new'):
    s, n, tmin, c, d, k = p
    return new_power_law_pdf(t, s, n, tmin, c, d, k, s2_roi, s1_roi, model = model)

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

    return exp_power_law_pdf(
        t,
        s, n, tau, t_switch, tmin, c, d, k,
        s2_roi,
        s1_roi,
        window_start_ms,
        window_stop_ms
    )

# -----------------------------------------------------------------------------------------------------------

def to_fit_exp_additive(t, s, n, tau, f_exp, tmin, c, d, k, s2_roi, s1_roi):
    return new_exp_additive_pdf(
        t, s, n, tau, f_exp, tmin, c, d, k,
        s2_roi, s1_roi
    )

# -----------------------------------------------------------------------------------------------------------

def multi_exp_additive_wrap(t, p, s2_roi, s1_roi):
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
        s1_roi
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

@njit(cache=False, parallel=True, fastmath=True)
def _powerlaw_pdf_basic_consistent(t_grid, s, n, tmin, c, d, k,
                                   A, r0, r_p, s2_t_sorted, s2_area_sorted, s2_rng_sorted, 
                                   s2_r_sorted, s1_t_sorted, model_flag):
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
        if _in_dead_s2(ti, s2_t_sorted, tmin) or _in_dead_s1(ti, s1_t_sorted):
            diff[i] = 0.0
            continue

        acc = 0.0
        for j in range(s2_t_sorted.size):
            dt = ti - s2_t_sorted[j]
            if dt > tmin:
                acc += norms[j] * scale * (dt / tmin)**(-n)

        diff[i] += acc

    # Correction term (as in GOF)
    cut = np.zeros(s2_t_sorted.size)
    for j in range(s2_t_sorted.size):
        up = _cdf_scalar(s2_t_sorted[j] + tmin, tmin, n)
        lo = _cdf_scalar(s2_t_sorted[j],         tmin, n)
        cut[j] = norms[j] * (up - lo)

    # Live time (same approximation as GOF)
    total_time = t_grid[-1] - t_grid[0]
    live = total_time - s2_t_sorted.size * tmin - s1_t_sorted.size * 4.6
    if live < 0.0:
        live = 0.0

    total_rate = norms.sum() - cut.sum() + k * live
    return total_rate, diff

def new_power_law_pdf(t_grid, s, n, tmin, c, d, k,
                      pS2s_struct, s1_times_ms,
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

    # Sort S1 times
    if s1_times_ms is not None and len(s1_times_ms) > 0:
        s1_sorted = np.ascontiguousarray(np.sort(s1_times_ms.astype(np.float64)))
    else:
        s1_sorted = np.zeros(0, dtype=np.float64)

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
        s2_t_sorted, s2_area_sorted, s2_rng_sorted, s2_r_sorted,
        s1_sorted,
        model_flag
    )
@njit(cache=False, parallel=True, fastmath=True)
def _exp_powerlaw_pdf_basic_consistent(
    t_grid,
    s, n, tau, t_switch, tmin, c, d, k,
    A, r0, r_p,
    s2_t_sorted, s2_area_sorted, s2_rng_sorted, s2_r_sorted,
    s1_t_sorted,
    model_flag,
    window_start_ms,
    window_stop_ms
):
    """
     Exponential-near-start + power-law-late model.

     λ(t) = k + Σ_p norm_p * g(t - t_p)

     where norm_p is either basic.
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

        if _in_dead_s2(ti, s2_t_sorted, tmin) or _in_dead_s1(ti, s1_t_sorted):
            diff[i] = 0.0
            continue

        acc = 0.0

        for j in range(s2_t_sorted.size):
            dt = ti - s2_t_sorted[j]

            if dt > tmin:
                acc += norms[j] * _hybrid_kernel_value_normed(dt,n,tau,tmin,t_switch)

        diff[i] += acc

    # Integrated expected event count over the fit window
    total_time = window_stop_ms - window_start_ms
    live = total_time - s2_t_sorted.size * tmin - s1_t_sorted.size * 4.6

    if live < 0.0:
        live = 0.0

    total_rate = k * live

    for j in range(s2_t_sorted.size):
        u0 = window_start_ms - s2_t_sorted[j]
        u1 = window_stop_ms - s2_t_sorted[j]

        total_rate += norms[j] * _hybrid_kernel_integral_normed(u0,u1,n,tau,tmin,t_switch)

    return total_rate, diff
def exp_power_law_pdf(t_grid, s, n, tau, t_switch, tmin, c, d, k, pS2s_struct, s1_times_ms,window_start_ms, window_stop_ms,
                      A = None, r0 = None,
                      r_p = None, model = 'exp'):
    """
    Wrapper for the Numba-compiled exponential-to-power-law evaluator.

    This leaves new_power_law_pdf untouched.
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
        float(window_start_ms),
        float(window_stop_ms),
    )
# defining how we go about integrating for a hybrid exponential+power law

@njit(cache=False)
def _hybrid_kernel_total_integral(n, tau, tmin, t_switch):
    """
    Integral of the unnormalized hybrid kernel over [tmin, infinity).
    """

    M = (t_switch ** (-n)) * np.exp((t_switch - tmin) / tau)

    # exponential part: tmin to t_switch
    exp_int = M * tau * (
        1.0 - np.exp(-(t_switch - tmin) / tau)
    )

    # power-law part: t_switch to infinity
    pl_int = (t_switch ** (1.0 - n)) / (n - 1.0)

    return exp_int + pl_int

@njit(cache=False)
def _hybrid_kernel_value_normed(dt, n, tau, tmin, t_switch):
    if dt <= tmin:
        return 0.0

    I = _hybrid_kernel_total_integral(n, tau, tmin, t_switch)
    if I <= 0.0:
        return 0.0

    M = (t_switch ** (-n)) * np.exp((t_switch - tmin) / tau)

    if dt < t_switch:
        g = M * np.exp(-(dt - tmin) / tau)
    else:
        g = dt ** (-n)

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
    M = (t_switch ** (-n)) * np.exp((t_switch - tmin) / tau)

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
            total += (pl_lo ** (1.0 - n) - pl_hi ** (1.0 - n)) / (n - 1.0)

    return total

@njit(cache=False)
def _hybrid_kernel_integral_normed(u0, u1, n, tau, tmin, t_switch):
    raw = _hybrid_kernel_integral(u0, u1, n, tau, tmin, t_switch)

    I = _hybrid_kernel_total_integral(n, tau, tmin, t_switch)
    if I <= 0.0:
        return 0.0

    return raw / I
@njit(cache=False)
def _hybrid_kernel_value(dt, n, tau, tmin, t_switch):
    if dt <= tmin:
        return 0.0

    M = (t_switch ** (-n)) * np.exp((t_switch - tmin) / tau)

    if dt < t_switch:
        return M * np.exp(-(dt - tmin) / tau)

    return dt ** (-n)
@njit(cache=False, parallel=True, fastmath=True)
def _exp_additive_pdf_basic_consistent(t_grid, s, n, tau, f_exp, tmin,
                                       c, d, k,
                                       A, r0, r_p,
                                       s2_t_sorted, s2_area_sorted, s2_rng_sorted,
                                       s2_r_sorted, s1_t_sorted, model_flag):
    """
    Core additive exponential + power-law evaluator.

    For each S2 and dt > tmin:
        rate_j(dt) = norm_j * [
            f_exp * exp_kernel(dt)
            + (1 - f_exp) * powerlaw_kernel(dt)
        ]
    """

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

    diff = np.full(t_grid.size, k)

    # Guard invalid parameters
    if tau <= 0.0 or n <= 1.0 or f_exp < 0.0 or f_exp > 1.0:
        for i in prange(t_grid.size):
            diff[i] = 0.0
        return 0.0, diff

    exp_scale = 1.0 / tau
    power_scale = (n - 1.0) / tmin

    for i in prange(t_grid.size):
        ti = t_grid[i]

        if _in_dead_s2(ti, s2_t_sorted, tmin) or _in_dead_s1(ti, s1_t_sorted):
            diff[i] = 0.0
            continue

        acc = 0.0

        for j in range(s2_t_sorted.size):
            dt = ti - s2_t_sorted[j]

            if dt > tmin:
                exp_kernel = exp_scale * np.exp(-(dt - tmin) / tau)
                power_kernel = power_scale * (dt / tmin) ** (-n)

                kernel = f_exp * exp_kernel + (1.0 - f_exp) * power_kernel

                acc += norms[j] * kernel

        diff[i] += acc

    # Numerically integrate over this actual grid, consistent with plotting
    total_rate = 0.0
    for i in range(t_grid.size - 1):
        dt_grid = t_grid[i + 1] - t_grid[i]
        total_rate += 0.5 * (diff[i + 1] + diff[i]) * dt_grid

    return total_rate, diff
def new_exp_additive_pdf(t_grid, s, n, tau, f_exp, tmin, c, d, k,
                         pS2s_struct, s1_times_ms,
                         A=None, r0=None, r_p=None,
                         model='new'):
    """
    Additive exponential + power-law model.

    For dt > tmin:
        kernel = f_exp * exponential_kernel + (1 - f_exp) * powerlaw_kernel
    """

    # Extract S2 fields
    s2_t = pS2s_struct['time_since_start'].astype(np.float64)
    s2_area = pS2s_struct['area'].astype(np.float64)
    s2_rng  = (pS2s_struct['range_50p_area'] / 1e6).astype(np.float64)

    if 'r' in pS2s_struct.dtype.names:
        s2_r = pS2s_struct['r'].astype(np.float64)
    else:
        s2_r = np.zeros_like(s2_t)

    # Sort S2s
    order = np.argsort(s2_t)
    s2_t_sorted    = np.ascontiguousarray(s2_t[order])
    s2_area_sorted = np.ascontiguousarray(s2_area[order])
    s2_rng_sorted  = np.ascontiguousarray(s2_rng[order])
    s2_r_sorted    = np.ascontiguousarray(s2_r[order])

    # Sort S1s
    if s1_times_ms is not None and len(s1_times_ms) > 0:
        s1_sorted = np.ascontiguousarray(np.sort(s1_times_ms.astype(np.float64)))
    else:
        s1_sorted = np.zeros(0, dtype=np.float64)

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
        s2_t_sorted, s2_area_sorted, s2_rng_sorted, s2_r_sorted,
        s1_sorted,
        model_flag
    )
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