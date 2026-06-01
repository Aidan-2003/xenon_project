"""
This file was developed to contain functions for data selection of pS2s and DEs from raw straxen peaks data.

You can use it one of two ways:
1. Be on Midway/wherever, input the run_id, and obtain the peaks data along with the selected pS2s and DEs etc.

2. If you already have the peaks data loaded but not processed/selected, you can pass peaks through and get the selection that way
You will also need the run information however, since some fields are added based on the run start time
"""

# import strax
import straxen
#import cutax
import numpy as np
from numpy.lib import recfunctions as rfn
import matplotlib.pyplot as plt
import sys
import getpass
user = getpass.getuser()

#sys.path.append('/home/ford2/project_code/plugins')
#sys.path.append

#st = cutax.xenonnt_offline(xedocs_version = "global_v12", #xedocs version not being recognised?
                           #output_folder = f'/scratch/midway2/{user}/strax_data')
st = straxen.contexts.xenonnt_online()
from SubTyping_Class import PeaksSubtypes
from Wrong_pS2_relabel import PS2_relabel

from count_ne import PiecewiseInfo, CountNElectron

plugins_to_register = [PeaksSubtypes, PS2_relabel, PiecewiseInfo, CountNElectron]
for i in plugins_to_register:
    st.register(i)
    print(i.provides, i.__version__)
#
# import importlib
# import helper_functions as hf
# import modelling_July as md
# importlib.reload(hf)
# importlib.reload(md)

def data_selection_ku(run_id, peaks = None, show_plot = True, drop_columns = True):
    """
    This is the original method of data selection used by Keerthana Umesh in her bachelor's thesis project.
    It uses the straxen mini-analysis of "plots_aft_histogram" to plot an area vs range_50p_area histogram
    with area_fraction_top on the colourbar. From this, Keerthana defined regions in which to select the different
    pS2, 1e, and ne populations.

    Note that this function does not apply prompt electron cuts

    Inputs:
     - run_id: Structured NumPy array with 'name' of run,'start' etc.

    Outputs:
     - pS2s_original: self explanatory
     - DEs_original: combined 1e and ne populations
     - DEs_original_cut: DEs after prompt electron cut
     - peaks_amended: peaks array with extra fields added
    """

    global run_start #Necessary? Idk
    run_start = int(run_id['start'].value)
    run_end = int(run_id['end'].value)

    if peaks is None:
        if drop_columns:
            columns_to_drop = ['range_90p_area', 'rise_time', 'tight_coincidence', 'max_diff', 'min_diff', 
                               'max_pmt_area', 'x_cnn', 'y_cnn', 'x_gcn', 'y_gcn', 'n_competing',
                               'n_competing_left', 't_to_prev_peak', 't_to_next_peak', 't_to_nearest_peak', 
                               'channel', 'area_decile_from_midpoint', 'saturated_channel', 'data_top', 
                               'max_gap', 'max_goodness_of_split']
            peaks = st.get_array(run_id['name'],
                                 targets = ['peaks', 'peak_basics', 'peak_proximity', 
                                            'peak_positions', 'subtype_mask', 'pS2_relabel', 'n_electron_rec'],
                                 drop_columns = columns_to_drop)
        else:
            peaks = st.get_array(run_id['name'],
                                 targets = ['peaks', 'peak_basics', 'peak_proximity', 
                                            'peak_positions', 'subtype_mask', 'pS2_relabel', 'n_electron_rec'])

        time_since_start = (peaks['time'] - run_start) / int(1e6) #Actual time since start of run in ms
        r = np.sqrt(peaks['x']**2 + peaks['y']**2)
        theta = np.arctan2(peaks['y'], peaks['x'])
        peaks_amended = rfn.append_fields(peaks, ['time_since_start', 'r', 'theta'], 
                                  [time_since_start, r, theta], usemask = False)
    
    else:
        if 'time_since_start' not in peaks.dtype.names:
            time_since_start = (peaks['time'] - run_start) / int(1e6) #Actual time since start of run in ms
            r = np.sqrt(peaks['x']**2 + peaks['y']**2)
            theta = np.arctan2(peaks['y'], peaks['x'])
            peaks_amended = rfn.append_fields(peaks, ['time_since_start', 'r', 'theta'], 
                                      [time_since_start, r, theta], usemask = False)
        else:
            print("Peaks all good")
            peaks_amended = peaks
        
    def plot_selection(xrange, yrange, axes, col, label):
        """Draws the selection as a filled in box on any plot
        
        xrange = tuple of min and max x-axis values from plot from which data is selected
        yrange = tuple of min and max y-axis values from plot from which data is selected
        axes = plot on which selection is to be visualized
        col = color of the filled in box
        """
        low_x, high_x = xrange
        low_y, high_y = yrange
        x = np.arange(low_x, high_x, 50)
        y = np.full(len(x), low_y)
        y2 = np.full(len(x), high_y)
        axes.fill_between(x, y, y2, alpha = 0.5, color = col)

        mid_x = np.sqrt(low_x * high_x)
        mid_y = np.sqrt(low_y * high_y)

        axes.text(mid_x, mid_y, label, ha = 'center', 
                va = 'center', fontsize = 15, color = 'black')
        return axes

    def select_area(peaks, xrange, yrange, params = ("area", "range_50p_area")):
        """
        Makes a selection based on x and y range array based on any two parameters

        peaks = input array of peaks
        xrange = tuple of min and max x-axis values from plot from which data is selected
        yrange = tuple of min and max y-axis values from plot from which data is selected
        params = tuple of column titles in input peaks array of the parameters to be used for selection
        """
        low_x,high_x = xrange
        low_y,high_y = yrange
        param1,param2 = params
        selects = peaks[
        (peaks[param1] > int(low_x))
        & (peaks[param1] < int(high_x))
        & (peaks[param2] > low_y)
        & (peaks[param2] < high_y)]
        return selects

    if show_plot:
        hf.plot_histogram(peaks_amended)

        ax = plt.gca()
        
        plot_selection((3e4, 1e7), (1.5e2, 2.5e4), ax, "yellow", label = "pS2")
        plot_selection((8e0, 1.5e2), (1.2e2, 6e2), ax, "green", label = "1e")
        plot_selection((3e1, 3e2), (6e2, 2e3), ax, "blue", label = "ne")

    s2s = select_area(peaks_amended, (3e4, 1e7), (1.5e2, 2.5e4))
    ses1 = select_area(peaks_amended, (8e0, 1.5e2), (1.2e2, 6e2))  
    ses2 = select_area(peaks_amended, (3e1, 3e2), (6e2, 2e3)) 

    pS2s_original = s2s
    DEs_original = np.concatenate((ses1, ses2))

    DEs_original_cut = prompt_electron_cut(pS2s_original, DEs_original, selection = 'old')
    return pS2s_original, DEs_original, DEs_original_cut, peaks_amended

def data_selection_new(run_id, peaks = None, show_plots = False, drop_columns = True, fiducial_cut = False):
    """
    This is the main function for the selection and subtyping of data. A lot is being done in this function,
    so have tried to split it up as best as possible.

    Inputs:
     - run_id: Structured NumPy array with 'name' of run, among other things
    The way it's being found should be runs = st.get_runs(...) -> then run_id = runs.iloc[0] or whatever
    This means it's not just a string of the run name, but a dataframe/structured array that contains all the info about the run
     - show_plot: I mean come on
    
    Outputs: 
    - pS2s: numpy structured array of pS2s, selected according to subtypes
    - DEs_cut: "" for DEs, prompt electrons cut out, along with all the other cuts
    - S1s: "" S1s ""
    - peaks: unprocessed peaks structured array from st.get_array(...), bit cumbersome but extremely useful for other analysis stuff
    - vetos: array of DAQ veto start and end times
    
    I've hopefully set this up so that later I can put in a list of runs and get everything out, we'll see
    That also might be quite computationally expensive idk though
    Future me don't shoot me in the head if this becomes a problem please
    """

    print("\n" + "-" * 115 + "\n")

    print(f"Now looking at run: {run_id['name']}")

    global run_start #Necessary?
    run_start = int(run_id['start'].values[0])
    run_end = int(run_id['end'].values[0])

    #Get peaks data, add some fields for extra info
    if peaks is None:
        if drop_columns:
            columns_to_drop = ['range_90p_area', 'rise_time', 'tight_coincidence', 'max_diff', 'min_diff', 
                               'max_pmt_area', 'x_cnn', 'y_cnn', 'x_gcn', 'y_gcn', 'n_competing',
                               'n_competing_left', 't_to_prev_peak', 't_to_next_peak', 't_to_nearest_peak', 
                               'channel', 'area_decile_from_midpoint', 'saturated_channel', 'data_top', 
                               'max_gap', 'max_goodness_of_split']
            peaks = st.get_array(run_id['name'],
                                 targets = ['peaks', 'peak_basics', 'peak_proximity', 
                                            'peak_positions', 'subtype_mask', 'pS2_relabel', 'n_electron_rec'],
                                 drop_columns = columns_to_drop)
        else:
            peaks = st.get_array(run_id['name'],
                                 targets = ['peaks', 'peak_basics', 'peak_proximity', 
                                            'peak_positions', 'subtype_mask', 'pS2_relabel', 'n_electron_rec'])

        """
        Late in development I've noticed that this actually results in a few peaks which occur *before*
        the start of the run, meaning the registered run_id['start'] number is maybe rounded or something
        So TODO if time: Readjust so time_since_start is from the first 'peaks' signal
        Not going to do as priority since I'd have to spend ages reprocessing all the other data, 
        was going to take time ranges from later in the run regardless
        """
        time_since_start = (peaks['time'] - run_start) / int(1e6) #Actual time since start of run in ms
        r = np.sqrt(peaks['x']**2 + peaks['y']**2)
        theta = np.arctan2(peaks['y'], peaks['x'])
        peaks_amended = rfn.append_fields(peaks, ['time_since_start', 'r', 'theta'], 
                                  [time_since_start, r, theta], usemask = False)
    else:
        if 'time_since_start' not in peaks.dtype.names:
            print("Adding necessary fields")
            time_since_start = (peaks['time'] - run_start) / int(1e6)
            r = np.sqrt(peaks['x']**2 + peaks['y']**2)
            theta = np.arctan2(peaks['y'], peaks['x'])
            peaks_amended = rfn.append_fields(peaks, ['time_since_start', 'r', 'theta'], 
                                    [time_since_start, r, theta], usemask=False)
        else:
            print("Peaks all good")
            peaks_amended = peaks
    

    #Collect DAQ veto intervals if they exist
    if st.is_stored(run_id['name'].values[0], 'veto_intervals'):
        vetos = st.get_array(run_id['name'].values[0], 'veto_intervals')

        start_time_ms = (vetos['time'] - int(run_id['start'].value)) / int(1e6)
        end_time_ms = (vetos['endtime'] - int(run_id['start'].value)) / int(1e6)

        vetos = rfn.append_fields(vetos, ['start(ms)', 'end(ms)'], 
                                  [start_time_ms, end_time_ms], usemask = False)
    else:
        print("DAQ veto intervals not found, oop")
        vetos = None
        # if 'time_since_start' not in peaks.dtype.names:
        #     r = np.sqrt(peaks['x']**2 + peaks['y']**2)
        #     theta = np.arctan2(peaks['y'], peaks['x'])
        #     peaks = rfn.append_fields(peaks, ['r', 'theta'], [r, theta], usemask=False)


    #---------------------------------------------------------------------------------------------------

    """
    Section for selection of pS2s, DEs, and S1s based on peak subtype.

    See the below notes to understand what this subtyping does, and what they mean:
    https://xe1t-wiki.lngs.infn.it/doku.php?id=jlong:peak_subtyping_study
    https://xe1t-wiki.lngs.infn.it/doku.php?id=jlong:peak_subtyping_dictionary

    See this note to understand the need for the relabelling, done in the section below:
    https://xe1t-wiki.lngs.infn.it/doku.php?id=yongyu:relabel_ps2_subtype_plugin

    For the avoidance of confusion however, the subtypes are briefly explained below:

    - 10: isoS1 - S1 without nearby peaks
    - 11: S1 - Regular S1 with a matched pS2
    - 12: loneS1 - Regular S1 without a matched pS2
    - 13: sloneS1 - Regular S1 without a matched pS2, with area < 0.5SE
    - 21: DE - Delayed Extracted few electron peak
    - 22: pS2 - S2 matched with an S1, with area >= 5SE
    - 23: isoDE - S2 without nearby peaks, with area < 5SE - not yet implemented
    - 24: isopS2 - S2 without nearby peaks, with area >= 5SE - not yet implemented
    - 25: lonepS2 - S2 with area >= 5SE, without a matched S1 but not categorized as other large S2s.
    - 261: S1olS2 - S2 with a nearby pS2, with area >= max(0.5*pS2, 5SE). After S1 but before S2
    - 262: S2olS2 - S2 with a nearby pS2, with area >= max(0.5*pS2, 5SE). After S2
    - 271: S1PH - Photoionization S2s, with area < max(0.5*pS2, 5SE), after S1 but before S2
    - 272: S2PH - Photoionization S2s, with area < max(0.5*pS2, 5SE), after S2
    - 273: slS1PH - Photoionization S2s after a sloneS1
    - 28: fakeS2 - S2 with area < 5SE, with a paired pS2 satisfying S1-S2 time correlation
    - 29: fakeS2 olS2 - olS2 after the fakeS2 and before the associated pS2
    - 20: fakeS2 PH - Photoionization after the fakeS2 and before the associated pS2

    From this we can see that pS2s are either the subtypes selected below, basically either a normal pS2, 
    a pS2 without a matched S1, or a pS2 that is relatively large, but close to a bigger pS2 (likely a double scatter).

    DEs actually fall under a fair amount of categories, but since many are forms of photoionisation 
    (defined within the study as having occurred within a few drift times of an S1 or large S2), 
    we can just save some effort in the prompt electron cut by not selecting them here.

    I believe some true pS2s are lost in this selection, as 271 and 272 can contain some very large signals by area.
    Could come back to this later and keep only 271s and 272s above a certain area threshold, for example.
    """

    pS2s = peaks_amended[np.isin(peaks_amended['subtype'], [22, 24, 25, 261, 262, 29])]

    DEs = peaks_amended[np.isin(peaks_amended['subtype'], [20, 21, 23, 271, 272, 273])]

    S1s = peaks_amended[np.isin(peaks_amended['subtype'], [10, 11, 12, 13, 28])] #, 28

    #---------------------------------------------------------------------------------------------------

    #TODO: come back and check all the logic of this section
    #TODO: Also need to explain a bit about why it's being done at all

    before = len(pS2s[pS2s['pS2_wrong_pairing'] == True])
    
    #Just going through and keeping all the pS2s that are properly subtyped/labelled
    pS2s['subtype'][pS2s['pS2_relabel_S2PH']] = 272
    pS2s['subtype'][pS2s['pS2_relabel_S2OLS2']] = 262

    reassigned = np.isin(pS2s['subtype'], [272])

    added_DEs = pS2s[reassigned]
    pS2s = pS2s[~reassigned]

    all_DEs = np.concatenate((DEs, added_DEs))
    all_DEs = all_DEs[np.argsort(all_DEs['time'])] #Making sure they're put back in order again
    
    """
    'pS2_wrong_pairing' is this broad classification that covers both of the other two but also some other peaks.
    Until I understand more about what subtypes these peaks might be or what I've decided just to cut them.
    The other two classifications are dealt with above however, so those should be fine
    """
    pS2s = pS2s[(pS2s['pS2_relabel_S2PH']) | 
                        (pS2s['pS2_relabel_S2OLS2']) | 
                        (~pS2s['pS2_wrong_pairing'])]

    pS2s = (pS2s[~((pS2s['range_50p_area'] > 4e4))]) #roughly remove e-bursts

    print(f"\n{before} values reassigned from pS2s into all_DEs (or cut entirely)")

    #---------------------------------------------------------------------------------------------------

    #Prompt electron cut. Also going to cut for 5 fdt (maybe less) for after an S1 to remove photoionisation electrons

    remaining_DEs = prompt_electron_cut(pS2s, all_DEs, S1s)

    #---------------------------------------------------------------------------------------------------

    """
    Final cut based on area and n_electron_rec. This is the most arbitrary cut, mostly just a common-sense check
    We keep pS2s with n_electron_rec = 0, since they can be real pS2s that just weren't counted properly,
    however DEs with n_electron_rec = 0 are usually pile-up or dark-counts, so are removed. #TODO: is this true?
    """

    remaining_DEs_2_electric_boogaloo = remaining_DEs[~((remaining_DEs['area'] > 500) | (remaining_DEs['n_electron_rec'] > 5))]

    DEs_left = remaining_DEs_2_electric_boogaloo #just renaming

    #Want to reassign any DEs_left with n_electron_rec = 0 to n_electron_rec = 1
    DEs_left['n_electron_rec'][DEs_left['n_electron_rec'] == 0] = 1
    DEs_left = DEs_left[np.argsort(DEs_left['time'])]

    #---------------------------------------------------------------------------------------------------

    #Just a wrap-up bit

    run_length = (int(run_end) - int(run_start)) / int(1e9)
    
    print(f"\nLength of run: {run_length:.2f}s")
    print(f"Loaded {(peaks_amended.nbytes) / 1e6:.1f} MB of peaks-data "
          f"({(pS2s.nbytes + DEs_left.nbytes) / 1e6:.1f} MB of which are pS2s and DEs)")

    if show_plots:
        #This is a terrible plot, need to fix TODO
        _, ax = hf.plot_histogram(peaks_amended)

        ax.scatter(pS2s['area'], pS2s['range_50p_area'], color = 'blue', 
                    marker = 'x', label='pS2s', alpha = 0.8, s = 5)

        ax.scatter(DEs_left['area'], DEs_left['range_50p_area'], color = 'green', 
                    marker = 'x', label = 'DEs', alpha = 0.8, s = 5)

        ax.legend()
        plt.show()

    pS2s_final, DEs_final = Other_quality_cuts(run_id, pS2s, DEs_left, show_plots = show_plots, fiducial_cut = fiducial_cut)

    diagnose_selection(peaks_amended, pS2s_final, DEs_final, S1s)

    a_ms = 39545.981
    b_ms = 39555.981

    a_ns = int(run_start + a_ms * 1e6)
    b_ns = int(run_start + b_ms * 1e6)

    st.plot_peaks(run_id["name"].values[0], time_range=(a_ns, b_ns))
    plt.show()

    return pS2s_final, DEs_final, S1s, peaks_amended, vetos

def prompt_electron_cut(pS2s, all_DEs, S1s = None, selection = 'new'):
    """
    Cuts all DEs that occur within a certain time window after an S1 or pS2.

    S1s = structured array of S1 peaks
    pS2s = structured array of pS2 peaks
    all_DEs = structured array of DE peaks

    Outputs:
        - remaining_DEs = structured array of DE peaks that have passed the cut
    """
    if S1s is None and selection == 'new':
        raise ValueError("Must provide S1s for 'new' selection prompt electron cut")
        #Also if you see this you've probably called this function externally and that's on you tbh

    if S1s is None and selection == 'old':
        print("No S1s provided, proceeding with 'old' selection prompt electron cut")
        s1_times = np.array([])
    else:
        s1_times = S1s['time']  
    
    s2_times = pS2s['time']
    de_times = all_DEs['time']

    assigned_indices_s1 = np.searchsorted(s1_times, de_times) -1
    assigned_indices_s2 = np.searchsorted(s2_times, de_times) -1
    #"-1" removes SEs at beginning of run since otherwise would assign it to the S1/S2 at end of run

    has_prev_s2 = assigned_indices_s2 >= 0
    assigned_ses = all_DEs[has_prev_s2]
    prev_s2_times = s2_times[assigned_indices_s2[has_prev_s2]]
    dt_to_prev_s2 = assigned_ses['time'] - prev_s2_times

    # For the SAME assigned DEs, find previous S1 (may or may not exist)
    prev_s1_idx = assigned_indices_s1[has_prev_s2]   # align lengths
    has_prev_s1 = prev_s1_idx >= 0

    # Initialise with +inf so "no S1" doesn’t cut them
    dt_to_prev_s1 = np.full(len(assigned_ses), np.inf, dtype=np.int64)
    dt_to_prev_s1[has_prev_s1] = assigned_ses['time'][has_prev_s1] - s1_times[prev_s1_idx[has_prev_s1]]

    # # Apply cut
    fdt = 2.3e6 # 1 full drift time in ns

    if selection == 'old':
        cut_window = 3 * fdt
        keep_mask = dt_to_prev_s2 > cut_window
    else: #selection == 'new'
        cut_window_s1 = 2 * fdt
        cut_window_s2 = 5 * fdt
        keep_mask = (dt_to_prev_s2 > cut_window_s2) & (dt_to_prev_s1 > cut_window_s1)

    remaining_DEs = assigned_ses[keep_mask]

    print(f"{len(all_DEs) - len(remaining_DEs)} single- or few-electron signals cut, "
            f"\nrepresenting {(1 - (len(remaining_DEs)/len(all_DEs))) * 100:.2f}% of all single- or few-electron signals")
    return remaining_DEs


def Other_quality_cuts(run_id, pS2s, DEs, show_plots = False, fiducial_cut = False):
    """
    Just a couple of miscellaneous quality cuts on both pS2s and DEs.

    First:
    Couple of things we want to remove from the DEs, mostly unphysical ones. Not many, but still important

    See this note for some justification:
    https://xe1t-wiki.lngs.infn.it/doku.php?id=jlong:sr0_update_se_selection_and_result

    We're not interested in everything in this note, for example because it tries to remove pile-ups, 
    events in which we may also be interested, and therefore want to keep.
    It also imposes a variety of area, width, and time window restrictions on the initial selection,
    and since our model does not care about ne- signals or pS2 windows, we're not interested in these selection cuts either

    Steps are:
    1. Remove any DEs where the maximum contributing PMT is a bottom one - characteristic of dark count pile-ups
    2. Remove any DEs where the area fraction top is > 0.99, or < max_pmt area / total area + 0.01 - bad waveforms

    All total this removes a fair percentage of DEs from the selection, I believe 
    (although these numbers are pre a fiducial volume cut, so certainly less here)
    """

    before = len(DEs)
    #1.
    ses1 = DEs[DEs["max_pmt"] < 253]

    #2.
    macr = ses1['area_per_channel'].max(axis = 1) / ses1['area']
    DEs_cut = ses1[(ses1['area_fraction_top'] < 0.99) & (ses1['area_fraction_top'] > macr - 0.01)] #What we actually want to keep

    #These signals have unphysical waveforms that oscillate into the negative region
    weird_DEs = ses1[ses1['area_fraction_top'] < macr + 0.01]

    high_aft = ses1[ses1['area_fraction_top'] >= 0.99]

    if show_plots:
        plt.figure(figsize = (10, 8))
        plt.hist2d(macr, ses1['area_fraction_top'], bins = 100, cmap = 'viridis', cmin = 1)
        plt.colorbar(label = 'Counts/bin')
        plt.axhline(0.99, color = 'red', linestyle = '--', label = 'Area Fraction Top = 0.99')

        x = np.linspace(0, 1, 1000)
        plt.plot(x, x + 0.01, color = 'red', linestyle = '--')
        plt.plot(x, x - 0.01, color = 'red', linestyle = '--')

        plt.xlabel('Max PMT Area / Total Area')
        plt.ylabel('Area Fraction Top')
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        n = 1
        indices1 = np.random.choice(len(weird_DEs), size = n, replace = False)
        indices2 = np.random.choice(len(high_aft), size = n, replace = False)
        for i in indices1, indices2:
            s1, s2 = weird_DEs[i], high_aft[i]
            start1, stop1, start2, stop2 = s1["time"], s1["endtime"], s2["time"], s2["endtime"]

            st.plot_peaks(run_id['name'], time_range = (start1, stop1))
            plt.title(
                f'S{s1["type"]} of ~{int(s1["area"])} PE (width {int(s1["range_50p_area"])} ns)\n'
                f'Area fraction top of {s1["area_fraction_top"]:.3f} \n'
                f'At a time of {s1["time"]}'
            )
            plt.tight_layout()

            st.plot_peaks(run_id['name'], time_range = (start2, stop2))
            plt.title(
                f'S{s2["type"]} of ~{int(s2["area"])} PE (width {int(s2["range_50p_area"])} ns)\n'
                f'Area fraction top of {s2["area_fraction_top"]:.3f} \n'
                f'At a time of {s2["time"]}'
            )
            plt.tight_layout()

    #------------------------------------------------------------------------------------------------------------

    """
    Also going to do the fiducial volume cut here, since it's just one line.
    Have chosen a radius of 43cm (currently) (Also did I do a study on this? Can't remember)

    Notes:
    https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt:analysis:ntsciencerun0:fiducial_volume
    https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:abby:fiducialization
    """

    #Fiducial volume cut:
    if fiducial_cut == True:
        DEs_cut = DEs_cut[DEs_cut['r'] < 45] #45cm inner radius being kept

    #----------------------------------------------------------------------------------------------------------

    #Also doing a quick cut on low-AFT pS2s. My investigation has found that these are generally extremely weird
    #and wide S2s that come immediately after a big S1. What this means I've forgotten I think, but I don't like 'em
    #TODO: Provide link to some notebook I have that shows why these are weird?

    #Have chosen 0.5 as cut value cause then it should be closer to the top like a normal S2. 
    #Even slightly above this value we start to see genuine pS2s that are just a bit low in AFT, so don't want to exclude those

    good_pS2s = pS2s[pS2s['area_fraction_top'] >= 0.5]
    
    #------------------------------------------------------------------------------------------------------------
    after = len(DEs_cut)

    print(f"{before - after} electrons cut, representing {((before - after) / before) * 100:.2f}% of all electrons")


    return good_pS2s, DEs_cut

##everything below here is functions I introduced to determine whether the selection in conors code makes sense or not
def diagnose_selection(peaks_amended, pS2s_final, DEs_final, S1s=None):
    """
    Diagnostic checks for possible DE/pS2 misclassification.

    Does not modify the arrays.
    Only prints summaries.
    """

    def print_subtype_counts(arr, name):
        print(f"\n{name}: {len(arr)} peaks")
        if len(arr) == 0:
            return

        vals, counts = np.unique(arr["subtype"], return_counts=True)
        for v, c in zip(vals, counts):
            print(f"  subtype {v}: {c}")

    print("\n" + "-" * 80)
    print("Selection diagnostics")
    print("-" * 80)

    print_subtype_counts(DEs_final, "Final DEs")
    print_subtype_counts(pS2s_final, "Final pS2s")

    # Small/few-electron-like candidates independent of subtype.
    small_candidates = peaks_amended[
        (peaks_amended["area"] <= 500)
        & (peaks_amended["n_electron_rec"] <= 5)
    ]

    selected_de_mask = np.isin(small_candidates["time"], DEs_final["time"])
    selected_ps2_mask = np.isin(small_candidates["time"], pS2s_final["time"])

    missed_small = small_candidates[
        ~selected_de_mask
        & ~selected_ps2_mask
    ]

    small_in_ps2s = small_candidates[selected_ps2_mask]

    print_subtype_counts(small_candidates, "All small candidates: area <= 500 and n_electron_rec <= 5")
    print_subtype_counts(small_in_ps2s, "Small candidates classified as pS2s")
    print_subtype_counts(missed_small, "Small candidates selected as neither DE nor pS2")

    # Suspicious DEs: large-ish or multi-electron.
    suspicious_des = DEs_final[
        (DEs_final["area"] > 150)
        | (DEs_final["n_electron_rec"] > 2)
        | np.isin(DEs_final["subtype"], [20, 271, 272, 273])
    ]

    print_subtype_counts(suspicious_des, "Suspicious final DEs: PH/fake or area>150 or n_electron_rec>2")

    if len(suspicious_des) > 0:
        print("\nSuspicious DE summary:")
        print("  area min/median/max:",
              np.min(suspicious_des["area"]),
              np.median(suspicious_des["area"]),
              np.max(suspicious_des["area"]))
        print("  n_electron_rec unique:",
              np.unique(suspicious_des["n_electron_rec"], return_counts=True))
        print("  range_50p_area ms min/median/max:",
              np.min(suspicious_des["range_50p_area"] / 1e6),
              np.median(suspicious_des["range_50p_area"] / 1e6),
              np.max(suspicious_des["range_50p_area"] / 1e6))

    print("-" * 80 + "\n")

def inspect_dense_bins(DEs, pS2s=None, bin_width_ms=10.0, threshold=5):
    """
    Inspect bins where many DE candidates occur within a short time span.
    Useful for checking whether high-count bins are dominated by certain subtypes.
    """

    if len(DEs) == 0:
        print("No DEs provided.")
        return

    t = DEs["time_since_start"]
    bins = np.arange(np.min(t), np.max(t) + bin_width_ms, bin_width_ms)
    counts, edges = np.histogram(t, bins=bins)

    dense_indices = np.where(counts >= threshold)[0]

    print(f"\nFound {len(dense_indices)} bins with >= {threshold} DEs in {bin_width_ms} ms bins.")

    for idx in dense_indices:
        a = edges[idx]
        b = edges[idx + 1]

        in_bin = DEs[(t >= a) & (t < b)]

        print("\n" + "-" * 70)
        print(f"Dense bin: {a:.3f} to {b:.3f} ms, N={len(in_bin)}")
        print("Subtype counts:", np.unique(in_bin["subtype"], return_counts=True))
        print("n_electron_rec counts:", np.unique(in_bin["n_electron_rec"], return_counts=True))
        print("times ms:", np.sort(in_bin["time_since_start"]))
        print("dt between DEs ms:", np.diff(np.sort(in_bin["time_since_start"])))
        print("areas:", in_bin["area"])
        print("range_50p_area ms:", in_bin["range_50p_area"] / 1e6)

    if pS2s is not None:
        near = pS2s[
            (pS2s["time_since_start"] >= a - 500.0)
            & (pS2s["time_since_start"] <= b + 50.0)
            ]

        print(f"Nearby pS2s from {a - 500:.1f} to {b + 50:.1f} ms: {len(near)}")
        if len(near) > 0:
            print("pS2 times ms:", near["time_since_start"])
            print("pS2 areas:", near["area"])
            print("pS2 subtypes:", np.unique(near["subtype"], return_counts=True))

def make_alternative_de_samples(DEs_final):
    """
    Create stricter/looser DE samples for robustness checks.
    """

    samples = {}

    samples["current"] = DEs_final

    samples["strict_subtype_21"] = DEs_final[
        DEs_final["subtype"] == 21
    ]

    samples["no_PH"] = DEs_final[
        ~np.isin(DEs_final["subtype"], [20, 271, 272, 273])
    ]

    samples["one_to_two_electron"] = DEs_final[
        (DEs_final["n_electron_rec"] >= 1)
        & (DEs_final["n_electron_rec"] <= 2)
    ]

    samples["small_area"] = DEs_final[
        (DEs_final["area"] <= 150)
        & (DEs_final["n_electron_rec"] <= 2)
    ]

    return samples


def inspect_dense_bins_with_prev_s2(DEs, pS2s, bin_width_ms=10.0, threshold=5):
    t = DEs["time_since_start"]
    bins = np.arange(np.min(t), np.max(t) + bin_width_ms, bin_width_ms)
    counts, edges = np.histogram(t, bins=bins)

    dense_indices = np.where(counts >= threshold)[0]
    s2_times = pS2s["time_since_start"]

    print(f"Found {len(dense_indices)} bins with >= {threshold} DEs in {bin_width_ms} ms bins.")

    for idx in dense_indices:
        a, b = edges[idx], edges[idx + 1]
        in_bin = DEs[(t >= a) & (t < b)]

        first_t = np.min(in_bin["time_since_start"])
        prev_s2_idx = np.searchsorted(s2_times, first_t) - 1

        print("\n" + "-" * 70)
        print(f"Dense bin: {a:.3f} to {b:.3f} ms, N={len(in_bin)}")
        print("Subtype counts:", np.unique(in_bin["subtype"], return_counts=True))
        print("n_electron_rec counts:", np.unique(in_bin["n_electron_rec"], return_counts=True))
        print("times ms:", np.sort(in_bin["time_since_start"]))
        print("dt between DEs ms:", np.diff(np.sort(in_bin["time_since_start"])))
        print("areas:", in_bin["area"])
        print("range_50p_area ms:", in_bin["range_50p_area"] / 1e6)

        if prev_s2_idx >= 0:
            prev_s2 = pS2s[prev_s2_idx]
            print("Previous pS2 time ms:", prev_s2["time_since_start"])
            print("dt from previous pS2 to first DE ms:", first_t - prev_s2["time_since_start"])
            print("Previous pS2 area:", prev_s2["area"])
            print("Previous pS2 subtype:", prev_s2["subtype"])

        near = pS2s[
            (pS2s["time_since_start"] >= a - 500.0)
            & (pS2s["time_since_start"] <= b + 50.0)
        ]

        print(f"Nearby pS2s within [-500,+50] ms: {len(near)}")
        if len(near) > 0:
            print("pS2 times ms:", near["time_since_start"])
            print("pS2 areas:", near["area"])
            print("pS2 subtypes:", np.unique(near["subtype"], return_counts=True))

def collapse_close_clusters(DEs, min_separation_ms=0.1):
    if len(DEs) == 0:
        return DEs

    order = np.argsort(DEs["time_since_start"])
    DEs_sorted = DEs[order]

    keep = np.ones(len(DEs_sorted), dtype=bool)
    last_kept_time = DEs_sorted["time_since_start"][0]

    for i in range(1, len(DEs_sorted)):
        t = DEs_sorted["time_since_start"][i]
        if t - last_kept_time < min_separation_ms:
            keep[i] = False
        else:
            last_kept_time = t

    return DEs_sorted[keep]

def inspect_empty_predicted_region(peaks, DEs_final, pS2s_final, S1s,
                                   start_ms, stop_ms):
    region_all = peaks[
        (peaks["time_since_start"] >= start_ms)
        & (peaks["time_since_start"] < stop_ms)
        ]

    region_DE = DEs_final[
        (DEs_final["time_since_start"] >= start_ms)
        & (DEs_final["time_since_start"] < stop_ms)
        ]

    region_pS2 = pS2s_final[
        (pS2s_final["time_since_start"] >= start_ms)
        & (pS2s_final["time_since_start"] < stop_ms)
        ]

    region_S1 = S1s[
        (S1s["time_since_start"] >= start_ms)
        & (S1s["time_since_start"] < stop_ms)
        ]

    small = region_all[
        (region_all["area"] <= 500)
        & (region_all["n_electron_rec"] <= 5)
        ]

    small_de_like = small[
        np.isin(small["subtype"], [20, 21, 23, 271, 272, 273])
    ]

    small_subtype21 = small[
        small["subtype"] == 21
        ]

    return {
        "start_ms": start_ms,
        "stop_ms": stop_ms,
        "n_all": len(region_all),
        "n_DE": len(region_DE),
        "n_pS2": len(region_pS2),
        "n_S1": len(region_S1),
        "n_small": len(small),
        "n_small_de_like": len(small_de_like),
        "n_small_subtype21": len(small_subtype21),
    }