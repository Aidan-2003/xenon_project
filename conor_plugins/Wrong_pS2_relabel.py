import strax
import straxen
import numpy as np

class PS2_relabel(strax.Plugin):
    
    '''
    This processor is designed for the relabeling of wrong pS2 (mainly photoionization). 
    One important selection criteria is that they are only paired with small S1 < 40 PE 
    (mainly lonehit pileup) and within 2 full drift time behind a good pS2.
    This kind of pairing has a high probability as random pairing.
    Please refer to the dedicated note 
    https://xe1t-wiki.lngs.infn.it/doku.php?id=yongyu:subtypes_check_small_s1_selection
    '''
    
    __version__ = '0.0.0'
    provides = 'pS2_relabel'
    depends_on = ('peak_basics','subtype_mask')
    parallel = True
    dtype = strax.time_fields + [
            ('pS2_relabel_S2PH', bool, 'Relabel wrong pS2 as S2 Photoionization'),
            ('pS2_relabel_S2OLS2', bool, 'Relabel wrong pS2 as Other Large S2 after pS2'),
            ('pS2_wrong_pairing', bool, 'pS2 without a S1>40PE before')
            ]

    ls2_threshold_ne = straxen.URLConfig(
        default = 10, type=int, # e-
        help="cutoff between small S2 and large S2, 5e- for now"
    )

    full_dt = straxen.URLConfig(
        default = 2300e3, type = float, #ns
        help = "maximum drift time from cathode"
    )

    ref_se_gain = straxen.URLConfig(
        default='bodega://se_gain?bodega_version=v1',
        help='Nominal single electron (SE) gain in PE / electron extracted.'
    )

    ref_se_span = straxen.URLConfig(
        default='bodega://se_spread?bodega_version=v0',
        help = "SE spread value"
    )
        
    def setup(self):

        self.se_gain = self.ref_se_gain['all_tpc']
        self.se_span = self.ref_se_span['all_tpc']
        self.ls2_threshold = self.ls2_threshold_ne*self.se_gain+np.sqrt(self.ls2_threshold_ne)*self.se_span


# Selection of pS2 paired with S1 under conditions
    
    def ps2_paired_s1(self, peaks, p_11):

        self.mask_22 = (peaks['subtype'] == 22)
           
        peaks_22_a = peaks[self.mask_22][:-1]
        peaks_22_b = peaks[self.mask_22][1:]

        # Create a 'container' with 'bins' (number of 'bins' = len(peaks_22_a)) 
        # and look for 'peaks_11_type' within the bins = look for s1 before pS2 as the right bin
        containers = np.zeros(len(peaks_22_a), dtype=[('time', np.float64),
                                                    ('endtime', np.float64)])
        
        # Searching window set by edges of the 'bins' (no search for first pS2)
        containers['time'] = peaks_22_a['time'] # left bins
        containers['endtime'] = peaks_22_b['time'] # right bins

        # Return index of 'p_11' for each bins
        touching_windows = strax.touching_windows(p_11, containers)
        mask_s2_paired_s1 = ((touching_windows[:, -1] - touching_windows[:, 0]) != 0) 
        # Difference of index =0 ==> no S1 inside the bins
        mask_s2_paired_s1 = np.insert(mask_s2_paired_s1, 0, False) # No search for first pS2

        # Merge two masks: mask_22 (dimension = peaks) and mask_s2_paired_s1 (dimension = peaks_22)
        indice = np.arange(len(peaks))
        comp_indice = indice[self.mask_22][mask_s2_paired_s1]
        mask_s2_paired_s1 = np.zeros(len(peaks),dtype=bool)
        mask_s2_paired_s1[comp_indice] = True # dimension = peaks

        return mask_s2_paired_s1
    

# Selection of pS2 within 2 full drift time under after a good pS2
    
    def ps2_after_ps2(self, peaks):
        
        peaks_22_good = peaks[self.mask_22 & ~self.mask_bad_ps2] # Good pS2: paired with at least one S1>40PE
        peaks_22_bad = peaks[self.mask_22 & self.mask_bad_ps2] # Bad pS2: not paired with any S1>40PE

        containers = np.zeros(len(peaks_22_bad), dtype=[('time', np.float64),
                                                    ('endtime', np.float64)])
        containers['time'] = peaks_22_bad['time'] - 2 * self.full_dt
        containers['endtime'] = peaks_22_bad['time']
        touching_windows = strax.touching_windows(peaks_22_good, containers)
        
        mask_after_ps2 = ((touching_windows[:, -1] - touching_windows[:, 0]) != 0)

        bad_ps2_area = peaks_22_bad[mask_after_ps2]['area'] 
        good_ps2_area = peaks_22_good[touching_windows[:, -1][mask_after_ps2]-1]['area'] 
        mask_PH = (bad_ps2_area <= np.maximum(0.5 * good_ps2_area, self.ls2_threshold))

        indice = np.arange(len(peaks))
        indice_ph = indice[self.mask_bad_ps2][mask_after_ps2][mask_PH]
        indice_ol = indice[self.mask_bad_ps2][mask_after_ps2][~mask_PH]

        mask_PH = np.zeros(len(peaks),dtype=bool) # Photoionization
        mask_OL = np.zeros(len(peaks),dtype=bool) # Other large S2
        mask_PH[indice_ph] = True
        mask_OL[indice_ol] = True

        return mask_PH, mask_OL

    
    def compute(self, peaks):

        result = np.zeros(len(peaks), dtype=self.dtype)
        result['time'] = peaks['time']
        result['endtime'] = strax.endtime(peaks)

        # relabel only pS2 paired with s1 but not fakes2
        peaks_11 = peaks[(peaks['subtype'] == 11)]        
        mask_s2_paired_s1 = self.ps2_paired_s1(peaks, peaks_11)

        # relabel pS2 not paired with any large S1
        # 1. choose pS2 paired with at least one S1>=40PE
        peaks_11_large = peaks[(peaks['subtype'] == 11) & (peaks['area'] >= 40)]  
        # 2. select the rest of pS2 = they pair only with S1<40PE
        mask_s2_paired_small_s1 = ~self.ps2_paired_s1(peaks, peaks_11_large)
        
        self.mask_bad_ps2 = mask_s2_paired_s1 & mask_s2_paired_small_s1

        mask_PH, mask_OL = self.ps2_after_ps2(peaks)

        result['pS2_relabel_S2PH'] = self.mask_bad_ps2 & mask_PH
        result['pS2_relabel_S2OLS2'] = self.mask_bad_ps2 & mask_OL
        result['pS2_wrong_pairing'] = self.mask_bad_ps2

        return result
    