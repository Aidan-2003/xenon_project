import numpy as np
from scipy.optimize import curve_fit as cf
from scipy.stats import binned_statistic_2d as bs2
from scipy.stats import binned_statistic as bs
import numba
import gc
import strax
import straxen
from scipy.interpolate import interp1d
import os

#TODO: Cut out unnecessary stuff here

#load cuts. Should use private files in the future
class SeparateSEWf:
    
    '''
    This class takes peaks waveforms and quantize them into pieces, then associates piecewise information to each identified piece. 
    
    Please refer to the dedicated note
    
    https://xe1t-wiki.lngs.infn.it/doku.php?id=jlong:sr0_few_e_quantization_from_wf
    '''
    
    def set_dtype(self,l):
        
        self.ext_dtype = [
            ('rise_points',np.int32,l),
            ('fall_points',np.int32,l),
            ('piecewise_area',np.float32,l),
            ('nsegs',np.int16)
        ]
    
    @staticmethod
    @numba.njit
    def average_wf(wf, window):

        res = np.zeros(len(wf))

        for i in numba.prange(len(wf)):
            if i <= window:

                res[i] = wf[:(i+window)].mean()

            elif i >= len(wf)-window:

                res[i] = wf[(i-window+1):].mean()

            else:

                res[i] = wf[(i-window):(i+window)].mean()

        return res
    
    def smear_wf(self,peaks,window):

        dt = peaks['dt']
        wf = peaks['data']
        res = np.zeros(wf.shape)

        for i in range(len(peaks)):

            res[i] = self.average_wf(wf[i],int(np.round(window/dt[i]))+1)

        return res

    @staticmethod
    @numba.njit
    def check_zero_points(wf,max_amp,threshold,l):
        
        # full scan, return rise and fall arrays
        
        trimmed_wf = wf-threshold*max_amp
        mult_res = trimmed_wf[1:]*trimmed_wf[:-1]
        true_turning = np.where(mult_res<0)[0]+1
        
        marker = trimmed_wf[true_turning-1]
        
        res_rise = np.zeros(len(true_turning))
        res_fall = np.zeros(len(true_turning))
        
        n_seg_fall = 0
        n_seg_rise = 0
        n_seg = 0
        
        # the first mark should in principle be always a falling, because the 
        # beginning of the wf is always rising
        
        for i in range(len(marker)):
            
            if marker[i] > 0:
                res_fall[n_seg] = true_turning[i]
                n_seg_fall += 1
                if n_seg_fall == 1:
                    n_seg_rise += 1
                n_seg += 1
            else:
                res_rise[n_seg] = true_turning[i] -1
                n_seg_rise += 1
                # extend rise point by 1 entries
                
        # we have one occasion where there is a rise, but the fall ends at the end of the waveform
        # then such fall will not be registered, and we will loss 1 peak
        if n_seg_rise == n_seg+1:
            res_fall[n_seg] = len(wf)
            n_seg += 1
        else:
            pass
        
        return res_rise[:n_seg], res_fall[:n_seg], n_seg
    
    @staticmethod
    @numba.njit
    def calc_area_piecewise(wf,rise,fall,nseg):

        res = np.zeros(int(nseg))
        
        for i in range(nseg):

            res[i] = np.sum(wf[rise[i]:fall[i]+1])
        return res
    
    def process_quantization(self,peaks,window=100,threshold=0.001):
        
        smeared_wf = self.smear_wf(peaks,window)
        
        areas = []
        rise_points = []
        fall_points = []
        length = []
        
        for i in range(len(peaks)):
            max_amp = peaks[i]['data'].max()
            max_ind = np.argmax(peaks[i]['data'])
            l = peaks[i]['length']
            rise,fall,nseg = self.check_zero_points(smeared_wf[i],max_amp,threshold,l)
            rise[rise>=1]-=1
            areas.append(self.calc_area_piecewise(peaks[i]['data'],rise,fall,nseg))
            length.append(int(nseg))
            rise_points.append(rise)
            fall_points.append(fall)
        
        if length:
            self.set_dtype(np.max(length))
        else:
            self.set_dtype(0)
        res = np.zeros(len(peaks),dtype=self.ext_dtype)
        
        for i in range(len(areas)):
            if length[i]:
                res[i]['piecewise_area'][:length[i]] = areas[i]
                res[i]['rise_points'][:length[i]] = rise_points[i]
                res[i]['fall_points'][:length[i]] = fall_points[i]
                res[i]['nsegs'] = length[i]
            else:
                continue
        return res

class PiecewiseInfo(strax.Plugin,SeparateSEWf):
    
    '''
    A Wrapper around SeparateSEWf for straxen plugin
    '''
    
    __version__ = '0.0.1'
    provides = 'wf_piecewise_info'
    depends_on = ('peak_basics','peaks')
    parallel = True
    
    smearing_window = straxen.URLConfig(
        default=100, type=(int, float),
        help="when doing waveform quantization, the moving average window that smoothen the original waveform")
    
    hit_finding_threshold = straxen.URLConfig(
        default=0.001, type=(int, float),
        help="for smoothened waveforms, the threshold for marking the waveform leaving baseline (0)")
    
    max_piece = straxen.URLConfig(
        default=100, type=(int),
        help="maximum amount of pieces to store")
    
    def infer_dtype(self):
        
        dtype = [
            (('beginning of each piece','rise_points'),np.int32,self.max_piece),
            (('ending of each piece','fall_points'),np.int32,self.max_piece),
            (('area of each piece','piecewise_area'),np.float32,self.max_piece),
            (('total number of pieces','nsegs'),np.int16)
        ] 
        
        dtype+=strax.time_fields
        
        return dtype
        
    def compute(self,peaks):
        
        res = np.zeros(len(peaks),dtype=self.dtype)
        res['time'] = peaks['time']
        res['endtime'] = strax.endtime(peaks)
        
        chunked = self.process_quantization(
            peaks,
            window=self.smearing_window,
            threshold=self.hit_finding_threshold
        )
        
        chunked_length = chunked['piecewise_area'].shape[1]
        compatible_l = np.min([self.max_piece,chunked_length])
        
        for dtype in ['rise_points','fall_points','piecewise_area']:
            res[dtype][:,:compatible_l] = chunked[dtype][:,:compatible_l]
            
        res['nsegs'] = chunked['nsegs']
            
        return res
    
class CountNElectron(strax.Plugin):
    
    '''
    S2 width cut with multiplicity of e-'s in consideration. Based on the quantization algorithm SeparateSEWf each S2 peaks are 
    associated with the number of e- inside such S2, then a cut on S2width 1/99 percentile is done. 
    '''
    
    __version__ = '0.0.2'
    provides = 'n_electron_rec'
    depends_on = ('peak_basics','wf_piecewise_info')
    parallel = True
    dtype = [('n_electron_rec', np.int32, 'number of electrons constituting this peak, reconstructed from waveform')]+strax.time_fields
    
    small_S2_boundary = straxen.URLConfig(
        default=1500, type=(int, float),
        help="threshold of S2 below which this cut is applied")
    
    smearing_window = straxen.URLConfig(
        default=100, type=(int, float),
        help="when doing waveform quantization, the moving average window that smoothen the original waveform")
    
    hit_finding_threshold = straxen.URLConfig(
        default=0.001, type=(int, float),
        help="for smoothened waveforms, the threshold for marking the waveform leaving baseline (0)")
    
    se_gain_val = straxen.URLConfig(
        default='bodega://se_gain?bodega_version=v1',
        help='Nominal single electron (SE) gain in PE / electron extracted.')

        
    @staticmethod
    @numba.njit
    def quantize_e(nseg,seg_area,se_gain):

        res = np.zeros(len(nseg))

        for i in range(len(nseg)):

            base_ne = 0

            for j in range(nseg[i]):

                this_seg_e = seg_area[i][j]/se_gain

                if this_seg_e<0.5:
                    if seg_area[i][j]>8:
                        base_ne += 1
                    else:
                        pass

                else: 
                    base_ne += np.round(seg_area[i][j]/se_gain)

            res[i] = base_ne

        return res
    
    def compute(self,peaks):
        
        ne_counted = np.zeros(len(peaks))
        
        '''# select target S2s
        small_S2_cond = peaks['type'] == 2
        small_S2_cond&= peaks['area'] <= self.small_S2_boundary
        
        chunked = self.process_quantization(
            peaks[small_S2_cond],
            window=self.smearing_window,
            threshold=self.hit_finding_threshold
        )
        
        ne_counted[small_S2_cond] = self.quantize_e(
            chunked['nsegs'],
            chunked['piecewise_area'],
            self.se_gain_val['all_tpc']
        )
        
        large_S2_cond = peaks['type'] == 2
        large_S2_cond&= peaks['area'] > self.small_S2_boundary
        
        ne_counted[large_S2_cond] = np.round(peaks['area'][large_S2_cond]/self.se_gain_val['all_tpc'])'''
        
        ne_counted = self.quantize_e(
            peaks['nsegs'],
            peaks['piecewise_area'],
            self.se_gain_val['all_tpc']
        )
        
        return dict(
            time=peaks['time'],
            endtime=strax.endtime(peaks),
            n_electron_rec=ne_counted
        )