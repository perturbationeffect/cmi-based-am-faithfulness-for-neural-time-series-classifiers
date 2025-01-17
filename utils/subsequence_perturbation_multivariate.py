import numpy as np
import pandas as pd
from tslearn.metrics import dtw
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter, windows
from scipy.ndimage.filters import laplace

class SubSequencePerturber:

    # Base perturbation class
    class BasePerturber():
        def __init__(self, sample):
            self.original = np.array(sample, copy=True)
            self.sample = sample

        def perturb_subsequence(self, start: int, end: int, channel = None):
            pass


    class UniformNoise100(BasePerturber):
        def perturb_subsequence(self, start: int, end: int, channel = None):
            if not hasattr(self, 'unoise100'):
                low = self.original.min()
                high = self.original.max()
                self.unoise100 = np.random.uniform(low, high, self.original.shape)
            if channel is not None:
                self.sample[channel][start:end] = self.unoise100[channel][start:end]
            else:
                self.sample[start:end] = self.unoise100[start:end]
            return self.sample

    class UniformNoise75(BasePerturber):
        def perturb_subsequence(self, start: int, end: int, channel = None):
            if not hasattr(self, 'unoise75'):
                low = self.original.min()
                high = self.original.max()
                self.unoise75 = np.random.uniform(low, high, self.original.shape)
            if channel is not None:
                self.sample[channel][start:end] = self.unoise75[channel][start:end]
            else:
                self.sample[start:end] = self.unoise75[start:end]
            return self.sample

    class UniformNoise50(BasePerturber):
        def perturb_subsequence(self, start: int, end: int, channel = None):
            if not hasattr(self, 'unoise50'):
                low = self.original.min()
                high = self.original.max()
                self.unoise50 = np.random.uniform(low, high, self.original.shape)
            if channel is not None:
                self.sample[channel][start:end] = self.unoise50[channel][start:end]
            else:
                self.sample[start:end] = self.unoise50[start:end]
            return self.sample

    class UniformNoise25(BasePerturber):
        def perturb_subsequence(self, start: int, end: int, channel = None):
            if not hasattr(self, 'unoise25'):
                low = self.original.min()
                high = self.original.max()
                self.unoise25 = np.random.uniform(low, high, self.original.shape)
            if channel is not None:
                self.sample[channel][start:end] = self.unoise25[channel][start:end]
            else:
                self.sample[start:end] = self.unoise25[start:end]
            return self.sample


    class SampleMean(BasePerturber):
        # t_i = (t_1 + ... + t_n) / n
        def perturb_subsequence(self, start: int, end: int, channel = None):
            if channel is not None:
                self.sample[channel][start:end] = np.mean(self.original[channel])
            else:    
                self.sample[start:end] = np.mean(self.original)
            return self.sample
        

    class GaussianBlur(BasePerturber):
        def perturb_subsequence(self, start: int, end: int, channel = None):
            if not hasattr(self, 'blur'):
                if channel is not None:
                    self.blur = np.array(self.original, copy=True)
                    for i in range(len(self.original)):
                        self.blur[i] = gaussian_filter1d(self.original[i], 5)
                else:
                    self.blur = gaussian_filter1d(self.original, 5)

            if channel is not None:
                self.sample[channel][start:end] = self.blur[channel][start:end]
            else:
                self.sample[start:end] = self.blur[start:end]
            return self.sample

    class Laplace(BasePerturber):
        def perturb_subsequence(self, start: int, end: int, channel = None):
            if not hasattr(self, 'laplace'):
                if channel is not None:
                    self.laplace = np.array(self.original, copy=True)
                    for i in range(len(self.original)):
                        self.laplace[i] = laplace(self.original[i])
                else:
                    self.laplace = laplace(self.original)

            if channel is not None:
                self.sample[channel][start:end] = self.laplace[channel][start:end]
            else:
                self.sample[start:end] = self.laplace[start:end]
            return self.sample


    class SavitzkyGolay(BasePerturber):
        def perturb_subsequence(self, start: int, end: int, channel = None):
            if not hasattr(self, 'savgol'):
                window_length = 21
                if window_length > self.original.shape[-1]:
                    window_length = self.original.shape[-1]
                    if window_length % 2 == 0:
                        window_length -= 1

                if channel is not None:
                    self.savgol = np.array(self.original, copy=True)
                    for i in range(len(self.original)):
                        self.savgol[i] = savgol_filter(self.original[i], window_length, 3)
                else:
                    self.savgol = savgol_filter(self.original, window_length, 3) 

            if channel is not None:
                self.sample[channel][start:end] = self.savgol[channel][start:end]
            else:
                self.sample[start:end] = self.savgol[start:end]
            return self.sample
        
    class Zero(BasePerturber):
        # t_i = 0
        def perturb_subsequence(self, start: int, end: int, channel = None):
            if channel is not None:
                self.sample[channel][start:end] = 0
            else:
                self.sample[start:end] = 0
            return self.sample


    class Inverse(BasePerturber):
        # t_i = t_max - t_i
        def perturb_subsequence(self, start: int, end: int, channel = None):
            if channel is not None:
                self.sample[channel][start:end] = self.original[channel].max() - self.sample[channel][start:end]
            else:
                self.sample[start:end] = self.original.max() - self.sample[start:end]
            return self.sample


    class Swap(BasePerturber):
        # vertical mirroring: t_1,t_2,...,t_n -> t_n,...,t_2,t_1
        def perturb_subsequence(self, start: int, end: int, channel = None):
            if channel is not None:
                self.sample[channel][start:end] = np.flip(self.sample[channel][start:end])
            else:
                self.sample[start:end] = np.flip(self.sample[start:end])
            return self.sample


    class SubsequenceMean(BasePerturber):
        # t_i = (t_1 + ... + t_n) / n
        def perturb_subsequence(self, start: int, end: int, channel = None):
            if channel is not None:
                self.sample[channel][start:end] = np.mean(self.sample[channel][start:end])
            else:
                self.sample[start:end] = np.mean(self.sample[start:end])
            return self.sample


    class OutOfDistHigh(BasePerturber):
        # t_i = max(abs(t_min), abs(t_max)) * 100
        def perturb_subsequence(self, start: int, end: int, channel = None):
            abs_max_val = max(abs(self.original.max()), abs(self.original.min()))
            if channel is not None:
                self.sample[channel][start:end] = abs_max_val * 100
            else:
                self.sample[start:end] = abs_max_val * 100
            return self.sample

    class OutOfDistLow(BasePerturber):
        # t_i = max(abs(t_min), abs(t_max)) * -100
        def perturb_subsequence(self, start: int, end: int, channel = None):
            abs_max_val = max(abs(self.original.max()), abs(self.original.min()))
            if channel is not None:
                self.sample[channel][start:end] = abs_max_val * -100
            else:
                self.sample[start:end] = abs_max_val * -100
            return self.sample

    class LinearInterpolation(BasePerturber):
        def perturb_subsequence(self, start: int, end: int, channel = None):
            if channel is not None:
                ts_data = self.sample[channel]
                ts_data_original = self.original[channel]
            else:
                ts_data = self.sample
                ts_data_original = self.original
                
            ts_data[start:end] = np.nan
            if np.isnan(ts_data[0]):
                ts_data[0] = np.mean(ts_data_original)

            if np.isnan(ts_data[-1]):
                ts_data[-1] = np.mean(ts_data_original)

            s = pd.Series(ts_data)
            s = s.interpolate(method='linear')
            ts_data = s.to_numpy()

            if channel is not None:
                self.sample[channel] = ts_data
            else:
                self.sample = ts_data

            return self.sample

    class QuadraticInterpolation(BasePerturber):
        def perturb_subsequence(self, start: int, end: int, channel = None):
            if channel is not None:
                ts_data = self.sample[channel]
                ts_data_original = self.original[channel]
            else:
                ts_data = self.sample
                ts_data_original = self.original
                
            ts_data[start:end] = np.nan
            if np.isnan(ts_data[0]):
                ts_data[0] = np.mean(ts_data_original)

            if np.isnan(ts_data[-1]):
                ts_data[-1] = np.mean(ts_data_original)

            s = pd.Series(ts_data)
            s = s.interpolate(method='quadratic')
            ts_data = s.to_numpy()
            if channel is not None:
                self.sample[channel] = ts_data
            else:
                self.sample = ts_data

            return self.sample


    class CubicInterpolation(BasePerturber):
        def perturb_subsequence(self, start: int, end: int, channel = None):
            if channel is not None:
                ts_data = self.sample[channel]
                ts_data_original = self.original[channel]
            else:
                ts_data = self.sample
                ts_data_original = self.original
                
            ts_data[start:end] = np.nan
            if np.isnan(ts_data[0]):
                ts_data[0] = np.mean(ts_data_original)

            if np.isnan(ts_data[-1]):
                ts_data[-1] = np.mean(ts_data_original)

            s = pd.Series(ts_data)
            s = s.interpolate(method='cubic')
            ts_data = s.to_numpy()
            if channel is not None:
                self.sample[channel] = ts_data
            else:
                self.sample = ts_data

            return self.sample


    class Padding(BasePerturber):
        # pad using left value:  1, nan, nan, 5 -> 1, 1, 1, 5
        def perturb_subsequence(self, start: int, end: int, channel = None):
            if channel is not None:
                ts_data = self.sample[channel]
                ts_data_original = self.original[channel]
            else:
                ts_data = self.sample
                ts_data_original = self.original

            ts_data[start:end] = np.nan
            if np.isnan(ts_data[0]):
                ts_data[0] = np.mean(ts_data_original)

            if np.isnan(ts_data[-1]):
                ts_data[-1] = np.mean(ts_data_original)

            s = pd.Series(ts_data)
            s = s.interpolate(method='pad')
            ts_data = s.to_numpy()
            if channel is not None:
                self.sample[channel] = ts_data
            else:
                self.sample = ts_data

            return self.sample

    class Nearest(BasePerturber):
        # Fill NaN with Nearest Neighbor: 1, nan, nan, 5 -> 1, 1, 5, 5
        def perturb_subsequence(self, start: int, end: int, channel = None):
            if channel is not None:
                ts_data = self.sample[channel]
                ts_data_original = self.original[channel]
            else:
                ts_data = self.sample
                ts_data_original = self.original
                
            ts_data[start:end] = np.nan
            if np.isnan(ts_data[0]):
                ts_data[0] = np.mean(ts_data_original)

            if np.isnan(ts_data[-1]):
                ts_data[-1] = np.mean(ts_data_original)

            s = pd.Series(ts_data)
            s = s.interpolate(method='nearest')
            ts_data = s.to_numpy()
            if channel is not None:
                self.sample[channel] = ts_data
            else:
                self.sample = ts_data

            return self.sample


    class LeftNeighborWindow(BasePerturber):
        def perturb_subsequence(self, start: int, end: int, channel = None):
            window_size = end - start

            if start == 0: # if the window is completely left, take the right neighbor
                start_shifted = start + window_size
                end_shifted = end + window_size
                if channel is not None:
                    self.sample[channel][start:end] = self.sample[channel][start_shifted:end_shifted]
                else:
                    self.sample[start:end] = self.sample[start_shifted:end_shifted]
            else: 
                if channel is not None:
                    left_neighbor  = self.sample[channel][start - window_size:start]
                    self.sample[channel][start:end] = left_neighbor
                else:
                    left_neighbor  = self.sample[start - window_size:start]
                    self.sample[start:end] = left_neighbor
            return self.sample


    class RightNeighborWindow(BasePerturber):
        def perturb_subsequence(self, start: int, end: int, channel = None):
            window_size = end - start

            if (end > self.sample.shape[-1] - window_size): # if the window is completely right, take the left neighbor
                start_shifted = start - window_size
                end_shifted = end - window_size
                if channel is not None:
                    self.sample[channel][start:end] = self.sample[channel][start_shifted:end_shifted]
                else:
                    self.sample[start:end] = self.sample[start_shifted:end_shifted]
            else: 
                if channel is not None:
                    right_neighbor = self.sample[channel][end:end + window_size]
                    self.sample[channel][start:end] = right_neighbor
                else:
                    right_neighbor = self.sample[end:end + window_size]
                    self.sample[start:end] = right_neighbor

            return self.sample


    class NearestNeighborWindow(BasePerturber): # 50% left, 50% right
        def perturb_subsequence(self, start: int, end: int, channel = None):
            window_size = end - start

            if start == 0: # if the window is completely left, take the right neighbor
                start_shifted = start + window_size
                end_shifted = end + window_size
                if channel is not None:
                    self.sample[channel][start:end] = self.sample[channel][start_shifted:end_shifted]
                else:
                    self.sample[start:end] = self.sample[start_shifted:end_shifted]
            elif (end > self.sample.shape[-1] - window_size): # if the window is completely right, take the left neighbor
                start_shifted = start - window_size
                end_shifted = end - window_size
                if channel is not None:
                    self.sample[channel][start:end] = self.sample[channel][start_shifted:end_shifted]
                else:
                    self.sample[start:end] = self.sample[start_shifted:end_shifted]
            else: # take 50% of the left, and 50% of the right
                if window_size % 2 == 0:
                    left_window_size = window_size // 2
                else:
                    left_window_size = (window_size // 2) + 1
                    
                right_window_size = window_size // 2

                left_shifted_start = start - left_window_size
                left_shifted_end = start
                right_shifted_start = end
                right_shifted_end = end + right_window_size

                if channel is not None:
                    new_sample = np.concatenate([ self.sample[channel][left_shifted_start:left_shifted_end], self.sample[channel][right_shifted_start:right_shifted_end]])
                    self.sample[channel][start:end] = new_sample
                else:
                    new_sample = np.concatenate([ self.sample[left_shifted_start:left_shifted_end], self.sample[right_shifted_start:right_shifted_end]])
                    self.sample[start:end] = new_sample

            return self.sample


    class SimilarNeighborWindow(BasePerturber): # Similarity measured using Pearson correlation
        def perturb_subsequence(self, start: int, end: int, channel = None):
            window_size = end - start

            if start == 0: # if the window is completely left, take the right neighbor
                start_shifted = start + window_size
                end_shifted = end + window_size
                if channel is not None:
                    self.sample[channel][start:end] = self.sample[channel][start_shifted:end_shifted]
                else:
                    self.sample[start:end] = self.sample[start_shifted:end_shifted]
            elif (end > self.sample.shape[-1] - window_size): # if the window is completely right, take the left neighbor
                start_shifted = start - window_size
                end_shifted = end - window_size
                if channel is not None:
                    self.sample[channel][start:end] = self.sample[channel][start_shifted:end_shifted]
                else:
                    self.sample[start:end] = self.sample[start_shifted:end_shifted]
            else: # take most similar neighbor
                if channel is not None:
                    left_neighbor  = self.sample[channel][start - window_size:start]
                    right_neighbor = self.sample[channel][end:end + window_size]
                    left_sim = dtw(left_neighbor, self.sample[channel][start:end])
                    right_sim = dtw(right_neighbor, self.sample[channel][start:end])

                    if left_sim < right_sim:
                        self.sample[channel][start:end] = left_neighbor
                    else:
                        self.sample[channel][start:end] = right_neighbor
                else:
                    left_neighbor  = self.sample[start - window_size:start]
                    right_neighbor = self.sample[end:end + window_size]
                    left_sim = dtw(left_neighbor, self.sample[start:end])
                    right_sim = dtw(right_neighbor, self.sample[start:end])

                    if left_sim < right_sim:
                        self.sample[start:end] = left_neighbor
                    else:
                        self.sample[start:end] = right_neighbor

            return self.sample


    class DissimilarNeighborWindow(BasePerturber): # Similarity measured using Pearson correlation
        def perturb_subsequence(self, start: int, end: int, channel = None):
            window_size = end - start

            if start == 0: # if the window is completely left, take the right neighbor
                start_shifted = start + window_size
                end_shifted = end + window_size
                if channel is not None:
                    self.sample[channel][start:end] = self.sample[channel][start_shifted:end_shifted]
                else:
                    self.sample[start:end] = self.sample[start_shifted:end_shifted]
            elif (end > self.sample.shape[-1] - window_size): # if the window is completely right, take the left neighbor
                start_shifted = start - window_size
                end_shifted = end - window_size
                if channel is not None:
                    self.sample[channel][start:end] = self.sample[channel][start_shifted:end_shifted]
                else:
                    self.sample[start:end] = self.sample[start_shifted:end_shifted]
            else: # take most similar neighbor
                if channel is not None:
                    left_neighbor  = self.sample[channel][start - window_size:start]
                    right_neighbor = self.sample[channel][end:end + window_size]
                    left_sim = dtw(left_neighbor, self.sample[channel][start:end])
                    right_sim = dtw(right_neighbor, self.sample[channel][start:end])

                    if left_sim < right_sim:
                        self.sample[channel][start:end] = right_neighbor
                    else:
                        self.sample[channel][start:end] = left_neighbor
                else:
                    left_neighbor  = self.sample[start - window_size:start]
                    right_neighbor = self.sample[end:end + window_size]
                    left_sim = dtw(left_neighbor, self.sample[start:end])
                    right_sim = dtw(right_neighbor, self.sample[start:end])

                    if left_sim < right_sim:
                        self.sample[start:end] = right_neighbor
                    else:
                        self.sample[start:end] = left_neighbor

            return self.sample
        
