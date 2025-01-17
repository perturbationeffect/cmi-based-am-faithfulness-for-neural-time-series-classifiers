import numpy as np
import pandas as pd
from tslearn.metrics import dtw
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter, windows
from scipy.ndimage.filters import laplace

class SubSequencePerturber:

    class BasePerturber():
        def __init__(self, sample):
            self.original = np.array(sample, copy=True)
            self.sample = sample

        def perturb_subsequence(self, start: int, end: int):
            pass

    class UniformNoise100(BasePerturber):
        def perturb_subsequence(self, start: int, end: int):
            if not hasattr(self, 'unoise100'):
                low = -1 # self.original.min()
                high = 1 # self.original.max()
                self.unoise100 = np.random.uniform(low, high, len(self.original))
            self.sample[start:end] = self.unoise100[start:end]
            return self.sample

    class UniformNoise75(BasePerturber):
        def perturb_subsequence(self, start: int, end: int):
            if not hasattr(self, 'unoise75'):
                low = -1 # self.original.min()
                high = 1 # self.original.max()
                self.unoise75 = np.random.uniform(low, high, len(self.original))
            self.sample[start:end] = (self.sample[start:end] * 0.25) + (self.unoise75[start:end] * 0.75)
            return self.sample

    class UniformNoise50(BasePerturber):
        def perturb_subsequence(self, start: int, end: int):
            if not hasattr(self, 'unoise50'):
                low = -1 # self.original.min()
                high = 1 # self.original.max()
                self.unoise50 = np.random.uniform(low, high, len(self.original))
            self.sample[start:end] = (self.sample[start:end] * 0.5) + (self.unoise50[start:end] * 0.5)
            return self.sample

    class UniformNoise25(BasePerturber):
        def perturb_subsequence(self, start: int, end: int):
            if not hasattr(self, 'unoise25'):
                low = -1 # self.original.min()
                high = 1 # self.original.max()
                self.unoise25 = np.random.uniform(low, high, len(self.original))
            self.sample[start:end] = (self.sample[start:end] * 0.75) + (self.unoise25[start:end] * 0.25)
            return self.sample


    class SampleMean(BasePerturber):
        # t_i = (t_1 + ... + t_n) / n
        def perturb_subsequence(self, start: int, end: int):
            self.sample[start:end] = np.mean(self.original)
            return self.sample


    class GaussianBlur(BasePerturber):
        def perturb_subsequence(self, start: int, end: int):
            if not hasattr(self, 'blur'):
                self.blur = gaussian_filter1d(self.original, 5)
            self.sample[start:end] = self.blur[start:end]
            return self.sample


    class Laplace(BasePerturber):
        def perturb_subsequence(self, start: int, end: int):
            if not hasattr(self, 'laplace'):
                self.laplace = laplace(self.original)
            self.sample[start:end] = self.laplace[start:end]
            return self.sample


    class SavitzkyGolay(BasePerturber):
        def perturb_subsequence(self, start: int, end: int):
            if not hasattr(self, 'savgol'):
                window_length = 21
                if window_length > len(self.original):
                    window_length = len(self.original)
                    if window_length % 2 == 0:
                        window_length -= 1
                self.savgol = savgol_filter(self.original, window_length, 3) #gaussian_filter1d(self.original, 7)
            self.sample[start:end] = self.savgol[start:end]
            return self.sample


    class Zero(BasePerturber):
        # t_i = 0
        def perturb_subsequence(self, start: int, end: int):
            self.sample[start:end] = 0
            return self.sample

    class Inverse(BasePerturber):
        # t_i = t_max - t_i
        def perturb_subsequence(self, start: int, end: int):
            self.sample[start:end] = self.original.max() - self.sample[start:end]
            return self.sample


    class Swap(BasePerturber):
        # vertical mirroring: t_1,t_2,...,t_n -> t_n,...,t_2,t_1
        def perturb_subsequence(self, start: int, end: int):
            self.sample[start:end] = np.flip(self.sample[start:end])
            return self.sample


    class SubsequenceMean(BasePerturber):
        # t_i = (t_1 + ... + t_n) / n
        def perturb_subsequence(self, start: int, end: int):
            self.sample[start:end] = np.mean(self.sample[start:end])
            return self.sample


    class OutOfDistHigh(BasePerturber):
        # t_i = max(abs(t_min), abs(t_max)) * 100
        def perturb_subsequence(self, start: int, end: int):
            abs_max_val = max(abs(self.original.max()), abs(self.original.min()))
            self.sample[start:end] = abs_max_val * 100
            return self.sample

    class OutOfDistLow(BasePerturber):
        # t_i = max(abs(t_min), abs(t_max)) * -100
        def perturb_subsequence(self, start: int, end: int):
            abs_max_val = max(abs(self.original.max()), abs(self.original.min()))
            self.sample[start:end] = abs_max_val * -100
            return self.sample

    class LinearInterpolation(BasePerturber):
        # 
        def perturb_subsequence(self, start: int, end: int):
            self.sample[start:end] = np.nan
            if np.isnan(self.sample[0]):
                self.sample[0] = np.mean(self.original)

            if np.isnan(self.sample[-1]):
                self.sample[-1] = np.mean(self.original)

            s = pd.Series(self.sample)
            s = s.interpolate(method='linear')
            self.sample = s.to_numpy()
            return self.sample

    class QuadraticInterpolation(BasePerturber):
        # 
        def perturb_subsequence(self, start: int, end: int):
            self.sample[start:end] = np.nan
            if np.isnan(self.sample[0]):
                self.sample[0] = np.mean(self.original)

            if np.isnan(self.sample[-1]):
                self.sample[-1] = np.mean(self.original)

            s = pd.Series(self.sample)
            s = s.interpolate(method='quadratic')
            self.sample = s.to_numpy()
            return self.sample

    class CubicInterpolation(BasePerturber):
        # 
        def perturb_subsequence(self, start: int, end: int):
            self.sample[start:end] = np.nan
            if np.isnan(self.sample[0]):
                self.sample[0] = np.mean(self.original)

            if np.isnan(self.sample[-1]):
                self.sample[-1] = np.mean(self.original)

            s = pd.Series(self.sample)
            s = s.interpolate(method='cubic')
            self.sample = s.to_numpy()
            return self.sample


    class Padding(BasePerturber):
        # pad using left value:  1, nan, nan, 5 -> 1, 1, 1, 5
        def perturb_subsequence(self, start: int, end: int):
            self.sample[start:end] = np.nan
            if np.isnan(self.sample[0]):
                self.sample[0] = np.mean(self.original)

            if np.isnan(self.sample[-1]):
                self.sample[-1] = np.mean(self.original)

            s = pd.Series(self.sample)
            s = s.interpolate(method='pad')
            self.sample = s.to_numpy()
            return self.sample

    class Nearest(BasePerturber):
        # Fill NaN with Nearest Neighbor: 1, nan, nan, 5 -> 1, 1, 5, 5
        def perturb_subsequence(self, start: int, end: int):
            self.sample[start:end] = np.nan
            if np.isnan(self.sample[0]):
                self.sample[0] = np.mean(self.original)

            if np.isnan(self.sample[-1]):
                self.sample[-1] = np.mean(self.original)

            s = pd.Series(self.sample)
            s = s.interpolate(method='nearest')
            self.sample = s.to_numpy()
            return self.sample

 
    class LeftNeighborWindow(BasePerturber):
        def perturb_subsequence(self, start: int, end: int):
            if start == 0:
                return self.sample
            else:
                window_size = end - start
                start_shifted = start - window_size
                # if start_shifted < 0:
                #     start_shifted = 0
                end_shifted = end - window_size
                self.sample[start:end] = self.sample[start_shifted:end_shifted]
                return self.sample


    class RightNeighborWindow(BasePerturber):
        def perturb_subsequence(self, start: int, end: int):
            window_size = end - start
            if end > len(self.sample) - window_size:
                return self.sample
            else:
                start_shifted = start + window_size
                end_shifted = end + window_size
                self.sample[start:end] = self.sample[start_shifted:end_shifted]
                return self.sample


    class NearestNeighborWindow(BasePerturber): # 50% left, 50% right
        def perturb_subsequence(self, start: int, end: int):
            window_size = end - start

            if start == 0: # if the window is completely left, take the right neighbor
                start_shifted = start + window_size
                end_shifted = end + window_size
                self.sample[start:end] = self.sample[start_shifted:end_shifted]
            elif (end > len(self.sample) - window_size): # if the window is completely right, take the left neighbor
                start_shifted = start - window_size
                end_shifted = end - window_size
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

                new_sample = np.concatenate([ self.sample[left_shifted_start:left_shifted_end], self.sample[right_shifted_start:right_shifted_end]])
                
                self.sample[start:end] = new_sample
            return self.sample


    class SimilarNeighborWindow(BasePerturber): # Similarity measured using Pearson correlation
        def perturb_subsequence(self, start: int, end: int):
            window_size = end - start

            if start == 0: # if the window is completely left, take the right neighbor
                start_shifted = start + window_size
                end_shifted = end + window_size
                self.sample[start:end] = self.sample[start_shifted:end_shifted]
            elif (end > len(self.sample) - window_size): # if the window is completely right, take the left neighbor
                start_shifted = start - window_size
                end_shifted = end - window_size
                self.sample[start:end] = self.sample[start_shifted:end_shifted]
            else: # take most similar neighbor
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
        def perturb_subsequence(self, start: int, end: int):
            window_size = end - start

            if start == 0: # if the window is completely left, take the right neighbor
                start_shifted = start + window_size
                end_shifted = end + window_size
                self.sample[start:end] = self.sample[start_shifted:end_shifted]
            elif (end > len(self.sample) - window_size): # if the window is completely right, take the left neighbor
                start_shifted = start - window_size
                end_shifted = end - window_size
                self.sample[start:end] = self.sample[start_shifted:end_shifted]
            else: # take most similar neighbor
                left_neighbor  = self.sample[start - window_size:start]
                right_neighbor = self.sample[end:end + window_size]
                left_sim = dtw(left_neighbor, self.sample[start:end])
                right_sim = dtw(right_neighbor, self.sample[start:end])
                if left_sim < right_sim:
                    self.sample[start:end] = right_neighbor
                else:
                    self.sample[start:end] = left_neighbor
            return self.sample