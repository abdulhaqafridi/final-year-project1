# # import numpy as np
# # import pandas as pd
# # from scipy import signal
# # from scipy.stats import skew, kurtosis
# # import pywt
# # import pickle
# # import os
# # from sklearn.preprocessing import StandardScaler

# # class EEGFeatureExtractor:
# #     def __init__(self, fs=256):
# #         self.fs = fs
        
# #     def extract_time_domain_features(self, data):
# #         """Extract time domain features"""
# #         features = {}
        
# #         # Basic statistical features
# #         features['mean'] = np.mean(data)
# #         features['std'] = np.std(data)
# #         features['var'] = np.var(data)
# #         features['min'] = np.min(data)
# #         features['max'] = np.max(data)
# #         features['range'] = np.max(data) - np.min(data)
# #         features['rms'] = np.sqrt(np.mean(data**2))
        
# #         # Higher order statistics
# #         features['skewness'] = skew(data)
# #         features['kurtosis'] = kurtosis(data)
        
# #         # Zero crossing rate
# #         zero_crossings = np.where(np.diff(np.signbit(data)))[0]
# #         features['zero_crossing_rate'] = len(zero_crossings) / len(data)
        
# #         # Activity (variance)
# #         features['activity'] = np.var(data)
        
# #         # Mobility (first derivative)
# #         diff1 = np.diff(data)
# #         features['mobility'] = np.sqrt(np.var(diff1) / np.var(data))
        
# #         # Complexity (second derivative)
# #         diff2 = np.diff(diff1)
# #         if np.var(diff1) != 0:
# #             features['complexity'] = np.sqrt(np.var(diff2) / np.var(diff1)) / features['mobility']
# #         else:
# #             features['complexity'] = 0
            
# #         return features
    
# #     def extract_frequency_domain_features(self, data):
# #         """Extract frequency domain features using FFT"""
# #         features = {}
        
# #         # Compute power spectral density
# #         freqs, psd = signal.welch(data, fs=self.fs, nperseg=len(data)//4)
        
# #         # Define frequency bands
# #         delta_band = (0.5, 4)
# #         theta_band = (4, 8)
# #         alpha_band = (8, 13)
# #         beta_band = (13, 30)
# #         gamma_band = (30, 50)
        
# #         bands = {
# #             'delta': delta_band,
# #             'theta': theta_band,
# #             'alpha': alpha_band,
# #             'beta': beta_band,
# #             'gamma': gamma_band
# #         }
        
# #         # Calculate power in each frequency band
# #         total_power = np.trapz(psd, freqs)
        
# #         for band_name, (low_freq, high_freq) in bands.items():
# #             band_mask = (freqs >= low_freq) & (freqs <= high_freq)
# #             band_power = np.trapz(psd[band_mask], freqs[band_mask])
            
# #             features[f'{band_name}_power'] = band_power
# #             features[f'{band_name}_relative_power'] = band_power / total_power if total_power > 0 else 0
        
# #         # Spectral centroid
# #         features['spectral_centroid'] = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0
        
# #         # Spectral bandwidth
# #         centroid = features['spectral_centroid']
# #         features['spectral_bandwidth'] = np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / np.sum(psd)) if np.sum(psd) > 0 else 0
        
# #         # Peak frequency
# #         peak_idx = np.argmax(psd)
# #         features['peak_frequency'] = freqs[peak_idx]
        
# #         # Spectral edge frequency (95% of power)
# #         cumsum_psd = np.cumsum(psd)
# #         features['spectral_edge_freq'] = freqs[np.where(cumsum_psd >= 0.95 * cumsum_psd[-1])[0][0]]
        
# #         return features
    
# #     def extract_wavelet_features(self, data):
# #         """Extract wavelet domain features"""
# #         features = {}
        
# #         # Wavelet decomposition using Daubechies wavelet
# #         wavelet = 'db4'
# #         levels = 5
        
# #         try:
# #             coeffs = pywt.wavedec(data, wavelet, level=levels)
            
# #             # Features from each decomposition level
# #             for i, coeff in enumerate(coeffs):
# #                 level_name = 'cA' if i == 0 else f'cD{i}'
                
# #                 features[f'wavelet_{level_name}_energy'] = np.sum(coeff**2)
# #                 features[f'wavelet_{level_name}_mean'] = np.mean(np.abs(coeff))
# #                 features[f'wavelet_{level_name}_std'] = np.std(coeff)
# #                 features[f'wavelet_{level_name}_entropy'] = self.calculate_entropy(coeff)
            
# #             # Relative wavelet energy
# #             total_energy = sum(np.sum(coeff**2) for coeff in coeffs)
# #             for i, coeff in enumerate(coeffs):
# #                 level_name = 'cA' if i == 0 else f'cD{i}'
# #                 features[f'wavelet_{level_name}_rel_energy'] = np.sum(coeff**2) / total_energy if total_energy > 0 else 0
                
# #         except Exception as e:
# #             print(f"Wavelet extraction error: {e}")
# #             # Fill with zeros if wavelet extraction fails
# #             for i in range(levels + 1):
# #                 level_name = 'cA' if i == 0 else f'cD{i}'
# #                 features[f'wavelet_{level_name}_energy'] = 0
# #                 features[f'wavelet_{level_name}_mean'] = 0
# #                 features[f'wavelet_{level_name}_std'] = 0
# #                 features[f'wavelet_{level_name}_entropy'] = 0
# #                 features[f'wavelet_{level_name}_rel_energy'] = 0
        
# #         return features
    
# #     def calculate_entropy(self, data):
# #         """Calculate Shannon entropy"""
# #         # Normalize data
# #         data = np.abs(data)
# #         if np.sum(data) == 0:
# #             return 0
        
# #         data = data / np.sum(data)
# #         # Remove zeros to avoid log(0)
# #         data = data[data > 0]
        
# #         if len(data) == 0:
# #             return 0
        
# #         return -np.sum(data * np.log2(data))
    
# #     def extract_nonlinear_features(self, data):
# #         """Extract nonlinear features"""
# #         features = {}
        
# #         # Approximate entropy
# #         features['approximate_entropy'] = self.approximate_entropy(data, m=2, r=0.2*np.std(data))
        
# #         # Sample entropy
# #         features['sample_entropy'] = self.sample_entropy(data, m=2, r=0.2*np.std(data))
        
# #         # Hjorth parameters (already calculated in time domain, but included here for completeness)
# #         diff1 = np.diff(data)
# #         diff2 = np.diff(diff1)
        
# #         features['hjorth_activity'] = np.var(data)
# #         features['hjorth_mobility'] = np.sqrt(np.var(diff1) / np.var(data)) if np.var(data) > 0 else 0
# #         if np.var(diff1) > 0:
# #             features['hjorth_complexity'] = (np.sqrt(np.var(diff2) / np.var(diff1)) / features['hjorth_mobility']) if features['hjorth_mobility'] > 0 else 0
# #         else:
# #             features['hjorth_complexity'] = 0
        
# #         return features
    
# #     def approximate_entropy(self, data, m=2, r=0.2):
# #         """Calculate approximate entropy"""
# #         N = len(data)
        
# #         def _maxdist(xi, xj, N, m):
# #             return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
# #         def _phi(m):
# #             patterns = np.array([data[i:i+m] for i in range(N - m + 1)])
# #             C = np.zeros(N - m + 1)
            
# #             for i in range(N - m + 1):
# #                 template_i = patterns[i]
# #                 for j in range(N - m + 1):
# #                     if _maxdist(template_i, patterns[j], N, m) <= r:
# #                         C[i] += 1.0
                        
# #             phi = (N - m + 1.0) ** (-1) * sum([np.log(c / (N - m + 1.0)) for c in C if c > 0])
# #             return phi
        
# #         try:
# #             return _phi(m) - _phi(m + 1)
# #         except:
# #             return 0
    
# #     def sample_entropy(self, data, m=2, r=0.2):
# #         """Calculate sample entropy"""
# #         N = len(data)
        
# #         def _maxdist(xi, xj):
# #             return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
# #         def _phi(m):
# #             patterns = np.array([data[i:i+m] for i in range(N - m + 1)])
# #             C = 0
            
# #             for i in range(N - m):
# #                 template_i = patterns[i]
# #                 for j in range(i + 1, N - m + 1):
# #                     if _maxdist(template_i, patterns[j]) <= r:
# #                         C += 1
                        
# #             return C
        
# #         try:
# #             A = _phi(m + 1)
# #             B = _phi(m)
# #             return np.log(B / A) if A > 0 else 0
# #         except:
# #             return 0
    
# #     def extract_all_features(self, data):
# #         """Extract all features from EEG data"""
# #         all_features = {}
        
# #         # Time domain features
# #         time_features = self.extract_time_domain_features(data)
# #         all_features.update({f'time_{k}': v for k, v in time_features.items()})
        
# #         # Frequency domain features
# #         freq_features = self.extract_frequency_domain_features(data)
# #         all_features.update({f'freq_{k}': v for k, v in freq_features.items()})
        
# #         # Wavelet features
# #         wavelet_features = self.extract_wavelet_features(data)
# #         all_features.update(wavelet_features)
        
# #         # Nonlinear features
# #         nonlinear_features = self.extract_nonlinear_features(data)
# #         all_features.update({f'nonlinear_{k}': v for k, v in nonlinear_features.items()})
        
# #         return all_features

# # def process_segments_to_features(seizure_segments, non_seizure_segments, output_dir="processed_data"):
# #     """Process segments and extract features"""
    
# #     feature_extractor = EEGFeatureExtractor(fs=256)
    
# #     all_features = []
# #     all_labels = []
    
# #     print("Extracting features from seizure segments...")
# #     for i, segment in enumerate(seizure_segments):
# #         if i % 100 == 0:
# #             print(f"Processing seizure segment {i+1}/{len(seizure_segments)}")
        
# #         features = feature_extractor.extract_all_features(segment['data'])
# #         features['patient'] = segment['patient']
# #         features['file'] = segment['file']
# #         features['channel'] = segment['channel']
        
# #         all_features.append(features)
# #         all_labels.append(1)  # Seizure
    
# #     print("Extracting features from non-seizure segments...")
# #     for i, segment in enumerate(non_seizure_segments):
# #         if i % 100 == 0:
# #             print(f"Processing non-seizure segment {i+1}/{len(non_seizure_segments)}")
        
# #         features = feature_extractor.extract_all_features(segment['data'])
# #         features['patient'] = segment['patient']
# #         features['file'] = segment['file']
# #         features['channel'] = segment['channel']
        
# #         all_features.append(features)
# #         all_labels.append(0)  # Non-seizure
    
# #     # Convert to DataFrame
# #     features_df = pd.DataFrame(all_features)
# #     features_df['label'] = all_labels
    
# #     # Save features
# #     os.makedirs(output_dir, exist_ok=True)
# #     features_df.to_csv(os.path.join(output_dir, 'eeg_features.csv'), index=False)
    
# #     # Save as pickle for faster loading
# #     with open(os.path.join(output_dir, 'eeg_features.pkl'), 'wb') as f:
# #         pickle.dump(features_df, f)
    
# #     print(f"\nFeature extraction completed!")
# #     print(f"Total samples: {len(features_df)}")
# #     print(f"Seizure samples: {sum(all_labels)}")
# #     print(f"Non-seizure samples: {len(all_labels) - sum(all_labels)}")
# #     print(f"Total features: {len(features_df.columns) - 4}")  # Excluding patient, file, channel, label
# #     print(f"Features saved to: {output_dir}/eeg_features.csv")
    
# #     return features_df

# # if __name__ == "__main__":
# #     # Load segments
# #     try:
# #         with open('processed_data/seizure_segments.pkl', 'rb') as f:
# #             seizure_segments = pickle.load(f)
        
# #         with open('processed_data/non_seizure_segments.pkl', 'rb') as f:
# #             non_seizure_segments = pickle.load(f)
        
# #         print(f"Loaded {len(seizure_segments)} seizure segments")
# #         print(f"Loaded {len(non_seizure_segments)} non-seizure segments")
        
# #         # Extract features
# #         features_df = process_segments_to_features(seizure_segments, non_seizure_segments)
        
# #         # Display feature summary
# #         print("\nFeature Summary:")
# #         print(f"Dataset shape: {features_df.shape}")
# #         print(f"Feature columns: {features_df.columns.tolist()}")
        
# #     except FileNotFoundError:
# #         print("Error: Segment files not found. Please run extract_eeg_segments.py first.")



# import os
# import pickle
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from scipy import signal
# from scipy.stats import skew, kurtosis
# import pywt
# from multiprocessing import Pool, cpu_count

# # EEG Feature Extractor class
# class EEGFeatureExtractor:
#     def __init__(self, fs=256):
#         self.fs = fs

#     def extract_time_domain_features(self, data):
#         features = {
#             'mean': np.mean(data),
#             'std': np.std(data),
#             'var': np.var(data),
#             'min': np.min(data),
#             'max': np.max(data),
#             'range': np.max(data) - np.min(data),
#             'rms': np.sqrt(np.mean(data ** 2)),
#             'skewness': skew(data),
#             'kurtosis': kurtosis(data),
#             'zero_crossing_rate': len(np.where(np.diff(np.signbit(data)))[0]) / len(data),
#             'activity': np.var(data),
#         }
#         diff1 = np.diff(data)
#         features['mobility'] = np.sqrt(np.var(diff1) / np.var(data)) if np.var(data) > 0 else 0
#         diff2 = np.diff(diff1)
#         if np.var(diff1) != 0:
#             features['complexity'] = np.sqrt(np.var(diff2) / np.var(diff1)) / features['mobility'] if features['mobility'] > 0 else 0
#         else:
#             features['complexity'] = 0
#         return features

#     def extract_frequency_domain_features(self, data):
#         features = {}
#         freqs, psd = signal.welch(data, fs=self.fs, nperseg=len(data)//4)

#         bands = {
#             'delta': (0.5, 4),
#             'theta': (4, 8),
#             'alpha': (8, 13),
#             'beta': (13, 30),
#             'gamma': (30, 50)
#         }

#         total_power = np.trapz(psd, freqs)
#         for band, (low, high) in bands.items():
#             mask = (freqs >= low) & (freqs <= high)
#             power = np.trapz(psd[mask], freqs[mask])
#             features[f'{band}_power'] = power
#             features[f'{band}_relative_power'] = power / total_power if total_power > 0 else 0

#         features['spectral_centroid'] = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0
#         centroid = features['spectral_centroid']
#         features['spectral_bandwidth'] = np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / np.sum(psd)) if np.sum(psd) > 0 else 0
#         peak_idx = np.argmax(psd)
#         features['peak_frequency'] = freqs[peak_idx]
#         cumsum_psd = np.cumsum(psd)
#         features['spectral_edge_freq'] = freqs[np.where(cumsum_psd >= 0.95 * cumsum_psd[-1])[0][0]]
#         return features

#     def extract_wavelet_features(self, data):
#         features = {}
#         wavelet = 'db4'
#         levels = 5
#         try:
#             coeffs = pywt.wavedec(data, wavelet, level=levels)
#             total_energy = sum(np.sum(c ** 2) for c in coeffs)
#             for i, coeff in enumerate(coeffs):
#                 level = 'cA' if i == 0 else f'cD{i}'
#                 features[f'wavelet_{level}_energy'] = np.sum(coeff ** 2)
#                 features[f'wavelet_{level}_mean'] = np.mean(np.abs(coeff))
#                 features[f'wavelet_{level}_std'] = np.std(coeff)
#                 features[f'wavelet_{level}_entropy'] = self.calculate_entropy(coeff)
#                 features[f'wavelet_{level}_rel_energy'] = np.sum(coeff ** 2) / total_energy if total_energy > 0 else 0
#         except:
#             for i in range(levels + 1):
#                 level = 'cA' if i == 0 else f'cD{i}'
#                 for name in ['energy', 'mean', 'std', 'entropy', 'rel_energy']:
#                     features[f'wavelet_{level}_{name}'] = 0
#         return features

#     def calculate_entropy(self, data):
#         data = np.abs(data)
#         data = data / np.sum(data) if np.sum(data) > 0 else data
#         data = data[data > 0]
#         return -np.sum(data * np.log2(data)) if len(data) > 0 else 0

#     def extract_nonlinear_features(self, data):
#         features = {
#             'hjorth_activity': np.var(data),
#             'hjorth_mobility': np.sqrt(np.var(np.diff(data)) / np.var(data)) if np.var(data) > 0 else 0
#         }
#         diff1 = np.diff(data)
#         diff2 = np.diff(diff1)
#         if np.var(diff1) > 0:
#             features['hjorth_complexity'] = (np.sqrt(np.var(diff2) / np.var(diff1)) / features['hjorth_mobility']) if features['hjorth_mobility'] > 0 else 0
#         else:
#             features['hjorth_complexity'] = 0
#         features['approximate_entropy'] = self.approximate_entropy(data)
#         features['sample_entropy'] = self.sample_entropy(data)
#         return features

#     def approximate_entropy(self, data, m=2, r=None):
#         N = len(data)
#         r = r if r is not None else 0.2 * np.std(data)
#         if N <= m + 1:
#             return 0
#         def _phi(m):
#             patterns = np.array([data[i:i+m] for i in range(N - m + 1)])
#             C = [np.sum(np.max(np.abs(patterns - p), axis=1) <= r) for p in patterns]
#             C = [c / (N - m + 1.0) for c in C]
#             return np.sum(np.log(C)) / (N - m + 1.0)
#         return _phi(m) - _phi(m + 1)

#     def sample_entropy(self, data, m=2, r=None):
#         N = len(data)
#         r = r if r is not None else 0.2 * np.std(data)
#         if N <= m + 1:
#             return 0
#         def _phi(m):
#             patterns = np.array([data[i:i+m] for i in range(N - m)])
#             count = 0
#             for i in range(len(patterns)):
#                 for j in range(i+1, len(patterns)):
#                     if np.max(np.abs(patterns[i] - patterns[j])) <= r:
#                         count += 1
#             return count
#         B = _phi(m)
#         A = _phi(m + 1)
#         return -np.log(A / B) if B > 0 and A > 0 else 0

#     def extract_all_features(self, data):
#         features = {}
#         features.update({f'time_{k}': v for k, v in self.extract_time_domain_features(data).items()})
#         features.update({f'freq_{k}': v for k, v in self.extract_frequency_domain_features(data).items()})
#         features.update(self.extract_wavelet_features(data))
#         features.update({f'nonlinear_{k}': v for k, v in self.extract_nonlinear_features(data).items()})
#         return features

# # Function wrapper for multiprocessing
# def extract_features_wrapper(segment_label):
#     segment, label = segment_label
#     extractor = EEGFeatureExtractor(fs=256)
#     features = extractor.extract_all_features(segment['data'])
#     features['patient'] = segment['patient']
#     features['file'] = segment['file']
#     features['channel'] = segment['channel']
#     features['label'] = label
#     return features

# # Main processing function
# def process_segments_to_features(seizure_segments, non_seizure_segments, output_dir="processed_data"):
#     all_segments = [(seg, 1) for seg in seizure_segments] + [(seg, 0) for seg in non_seizure_segments]

#     print(f"\n🧠 Extracting features using {cpu_count()} CPU cores...")
#     with Pool(processes=cpu_count()) as pool:
#         all_features = list(tqdm(pool.imap(extract_features_wrapper, all_segments), total=len(all_segments)))

#     features_df = pd.DataFrame(all_features)
#     os.makedirs(output_dir, exist_ok=True)
#     features_df.to_csv(os.path.join(output_dir, 'eeg_features.csv'), index=False)
#     with open(os.path.join(output_dir, 'eeg_features.pkl'), 'wb') as f:
#         pickle.dump(features_df, f)

#     print(f"\n✅ Feature extraction completed!")
#     print(f"  🔢 Total samples: {len(features_df)}")
#     print(f"  💾 Features saved to: {output_dir}/eeg_features.csv")
#     return features_df

# # Entry point
# if __name__ == "__main__":
#     try:
#         with open('processed_data/seizure_segments.pkl', 'rb') as f:
#             seizure_segments = pickle.load(f)
#         with open('processed_data/non_seizure_segments.pkl', 'rb') as f:
#             non_seizure_segments = pickle.load(f)

#         print(f"📦 Loaded {len(seizure_segments)} seizure segments")
#         print(f"📦 Loaded {len(non_seizure_segments)} non-seizure segments")
#         process_segments_to_features(seizure_segments, non_seizure_segments)

#     except FileNotFoundError:
#         print("❌ Segment files not found. Please run extract_eeg_segments.py first.")
  


import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import signal
from scipy.stats import skew, kurtosis
import pywt
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import seaborn as sns

# EEG Feature Extractor class
class EEGFeatureExtractor:
    def __init__(self, fs=256):
        self.fs = fs

    def extract_time_domain_features(self, data):
        features = {
            'mean': np.mean(data),
            'std': np.std(data),
            'var': np.var(data),
            'min': np.min(data),
            'max': np.max(data),
            'range': np.max(data) - np.min(data),
            'rms': np.sqrt(np.mean(data ** 2)),
            'skewness': skew(data),
            'kurtosis': kurtosis(data),
            'zero_crossing_rate': len(np.where(np.diff(np.signbit(data)))[0]) / len(data),
            'activity': np.var(data),
        }
        diff1 = np.diff(data)
        features['mobility'] = np.sqrt(np.var(diff1) / np.var(data)) if np.var(data) > 0 else 0
        diff2 = np.diff(diff1)
        if np.var(diff1) != 0:
            features['complexity'] = np.sqrt(np.var(diff2) / np.var(diff1)) / features['mobility'] if features['mobility'] > 0 else 0
        else:
            features['complexity'] = 0
        return features

    def extract_frequency_domain_features(self, data):
        features = {}
        freqs, psd = signal.welch(data, fs=self.fs, nperseg=len(data)//4)

        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }

        total_power = np.trapz(psd, freqs)
        for band, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs <= high)
            power = np.trapz(psd[mask], freqs[mask])
            features[f'{band}_power'] = power
            features[f'{band}_relative_power'] = power / total_power if total_power > 0 else 0

        features['spectral_centroid'] = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0
        centroid = features['spectral_centroid']
        features['spectral_bandwidth'] = np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / np.sum(psd)) if np.sum(psd) > 0 else 0
        peak_idx = np.argmax(psd)
        features['peak_frequency'] = freqs[peak_idx]
        cumsum_psd = np.cumsum(psd)
        features['spectral_edge_freq'] = freqs[np.where(cumsum_psd >= 0.95 * cumsum_psd[-1])[0][0]]
        return features

    def extract_wavelet_features(self, data):
        features = {}
        wavelet = 'db4'
        levels = 5
        try:
            coeffs = pywt.wavedec(data, wavelet, level=levels)
            total_energy = sum(np.sum(c ** 2) for c in coeffs)
            for i, coeff in enumerate(coeffs):
                level = 'cA' if i == 0 else f'cD{i}'
                features[f'wavelet_{level}_energy'] = np.sum(coeff ** 2)
                features[f'wavelet_{level}_mean'] = np.mean(np.abs(coeff))
                features[f'wavelet_{level}_std'] = np.std(coeff)
                features[f'wavelet_{level}_entropy'] = self.calculate_entropy(coeff)
                features[f'wavelet_{level}_rel_energy'] = np.sum(coeff ** 2) / total_energy if total_energy > 0 else 0
        except:
            for i in range(levels + 1):
                level = 'cA' if i == 0 else f'cD{i}'
                for name in ['energy', 'mean', 'std', 'entropy', 'rel_energy']:
                    features[f'wavelet_{level}_{name}'] = 0
        return features

    def calculate_entropy(self, data):
        data = np.abs(data)
        data = data / np.sum(data) if np.sum(data) > 0 else data
        data = data[data > 0]
        return -np.sum(data * np.log2(data)) if len(data) > 0 else 0

    def extract_nonlinear_features(self, data):
        features = {
            'hjorth_activity': np.var(data),
            'hjorth_mobility': np.sqrt(np.var(np.diff(data)) / np.var(data)) if np.var(data) > 0 else 0
        }
        diff1 = np.diff(data)
        diff2 = np.diff(diff1)
        if np.var(diff1) > 0:
            features['hjorth_complexity'] = (np.sqrt(np.var(diff2) / np.var(diff1)) / features['hjorth_mobility']) if features['hjorth_mobility'] > 0 else 0
        else:
            features['hjorth_complexity'] = 0
        features['approximate_entropy'] = self.approximate_entropy(data)
        features['sample_entropy'] = self.sample_entropy(data)
        return features

    def approximate_entropy(self, data, m=2, r=None):
        N = len(data)
        r = r if r is not None else 0.2 * np.std(data)
        if N <= m + 1:
            return 0
        def _phi(m):
            patterns = np.array([data[i:i+m] for i in range(N - m + 1)])
            C = [np.sum(np.max(np.abs(patterns - p), axis=1) <= r) for p in patterns]
            C = [c / (N - m + 1.0) for c in C]
            return np.sum(np.log(C)) / (N - m + 1.0)
        return _phi(m) - _phi(m + 1)

    def sample_entropy(self, data, m=2, r=None):
        N = len(data)
        r = r if r is not None else 0.2 * np.std(data)
        if N <= m + 1:
            return 0
        def _phi(m):
            patterns = np.array([data[i:i+m] for i in range(N - m)])
            count = 0
            for i in range(len(patterns)):
                for j in range(i+1, len(patterns)):
                    if np.max(np.abs(patterns[i] - patterns[j])) <= r:
                        count += 1
            return count
        B = _phi(m)
        A = _phi(m + 1)
        return -np.log(A / B) if B > 0 and A > 0 else 0

    def extract_all_features(self, data):
        features = {}
        features.update({f'time_{k}': v for k, v in self.extract_time_domain_features(data).items()})
        features.update({f'freq_{k}': v for k, v in self.extract_frequency_domain_features(data).items()})
        features.update(self.extract_wavelet_features(data))
        features.update({f'nonlinear_{k}': v for k, v in self.extract_nonlinear_features(data).items()})
        return features

# Function wrapper for multiprocessing
def extract_features_wrapper(segment_label):
    segment, label = segment_label
    extractor = EEGFeatureExtractor(fs=256)
    features = extractor.extract_all_features(segment['data'])
    features['patient'] = segment['patient']
    features['file'] = segment['file']
    features['channel'] = segment['channel']
    features['label'] = label
    return features

def visualize_features(features_df, output_dir="processed_data"):
    """Create feature visualizations with fixed layout issues"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Separate metadata and feature columns
    metadata_cols = ['patient', 'file', 'channel', 'label']
    feature_cols = [col for col in features_df.columns if col not in metadata_cols]
    
    print(f"📊 Creating visualizations for {len(feature_cols)} features...")
    
    # 1. Feature Distribution by Class (Fixed title overlap)
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.ravel()
    
    for i, feature in enumerate(feature_cols[:16]):  # Show first 16 features
        seizure_data = features_df[features_df['label'] == 1][feature]
        non_seizure_data = features_df[features_df['label'] == 0][feature]
        
        axes[i].hist(non_seizure_data, alpha=0.6, label='Non-Seizure', bins=30, color='blue')
        axes[i].hist(seizure_data, alpha=0.6, label='Seizure', bins=30, color='red')
        axes[i].set_title(feature, fontsize=10, pad=15)  # Added padding to fix overlap
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout(pad=3.0)  # Increased padding
    plt.savefig(os.path.join(output_dir, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature Correlation Heatmap (Fixed values display)
    plt.figure(figsize=(16, 14))
    
    # Select a subset of features for correlation (to avoid overcrowding)
    important_features = feature_cols[:25]  # Top 25 features
    correlation_data = features_df[important_features]
    
    # Calculate correlation matrix
    correlation_matrix = correlation_data.corr()
    
    # Create heatmap with proper annotation
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True,  # Show correlation values
                fmt='.2f',   # Format to 2 decimal places
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=.5,
                cbar_kws={"shrink": .8},
                annot_kws={'size': 8})  # Adjust annotation size
    
    plt.title('Feature Correlation Matrix (Top 25 Features)', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Class Distribution
    plt.figure(figsize=(8, 6))
    class_counts = features_df['label'].value_counts()
    colors = ['lightblue', 'lightcoral']
    bars = plt.bar(['Non-Seizure', 'Seizure'], class_counts.values, color=colors)
    
    # Add value labels on bars
    for bar, count in zip(bars, class_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(class_counts)*0.01,
                str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.title('Class Distribution', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Samples')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Feature Statistics Summary
    plt.figure(figsize=(12, 8))
    
    # Calculate feature statistics by class
    stats_by_class = features_df.groupby('label')[feature_cols[:10]].mean()
    
    x = np.arange(len(feature_cols[:10]))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, stats_by_class.loc[0], width, label='Non-Seizure', alpha=0.8)
    bars2 = plt.bar(x + width/2, stats_by_class.loc[1], width, label='Seizure', alpha=0.8)
    
    plt.xlabel('Features')
    plt.ylabel('Mean Values')
    plt.title('Mean Feature Values by Class (Top 10 Features)')
    plt.xticks(x, [col.replace('_', '\n') for col in feature_cols[:10]], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Visualizations saved successfully!")

# Main processing function
def process_segments_to_features(seizure_segments, non_seizure_segments, output_dir="processed_data"):
    all_segments = [(seg, 1) for seg in seizure_segments] + [(seg, 0) for seg in non_seizure_segments]

    print(f"\n🧠 Extracting features using {cpu_count()} CPU cores...")
    with Pool(processes=cpu_count()) as pool:
        all_features = list(tqdm(pool.imap(extract_features_wrapper, all_segments), total=len(all_segments)))

    features_df = pd.DataFrame(all_features)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save features
    features_df.to_csv(os.path.join(output_dir, 'eeg_features.csv'), index=False)
    with open(os.path.join(output_dir, 'eeg_features.pkl'), 'wb') as f:
        pickle.dump(features_df, f)

    print(f"\n✅ Feature extraction completed!")
    print(f"  🔢 Total samples: {len(features_df)}")
    print(f"  ⚡ Seizure samples: {sum(features_df['label'])}")
    print(f"  🔵 Non-seizure samples: {len(features_df) - sum(features_df['label'])}")
    print(f"  📊 Total features: {len(features_df.columns) - 4}")  # Excluding metadata
    print(f"  💾 Features saved to: {output_dir}/eeg_features.csv")
    
    # Create visualizations
    visualize_features(features_df, output_dir)
    
    return features_df

# Entry point
if __name__ == "__main__":
    try:
        with open('processed_data/seizure_segments.pkl', 'rb') as f:
            seizure_segments = pickle.load(f)
        with open('processed_data/non_seizure_segments.pkl', 'rb') as f:
            non_seizure_segments = pickle.load(f)

        print(f"📦 Loaded {len(seizure_segments)} seizure segments")
        print(f"📦 Loaded {len(non_seizure_segments)} non-seizure segments")
        process_segments_to_features(seizure_segments, non_seizure_segments)

    except FileNotFoundError:
        print("❌ Segment files not found. Please run extract_eeg_segments.py first.")