import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import os
from scipy import signal

def plot_raw_eeg_segments(seizure_segments, non_seizure_segments, n_samples=5):
    """Plot raw EEG segments for comparison"""
    
    fig, axes = plt.subplots(2, n_samples, figsize=(20, 8))
    fig.suptitle('Raw EEG Segments Comparison', fontsize=16)
    
    # Plot seizure segments
    for i in range(min(n_samples, len(seizure_segments))):
        segment = seizure_segments[i]
        axes[0, i].plot(segment['data'])
        axes[0, i].set_title(f'Seizure - {segment["patient"]}\nCh{segment["channel"]}')
        axes[0, i].set_ylabel('Amplitude (μV)')
        axes[0, i].grid(True, alpha=0.3)
    
    # Plot non-seizure segments
    for i in range(min(n_samples, len(non_seizure_segments))):
        segment = non_seizure_segments[i]
        axes[1, i].plot(segment['data'])
        axes[1, i].set_title(f'Non-Seizure - {segment["patient"]}\nCh{segment["channel"]}')
        axes[1, i].set_ylabel('Amplitude (μV)')
        axes[1, i].set_xlabel('Samples')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('processed_data/raw_eeg_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_frequency_analysis(seizure_segments, non_seizure_segments, fs=256, n_samples=3):
    """Plot frequency domain analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Frequency Domain Analysis', fontsize=16)
    
    # Calculate PSDs for seizure segments
    seizure_psds = []
    for i in range(min(n_samples, len(seizure_segments))):
        freqs, psd = signal.welch(seizure_segments[i]['data'], fs=fs, nperseg=fs)
        seizure_psds.append(psd)
        axes[0, 0].semilogy(freqs, psd, alpha=0.7, label=f'Seizure {i+1}')
    
    axes[0, 0].set_title('Seizure Segments - PSD')
    axes[0, 0].set_xlabel('Frequency (Hz)')
    axes[0, 0].set_ylabel('Power Spectral Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(0, 50)
    
    # Calculate PSDs for non-seizure segments
    non_seizure_psds = []
    for i in range(min(n_samples, len(non_seizure_segments))):
        freqs, psd = signal.welch(non_seizure_segments[i]['data'], fs=fs, nperseg=fs)
        non_seizure_psds.append(psd)
        axes[0, 1].semilogy(freqs, psd, alpha=0.7, label=f'Non-Seizure {i+1}')
    
    axes[0, 1].set_title('Non-Seizure Segments - PSD')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Power Spectral Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0, 50)
    
    # Average PSD comparison
    if seizure_psds and non_seizure_psds:
        avg_seizure_psd = np.mean(seizure_psds, axis=0)
        avg_non_seizure_psd = np.mean(non_seizure_psds, axis=0)
        
        axes[1, 0].semilogy(freqs, avg_seizure_psd, 'r-', linewidth=2, label='Seizure Average')
        axes[1, 0].semilogy(freqs, avg_non_seizure_psd, 'b-', linewidth=2, label='Non-Seizure Average')
        axes[1, 0].set_title('Average PSD Comparison')
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('Power Spectral Density')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim(0, 50)
        
        # Frequency band power comparison
        bands = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30), 'Gamma': (30, 50)}
        band_names = list(bands.keys())
        seizure_powers = []
        non_seizure_powers = []
        
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs <= high)
            seizure_power = np.trapz(avg_seizure_psd[mask], freqs[mask])
            non_seizure_power = np.trapz(avg_non_seizure_psd[mask], freqs[mask])
            seizure_powers.append(seizure_power)
            non_seizure_powers.append(non_seizure_power)
        
        x = np.arange(len(band_names))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, seizure_powers, width, label='Seizure', color='red', alpha=0.7)
        axes[1, 1].bar(x + width/2, non_seizure_powers, width, label='Non-Seizure', color='blue', alpha=0.7)
        axes[1, 1].set_title('Frequency Band Power Comparison')
        axes[1, 1].set_xlabel('Frequency Bands')
        axes[1, 1].set_ylabel('Power')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(band_names)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('processed_data/frequency_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_distributions(features_df):
    """Plot feature distributions"""
    
    # Select some important features for visualization
    important_features = [
        'time_mean', 'time_std', 'time_skewness', 'time_kurtosis',
        'freq_delta_power', 'freq_theta_power', 'freq_alpha_power', 'freq_beta_power',
        'wavelet_cD1_energy', 'wavelet_cD2_energy', 'nonlinear_approximate_entropy'
    ]
    
    # Filter features that exist in the dataframe
    available_features = [f for f in important_features if f in features_df.columns]
    
    if not available_features:
        print("No matching features found for visualization")
        return
    
    n_features = len(available_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    fig.suptitle('Feature Distributions: Seizure vs Non-Seizure', fontsize=16)
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, feature in enumerate(available_features):
        row = i // n_cols
        col = i % n_cols
        
        seizure_data = features_df[features_df['label'] == 1][feature]
        non_seizure_data = features_df[features_df['label'] == 0][feature]
        
        axes[row, col].hist(seizure_data, bins=30, alpha=0.7, label='Seizure', color='red', density=True)
        axes[row, col].hist(non_seizure_data, bins=30, alpha=0.7, label='Non-Seizure', color='blue', density=True)
        axes[row, col].set_title(feature)
        axes[row, col].set_xlabel('Feature Value')
        axes[row, col].set_ylabel('Density')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_features, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('processed_data/feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_correlation(features_df):
    """Plot feature correlation matrix"""
    
    # Select numerical features only
    numerical_features = features_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove metadata columns
    metadata_cols = ['patient', 'file', 'channel', 'label']
    feature_cols = [col for col in numerical_features if col not in metadata_cols]
    
    if len(feature_cols) > 50:  # Limit to first 50 features for readability
        feature_cols = feature_cols[:50]
    
    correlation_matrix = features_df[feature_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, square=True, 
                xticklabels=False, yticklabels=False)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('processed_data/feature_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_patient_statistics(features_df):
    """Plot patient-wise statistics"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Patient-wise Statistics', fontsize=16)
    
    # Patient distribution
    patient_counts = features_df['patient'].value_counts()
    axes[0, 0].bar(patient_counts.index, patient_counts.values)
    axes[0, 0].set_title('Total Segments per Patient')
    axes[0, 0].set_xlabel('Patient ID')
    axes[0, 0].set_ylabel('Number of Segments')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Seizure vs Non-seizure per patient
    patient_label_counts = pd.crosstab(features_df['patient'], features_df['label'])
    patient_label_counts.plot(kind='bar', ax=axes[0, 1], color=['blue', 'red'])
    axes[0, 1].set_title('Seizure vs Non-Seizure Segments per Patient')
    axes[0, 1].set_xlabel('Patient ID')
    axes[0, 1].set_ylabel('Number of Segments')
    axes[0, 1].legend(['Non-Seizure', 'Seizure'])
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=0)
    
    # Channel distribution
    if 'channel' in features_df.columns:
        channel_counts = features_df['channel'].value_counts().sort_index()
        axes[1, 0].bar(channel_counts.index, channel_counts.values)
        axes[1, 0].set_title('Segments per Channel')
        axes[1, 0].set_xlabel('Channel Number')
        axes[1, 0].set_ylabel('Number of Segments')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Label distribution
    label_counts = features_df['label'].value_counts()
    axes[1, 1].pie(label_counts.values, labels=['Non-Seizure', 'Seizure'], 
                   colors=['blue', 'red'], autopct='%1.1f%%')
    axes[1, 1].set_title('Overall Label Distribution')
    
    plt.tight_layout()
    plt.savefig('processed_data/patient_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_data_summary(features_df):
    """Generate and save data summary"""
    
    summary = {
        'Dataset Overview': {
            'Total Samples': len(features_df),
            'Seizure Samples': len(features_df[features_df['label'] == 1]),
            'Non-Seizure Samples': len(features_df[features_df['label'] == 0]),
            'Total Features': len(features_df.columns) - 4,  # Excluding metadata
            'Patients': features_df['patient'].nunique(),
            'Patient IDs': list(features_df['patient'].unique())
        },
        'Class Distribution': {
            'Seizure Percentage': f"{(len(features_df[features_df['label'] == 1]) / len(features_df)) * 100:.2f}%",
            'Non-Seizure Percentage': f"{(len(features_df[features_df['label'] == 0]) / len(features_df)) * 100:.2f}%"
        },
        'Patient-wise Distribution': {}
    }
    
    # Patient-wise breakdown
    for patient in features_df['patient'].unique():
        patient_data = features_df[features_df['patient'] == patient]
        seizure_count = len(patient_data[patient_data['label'] == 1])
        non_seizure_count = len(patient_data[patient_data['label'] == 0])
        
        summary['Patient-wise Distribution'][patient] = {
            'Total Segments': len(patient_data),
            'Seizure Segments': seizure_count,
            'Non-Seizure Segments': non_seizure_count,
            'Seizure Ratio': f"{(seizure_count / len(patient_data)) * 100:.2f}%" if len(patient_data) > 0 else "0%"
        }
    
    # Feature statistics
    numerical_cols = features_df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numerical_cols if col not in ['label']]
    
    if feature_cols:
        feature_stats = features_df[feature_cols].describe()
        summary['Feature Statistics'] = {
            'Mean Values Range': f"{feature_stats.loc['mean'].min():.4f} to {feature_stats.loc['mean'].max():.4f}",
            'Std Values Range': f"{feature_stats.loc['std'].min():.4f} to {feature_stats.loc['std'].max():.4f}",
            'Features with High Variance': len(feature_stats.columns[feature_stats.loc['std'] > feature_stats.loc['std'].median()])
        }
    
    # Save summary
    summary_text = ""
    for section, content in summary.items():
        summary_text += f"\n{'='*50}\n{section}\n{'='*50}\n"
        if isinstance(content, dict):
            for key, value in content.items():
                if isinstance(value, dict):
                    summary_text += f"\n{key}:\n"
                    for sub_key, sub_value in value.items():
                        summary_text += f"  {sub_key}: {sub_value}\n"
                else:
                    summary_text += f"{key}: {value}\n"
        else:
            summary_text += f"{content}\n"
    
    with open('processed_data/data_summary.txt', 'w') as f:
        f.write(summary_text)
    
    print("Data Summary:")
    print(summary_text)
    
    return summary

if __name__ == "__main__":
    # Create output directory
    os.makedirs('processed_data', exist_ok=True)
    
    try:
        # Load segments for raw data visualization
        print("Loading segments for visualization...")
        with open('processed_data/seizure_segments.pkl', 'rb') as f:
            seizure_segments = pickle.load(f)
        
        with open('processed_data/non_seizure_segments.pkl', 'rb') as f:
            non_seizure_segments = pickle.load(f)
        
        # Load features
        print("Loading features...")
        if os.path.exists('processed_data/eeg_features.pkl'):
            with open('processed_data/eeg_features.pkl', 'rb') as f:
                features_df = pickle.load(f)
        else:
            features_df = pd.read_csv('processed_data/eeg_features.csv')
        
        print(f"Loaded {len(seizure_segments)} seizure segments")
        print(f"Loaded {len(non_seizure_segments)} non-seizure segments")
        print(f"Loaded features: {features_df.shape}")
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        
        print("1. Plotting raw EEG segments...")
        plot_raw_eeg_segments(seizure_segments, non_seizure_segments)
        
        print("2. Plotting frequency analysis...")
        plot_frequency_analysis(seizure_segments, non_seizure_segments)
        
        print("3. Plotting feature distributions...")
        plot_feature_distributions(features_df)
        
        print("4. Plotting feature correlation...")
        plot_feature_correlation(features_df)
        
        print("5. Plotting patient statistics...")
        plot_patient_statistics(features_df)
        
        print("6. Generating data summary...")
        generate_data_summary(features_df)
        
        print("\nAll visualizations saved to 'processed_data/' directory!")
        
    except FileNotFoundError as e:
        print(f"Error: Required files not found - {e}")
        print("Please run extract_eeg_segments.py and feature_extraction.py first.")
    except Exception as e:
        print(f"Error during visualization: {e}")