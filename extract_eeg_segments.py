import os
import numpy as np
import pyedflib
import re
from datetime import datetime, timedelta
from scipy.signal import butter, filtfilt, resample

def parse_time_str(time_str):
    time_str = time_str.replace('.', ':').strip()
    return datetime.strptime(time_str, "%H:%M:%S")

def extract_time(line):
    match = re.search(r"(\d{1,2}[:.]\d{1,2}[:.]\d{1,2})", line)
    return match.group(1) if match else None

def get_seizure_events(base_path, patient_id):
    """Extract seizure events from seizure list file"""
    seizure_file = os.path.join(base_path, patient_id, f"Seizures-list-{patient_id}.txt")
    
    if not os.path.exists(seizure_file):
        print(f"No seizure list found for {patient_id}")
        return []
    
    seizure_events = []
    
    try:
        with open(seizure_file, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        reg_start_time = None
        current_edf = None
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            if "file name" in line_lower:
                match = re.search(r"file name\s*:\s*(.+\.edf)", line, re.IGNORECASE)
                if match:
                    current_edf = match.group(1).strip()
                    reg_start_time = None
                    
            elif "registration start time" in line_lower:
                time_str = extract_time(line)
                if time_str:
                    reg_start_time = parse_time_str(time_str)
                    
            elif ("seizure start time" in line_lower or line_lower.startswith("start time")) and current_edf:
                start_text = extract_time(line)
                end_text = None
                
                # Try to get end from next line
                if i + 1 < len(lines):
                    next_line = lines[i + 1].lower()
                    if ("seizure end time" in next_line or next_line.startswith("end time")):
                        end_text = extract_time(lines[i + 1])
                
                if start_text and end_text and reg_start_time:
                    seizure_start = parse_time_str(start_text)
                    seizure_end = parse_time_str(end_text)
                    
                    if seizure_start < reg_start_time:
                        seizure_start += timedelta(days=1)
                    if seizure_end < reg_start_time:
                        seizure_end += timedelta(days=1)
                    
                    start_sec = (seizure_start - reg_start_time).total_seconds()
                    end_sec = (seizure_end - reg_start_time).total_seconds()
                    seizure_events.append((current_edf, start_sec, end_sec))
                    
    except Exception as e:
        print(f"Error reading seizure file for {patient_id}: {e}")
    
    return seizure_events

def apply_bandpass_filter(data, low_freq=0.5, high_freq=50, fs=256, order=4):
    """Apply bandpass filter to EEG data"""
    nyquist = fs / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

def extract_windows(signal_data, window_size_samples, overlap_samples):
    """Extract overlapping windows from signal data"""
    windows = []
    start = 0
    
    while start + window_size_samples <= len(signal_data):
        window = signal_data[start:start + window_size_samples]
        windows.append(window)
        start += (window_size_samples - overlap_samples)
    
    return np.array(windows)

def extract_seizure_segments(base_path, patient_ids, window_size=4, overlap=0.5, target_fs=256):
    """Extract seizure and non-seizure segments from EEG files"""
    
    all_seizure_segments = []
    all_non_seizure_segments = []
    
    for patient_id in patient_ids:
        print(f"\n--- Processing Patient: {patient_id} ---")
        
        # Get seizure events
        seizure_events = get_seizure_events(base_path, patient_id)
        print(f"Found {len(seizure_events)} seizure events")
        
        # Get EDF files for this patient
        patient_path = os.path.join(base_path, patient_id)
        edf_files = [f for f in os.listdir(patient_path) if f.endswith(".edf")]
        
        for edf_file in edf_files:
            edf_path = os.path.join(patient_path, edf_file)
            print(f"Processing: {edf_file}")
            
            try:
                # Read EDF file
                f = pyedflib.EdfReader(edf_path)
                n_channels = f.signals_in_file
                sample_freqs = f.getSampleFrequencies()
                duration = f.getFileDuration()
                
                # Read all signals
                signals = []
                for i in range(n_channels):
                    signal_data = f.readSignal(i)
                    signals.append(signal_data)
                
                f._close()
                
                # Convert to numpy array (channels x samples)
                signals = np.array(signals)
                
                # Use the first channel's sampling frequency as reference
                original_fs = int(sample_freqs[0])
                
                # Resample if necessary
                if original_fs != target_fs:
                    resample_ratio = target_fs / original_fs
                    new_length = int(signals.shape[1] * resample_ratio)
                    signals = resample(signals, new_length, axis=1)
                    print(f"Resampled from {original_fs}Hz to {target_fs}Hz")
                
                # Apply bandpass filter
                signals = apply_bandpass_filter(signals.T, fs=target_fs).T
                
                # Calculate window parameters
                window_size_samples = int(window_size * target_fs)
                overlap_samples = int(window_size_samples * overlap)
                
                # Find seizure events for this file
                file_seizures = [s for s in seizure_events if s[0] == edf_file]
                
                # Extract seizure segments
                for _, start_sec, end_sec in file_seizures:
                    start_sample = int(start_sec * target_fs)
                    end_sample = int(end_sec * target_fs)
                    
                    # Ensure we don't exceed signal length
                    start_sample = max(0, start_sample)
                    end_sample = min(signals.shape[1], end_sample)
                    
                    if end_sample > start_sample:
                        seizure_segment = signals[:, start_sample:end_sample]
                        
                        # Extract windows from seizure segment
                        if seizure_segment.shape[1] >= window_size_samples:
                            for ch in range(seizure_segment.shape[0]):
                                windows = extract_windows(seizure_segment[ch], window_size_samples, overlap_samples)
                                for window in windows:
                                    all_seizure_segments.append({
                                        'patient': patient_id,
                                        'file': edf_file,
                                        'channel': ch,
                                        'data': window,
                                        'label': 1  # Seizure
                                    })
                
                # Extract non-seizure segments
                # Create a mask for seizure periods
                seizure_mask = np.zeros(signals.shape[1], dtype=bool)
                for _, start_sec, end_sec in file_seizures:
                    start_sample = max(0, int(start_sec * target_fs))
                    end_sample = min(signals.shape[1], int(end_sec * target_fs))
                    seizure_mask[start_sample:end_sample] = True
                
                # Extract non-seizure windows
                non_seizure_signal = signals[:, ~seizure_mask]
                
                if non_seizure_signal.shape[1] >= window_size_samples:
                    # Limit non-seizure segments to balance dataset
                    max_non_seizure = len([s for _, s_start, s_end in file_seizures 
                                         for s in all_seizure_segments if s['file'] == edf_file]) * 2
                    
                    non_seizure_count = 0
                    for ch in range(non_seizure_signal.shape[0]):
                        windows = extract_windows(non_seizure_signal[ch], window_size_samples, overlap_samples)
                        for window in windows[:max_non_seizure//n_channels]:
                            all_non_seizure_segments.append({
                                'patient': patient_id,
                                'file': edf_file,
                                'channel': ch,
                                'data': window,
                                'label': 0  # Non-seizure
                            })
                            non_seizure_count += 1
                            if non_seizure_count >= max_non_seizure:
                                break
                        if non_seizure_count >= max_non_seizure:
                            break
                
                print(f"  Extracted segments from {edf_file}")
                
            except Exception as e:
                print(f"Error processing {edf_file}: {e}")
    
    print(f"\nTotal seizure segments: {len(all_seizure_segments)}")
    print(f"Total non-seizure segments: {len(all_non_seizure_segments)}")
    
    return all_seizure_segments, all_non_seizure_segments

if __name__ == "__main__":
    base_dataset_path = r"D:\Saqib\AbdulHaq\siena-scalap dataset half"
    selected_patients = ['PN00', 'PN01', 'PN03']
    
    # Extract segments
    seizure_segments, non_seizure_segments = extract_seizure_segments(
        base_dataset_path, 
        selected_patients,
        window_size=4,
        overlap=0.5,
        target_fs=256
    )
    
    # Save the segments
    import pickle
    
    # Create output directory
    output_dir = "processed_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save segments
    with open(os.path.join(output_dir, 'seizure_segments.pkl'), 'wb') as f:
        pickle.dump(seizure_segments, f)
    
    with open(os.path.join(output_dir, 'non_seizure_segments.pkl'), 'wb') as f:
        pickle.dump(non_seizure_segments, f)
    
    print(f"\nData saved to {output_dir}/")
    print("seizure_segments.pkl - Contains all seizure segments")
    print("non_seizure_segments.pkl - Contains all non-seizure segments")