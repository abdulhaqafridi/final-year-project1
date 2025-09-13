import os
import pyedflib

def inspect_edf_files(base_path, patient_ids):
    for patient in patient_ids:
        patient_path = os.path.join(base_path, patient)
        edf_files = [f for f in os.listdir(patient_path) if f.endswith(".edf")]

        print(f"\n--- Patient: {patient} ---")
        for edf_file in edf_files:
            edf_path = os.path.join(patient_path, edf_file)
            try:
                f = pyedflib.EdfReader(edf_path)
                n_channels = f.signals_in_file
                signal_labels = f.getSignalLabels()
                duration = f.getFileDuration()
                sample_freqs = f.getSampleFrequencies()

                print(f"File: {edf_file}")
                print(f"  Channels: {n_channels}")
                print(f"  Duration: {duration:.2f} seconds")
                print(f"  Sample Frequencies: {sample_freqs}")
                print(f"  Labels: {signal_labels[:5]}...")  # preview first 5
                f._close()
            except Exception as e:
                print(f"Error reading {edf_file}: {e}")

if __name__ == "__main__":
    base_dataset_path = r"D:\Saqib\AbdulHaq\siena-scalap dataset half"
    selected_patients = ['PN00', 'PN01', 'PN03']
    inspect_edf_files(base_dataset_path, selected_patients)
