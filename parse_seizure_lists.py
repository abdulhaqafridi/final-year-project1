import os
import re
from datetime import datetime, timedelta

def parse_time_str(time_str):
    time_str = time_str.replace('.', ':').strip()
    return datetime.strptime(time_str, "%H:%M:%S")

def extract_time(line):
    match = re.search(r"(\d{1,2}[:.]\d{1,2}[:.]\d{1,2})", line)
    return match.group(1) if match else None

def parse_seizure_lists(base_path, patient_ids):
    for patient in patient_ids:
        seizure_file = os.path.join(base_path, patient, f"Seizures-list-{patient}.txt")
        print(f"\n--- Seizure List for {patient} ---")

        if not os.path.exists(seizure_file):
            print("Seizure list file not found.")
            continue

        try:
            with open(seizure_file, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            seizure_events = []
            reg_start_time = None
            current_edf = None
            seizure_start_time = None

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

            print(f"Total seizure events found: {len(seizure_events)}")
            for edf_file, start, end in seizure_events:
                print(f"  File: {edf_file} | Start: {start:.2f}s | End: {end:.2f}s")

        except Exception as e:
            print(f"Error reading {seizure_file}: {e}")

if __name__ == "__main__":
    base_dataset_path = r"D:\Saqib\AbdulHaq\siena-scalap dataset half"
    selected_patients = ['PN00', 'PN01', 'PN03']
    parse_seizure_lists(base_dataset_path, selected_patients)
