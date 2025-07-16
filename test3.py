import numpy as np
import pandas as pd
from math import log2
from collections import defaultdict
import glob
import os

def parse_vcd(filepath):
    signals = defaultdict(list)
    current_time = 0
    parsing_values = False
    signal_names = {}

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('$enddefinitions'):
                parsing_values = True
                continue

            if not parsing_values:
                if line.startswith('$var'):
                    parts = line.split()
                    if len(parts) >= 5:
                        signal_names[parts[3]] = parts[4:]
                continue

            if line.startswith('#'):
                current_time = int(line[1:])
                continue

            if line.startswith('b'):
                parts = line.split()
                if len(parts) >= 2:
                    value = parts[0][1:]
                    signal_id = parts[1]
                    signals[signal_id].append((current_time, value))
            elif line and line[0] in '01xz':
                value = line[0]
                signal_id = line[1:]
                signals[signal_id].append((current_time, value))

    return signals, signal_names

def identify_signal(signal_names, is_trusted):
    for symbol, parts in signal_names.items():
        parts_str = ' '.join(parts).lower()
        if is_trusted and 'count' in parts_str and 'trust' in parts_str:
            return symbol
        elif not is_trusted and 'count' in parts_str and 'trojan' in parts_str:
            return symbol

    for symbol, parts in signal_names.items():
        if any('count' in part.lower() for part in parts):
            return symbol
    return None

def calculate_entropy(values):
    diffs = np.diff(values)
    unique, counts = np.unique(diffs, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

def calculate_features(times, values):
    samples = len(times)
    if samples < 2:
        return None

    times = np.array(times)
    values = np.array(values)
    duration = times[-1] - times[0] if samples > 1 else 0

    ints = []
    for val in values:
        try:
            ints.append(int(val, 2))
        except:
            ints.append(0)
    ints = np.array(ints)

    raw_len = len(values)
    diffs = np.diff(ints) if samples > 1 else np.array([0])
    abs_diffs = np.abs(diffs)
    bitflips = [bin(a ^ b).count('1') for a, b in zip(ints[:-1], ints[1:])] if samples > 1 else [0]

    return {
        'samples': samples,
        'min_time': times.min() if samples > 0 else 0,
        'max_time': times.max() if samples > 0 else 0,
        'duration_ps': duration,
        'toggles': np.count_nonzero(diffs),
        'toggle_rate': np.count_nonzero(diffs) / duration if duration > 0 else 0,
        'avg_value': np.mean(ints) if samples > 0 else 0,
        'glitches': raw_len - samples,
        'mean_time_diff': duration / samples if samples > 1 else 0,
        'avg_jump': np.mean(abs_diffs) if samples > 1 else 0,
        'jump_std': np.std(abs_diffs) if samples > 1 else 0,
        'jump_max': np.max(abs_diffs) if samples > 1 else 0,
        'bitflip_rate': np.mean(bitflips) if samples > 1 else 0,
        'entropy': calculate_entropy(ints) if samples > 1 else 0,
    }

def process_vcd_files(vcd_dir):
    all_data = []
    for filepath in glob.glob(os.path.join(vcd_dir, '*.vcd')):
        filename = os.path.basename(filepath)
        signals, signal_names = parse_vcd(filepath)
        is_trusted = 'trusted' in filename.lower()
        label = 0 if is_trusted else 1
        signal_id = identify_signal(signal_names, is_trusted)

        if signal_id is None or signal_id not in signals:
            print(f"Warning: Could not identify target signal in {filename}")
            continue

        times, values = zip(*signals[signal_id])
        features = calculate_features(times, values)

        if features:
            features.update({
                'file': filename,
                'signal': ' '.join(signal_names[signal_id]),
                'label': label
            })
            all_data.append(features)

    if not all_data:
        return None

    df = pd.DataFrame(all_data)
    thresholds = calculate_thresholds(df)

    df['auto_label'] = 0
    for feat, thresh in thresholds.items():
        df.loc[df[feat] > thresh, 'auto_label'] = 1

    return df

def calculate_thresholds(df):
    trusted = df[(df['signal'].str.contains('trusted')) | (df['file'].str.contains('trusted'))]
    trojan = df[df['signal'].str.contains('trojan')]

    features = [
        'toggle_rate', 'avg_jump', 'jump_std', 
        'jump_max', 'glitches', 'mean_time_diff',
        'bitflip_rate', 'entropy'
    ]

    thresholds = {}

    for feat in features:
        trusted_vals = trusted[feat]
        trojan_vals = trojan[feat]

        mean_trusted = trusted_vals.mean()
        std_trusted = trusted_vals.std()
        threshold1 = mean_trusted + 3 * std_trusted
        threshold2 = np.percentile(trusted_vals, 90)
        mean_trojan = trojan_vals.mean()
        threshold3 = (mean_trusted + mean_trojan) / 2

        final_threshold = max(threshold1, threshold2, threshold3)
        if feat == 'glitches':
            final_threshold = 0.5

        print(f"\nFeature: {feat}")
        print(f"Trusted - Mean: {mean_trusted:.4f}, Std: {std_trusted:.4f}")
        print(f"Trojan  - Mean: {mean_trojan:.4f}")
        print(f"Thresholds - Method1: {threshold1:.4f}, Method2: {threshold2:.4f}, Method3: {threshold3:.4f}")
        print(f"Selected threshold: {final_threshold:.4f}")

        thresholds[feat] = final_threshold

    return thresholds

def main():
    vcd_dir = 'vcd_dumps'
    output_csv = 'trojan_dataset22.csv'
    print(f"Processing VCD files in '{vcd_dir}'...")
    df = process_vcd_files(vcd_dir)

    if df is not None:
        df.to_csv(output_csv, index=False)
        print(f"\n✅ Dataset saved to {output_csv}")
        print("\n📊 Label Distribution:")
        print(f"Original labels - Trusted: {sum(df['label'] == 0)}, Trojan: {sum(df['label'] == 1)}")
        print(f"Auto labels    - Trusted: {sum(df['auto_label'] == 0)}, Trojan: {sum(df['auto_label'] == 1)}")
        print("\n🔍 Sample Data:")
        print(df.head())
    else:
        print("⚠️ No valid VCD files found or processed.")

if __name__ == "__main__":
    main()
