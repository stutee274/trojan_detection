import os
import glob
import numpy as np
import pandas as pd
from math import log2
import statistics

def parse_vcd(file_path):
    """Simplified VCD parser that extracts signal transitions"""
    signals = {}
    current_time = 0
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                current_time = int(line[1:])
            elif line.startswith('b'):
                parts = line.split()
                if len(parts) >= 2:
                    value = parts[0][1:]
                    signal_id = parts[1]
                    if signal_id not in signals:
                        signals[signal_id] = []
                    signals[signal_id].append((current_time, value))
    return signals
def calculate_features(signal_data):
    """Calculate features with proper glitch detection syntax"""
    if not signal_data or len(signal_data) < 2:
        return None
    
    times = [t for t, _ in signal_data]
    values = [v for _, v in signal_data]
    
    # Time statistics
    duration = max(times) - min(times)
    
    # Corrected glitch detection (transitions < 2ns apart)
    glitches = sum(1 for i in range(1, len(times)) if (times[i] - times[i-1]) < 2000)
    
    # Value changes
    jumps = []
    prev_val = None
    for val in values:
        try:
            num = int(val, 2)
            if prev_val is not None:
                jumps.append(abs(num - prev_val))
            prev_val = num
        except:
            continue

    return {
        'toggles': len(signal_data) - 1,
        'toggle_rate': (len(signal_data) - 1) / (duration / 1e6) if duration > 0 else 0,
        'glitches': glitches,
        'avg_jump': statistics.mean(jumps) if jumps else 0,
        'jump_max': max(jumps) if jumps else 0,
        'bitflips': sum(1 for i in range(1, len(values)) if values[i] != values[i-1]),
    }


def get_thresholds(df, feature):
    """Calculate mean thresholds for trusted vs trojan"""
    trusted = df[df['label'] == 0][feature]
    trojan = df[df['label'] == 1][feature]
    return {
        'feature': feature,
        'trusted_mean': trusted.mean(),
        'trojan_mean': trojan.mean(),
        'threshold': (trusted.mean() + trojan.mean()) / 2,
        'separation': abs(trusted.mean() - trojan.mean())
    }

def main():
    # Process all VCD files
    features = []
    for file in glob.glob('vcd_dumps/*.vcd'):
        signals = parse_vcd(file)
        label = 1 if 'trojan' in os.path.basename(file).lower() else 0
        signal_id = '$' if label else '#'
        
        if signal_id in signals:
            feats = calculate_features(signals[signal_id])
            if feats:
                feats.update({'file': os.path.basename(file), 'label': label})
                features.append(feats)
    
    if not features:
        print("No valid signal data found in any files")
        return
    
    df = pd.DataFrame(features)
    
    # Calculate and print thresholds
    print("\nFeature Threshold Analysis:")
    print("="*50)
    for feature in ['toggles', 'toggle_rate', 'glitches', 'avg_jump', 'jump_max', 'bitflips']:
        if feature in df.columns:
            thresholds = get_thresholds(df, feature)
            print(f"{feature:>12}: Trusted {thresholds['trusted_mean']:.2f} | "
                  f"Trojan {thresholds['trojan_mean']:.2f} | "
                  f"Threshold: {thresholds['threshold']:.2f} | "
                  f"Separation: {thresholds['separation']:.2f}")
    
    # Save results
    df.to_csv('trojan_features.csv', index=False)
    print("\nResults saved to trojan_features.csv")

if __name__ == "__main__":
    main()