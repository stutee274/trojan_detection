import numpy as np
import pandas as pd
from collections import defaultdict
import glob
import os
from math import log2

def parse_vcd(filepath):
    """Basic VCD parser that extracts signal data"""
    signals = defaultdict(list)
    signal_names = {}
    var_types = {}
    current_time = 0
    parsing_values = False
    
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
                        var_type = parts[1]
                        symbol = parts[3]
                        name = ' '.join(parts[4:])
                        signal_names[symbol] = name
                        var_types[symbol] = var_type
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
                    
    return signals, signal_names, var_types

def calculate_features(times, values):
    """Calculate all features from signal data"""
    if len(times) < 2:
        return None
        
    # Convert to numpy arrays
    times = np.array(times, dtype=np.int64)
    values = np.array(values)
    
    # Time statistics
    min_t = np.min(times)
    max_t = np.max(times)
    duration = max_t - min_t
    
    # Value conversion and analysis
    ints = []
    for val in values:
        try:
            ints.append(int(val, 2))
        except:
            ints.append(0)
    ints = np.array(ints)
    
    # Transition analysis
    diffs = np.diff(ints)
    abs_diffs = np.abs(diffs)
    time_diffs = np.diff(times)
    
    # Glitch detection (300ps threshold)
    glitches = np.sum(time_diffs <= 2000)
    
    # Bit flip analysis
    bitflips = []
    for i in range(1, len(values)):
        try:
            if len(values[i]) == len(values[i-1]):
                bitflips.append(bin(int(values[i], 2) ^ int(values[i-1], 2)).count('1'))
        except:
            continue
    bitflip_rate = np.mean(bitflips) if bitflips else 0
    
    # Entropy calculation
    value_counts = {}
    for val in values:
        value_counts[val] = value_counts.get(val, 0) + 1
    probs = [count / len(values) for count in value_counts.values()]
    entropy = -sum(p * log2(p) for p in probs if p > 0)
    
    return {
        'samples': len(times),
        'min_time': min_t,
        'max_time': max_t,
        'duration_ps': duration,
        'toggles': np.count_nonzero(diffs),
        'toggle_rate': np.count_nonzero(diffs) / (duration / 1e6) if duration > 0 else 0,
        'avg_value': np.mean(ints),
        'glitches': glitches,
        'mean_time_diff': np.mean(time_diffs) if len(time_diffs) > 0 else 0,
        'avg_jump': np.mean(abs_diffs) if len(abs_diffs) > 0 else 0,
        'jump_std': np.std(abs_diffs) if len(abs_diffs) > 1 else 0,
        'jump_max': np.max(abs_diffs) if len(abs_diffs) > 0 else 0,
        'bitflip_rate': bitflip_rate,
        'entropy': entropy,
    }

def process_vcd_files(vcd_dir):
    """Process all VCD files in directory and return DataFrame"""
    all_data = []
    
    for filepath in glob.glob(os.path.join(vcd_dir, '*.vcd')):
        filename = os.path.basename(filepath)
        is_trusted = 'trusted' in filename.lower()
        
        signals, signal_names, var_types = parse_vcd(filepath)
        if not signals:
            continue
            
        # Use first signal found (simplified)
        signal_id = next(iter(signals.keys()))
        times, values = zip(*signals[signal_id])
        
        features = calculate_features(times, values)
        if features:
            features.update({
                'file': filename,
                'signal': signal_names.get(signal_id, 'unknown'),
                'label': 0 if is_trusted else 1  # Only original label
            })
            all_data.append(features)
    
    # Create DataFrame with all features in desired order
    columns = [
        'samples', 'min_time', 'max_time', 'duration_ps', 
        'toggles', 'toggle_rate', 'avg_value', 'glitches', 
        'mean_time_diff', 'avg_jump', 'jump_std', 'jump_max',
        'bitflip_rate', 'entropy', 'file', 'signal', 'label'
    ]
    return pd.DataFrame(all_data, columns=columns)

def main():
    vcd_dir = 'vcd_dumps'
    output_dir = 'output'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df = process_vcd_files(vcd_dir)
    
    if df is not None:
        csv_path = os.path.join(output_dir, 'trojan_dataset_23.csv')
        df.to_csv(csv_path, index=False)
        print(f"CSV file saved to: {csv_path}")
        
        # Also save to current directory
        df.to_csv('trojan_dataset_23.csv', index=False)
        print("\nResults saved to trojan_dataset_23.csv with exact feature values")
    
    else:
        print("No data processed - no CSV file generated")

if __name__ == "__main__":
    main()