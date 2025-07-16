import numpy as np
import pandas as pd
from collections import defaultdict
import glob
import os
from math import log2

def parse_vcd(filepath):
    """Enhanced VCD parser with robust signal extraction"""
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

def identify_signal(signal_names, var_types, filename):
    """Smart signal identification with priority to counters"""
    # Priority 1: Look for counter signals
    counter_signals = [s for s, n in signal_names.items() 
                      if 'count' in n.lower() and var_types.get(s) in ['wire', 'reg']]
    
    # Priority 2: Look for clock signals
    clock_signals = [s for s, n in signal_names.items() 
                    if 'clk' in n.lower() and var_types.get(s) in ['wire', 'reg']]
    
    # Return first counter signal if found, otherwise first clock signal
    return counter_signals[0] if counter_signals else (clock_signals[0] if clock_signals else None)

def calculate_features(times, values):
    """Comprehensive feature calculation with validation"""
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
    
    # Glitch detection (2ns threshold)
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
    
    # Prepare feature dictionary
    features = {
        'samples': len(times),
        'min_time': min_t,
        'max_time': max_t,
        'duration_ps': duration,
        'toggles': np.count_nonzero(diffs),
        'toggle_rate': np.count_nonzero(diffs) / (duration / 1e6) if duration > 0 else 0,
        'avg_value': np.mean(ints),
        'median_value': np.median(ints),
        'value_range': np.ptp(ints),
        'glitches': glitches,
        'mean_time_diff': np.mean(time_diffs) if len(time_diffs) > 0 else 0,
        'min_time_diff': np.min(time_diffs) if len(time_diffs) > 0 else 0,
        'max_time_diff': np.max(time_diffs) if len(time_diffs) > 0 else 0,
        'avg_jump': np.mean(abs_diffs) if len(abs_diffs) > 0 else 0,
        'median_jump': np.median(abs_diffs) if len(abs_diffs) > 0 else 0,
        'jump_std': np.std(abs_diffs) if len(abs_diffs) > 1 else 0,
        'jump_max': np.max(abs_diffs) if len(abs_diffs) > 0 else 0,
        'bitflip_rate': bitflip_rate,
        'entropy': entropy,
        'unique_values': len(value_counts)
    }
    
    return features

def process_vcd_files(vcd_dir):
    """Process all VCD files with intelligent signal selection"""
    all_data = []
    
    for filepath in glob.glob(os.path.join(vcd_dir, '*.vcd')):
        filename = os.path.basename(filepath)
        is_trusted = 'trusted' in filename.lower()
        
        signals, signal_names, var_types = parse_vcd(filepath)
        if not signals:
            continue
            
        # Smart signal identification
        signal_id = identify_signal(signal_names, var_types, filename)
        if not signal_id:
            continue
            
        times, values = zip(*signals[signal_id])
        
        features = calculate_features(times, values)
        if features:
            features.update({
                'file': filename,
                'signal': signal_names.get(signal_id, 'unknown'),
                'signal_type': var_types.get(signal_id, 'unknown'),
                'label': 0 if is_trusted else 1
            })
            all_data.append(features)
    
    # Define column order for CSV
    columns = [
        'samples', 'min_time', 'max_time', 'duration_ps',
        'toggles', 'toggle_rate', 'avg_value', 'median_value', 'value_range',
        'glitches', 'mean_time_diff', 'min_time_diff', 'max_time_diff',
        'avg_jump', 'median_jump', 'jump_std', 'jump_max',
        'bitflip_rate', 'entropy', 'unique_values',
        'file', 'signal', 'signal_type', 'label'
    ]
    
    return pd.DataFrame(all_data, columns=columns)

def main():
    vcd_dir = 'vcd_dumps'
    output_dir = 'output'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df = process_vcd_files(vcd_dir)
    
    if df is not None:
        csv_path = os.path.join(output_dir, 'trojan_dataset_combined.csv')
        df.to_csv(csv_path, index=False)
        print(f"CSV file saved to: {csv_path}")
        
        # Also save to current directory
        df.to_csv('trojan_dataset_combined.csv', index=False)
        print("\nDataset with all features saved to trojan_dataset_combined.csv")
    
    else:
        print("No data processed - no CSV file generated")

if __name__ == "__main__":
    main()