# import os
# import glob
# import numpy as np
# import pandas as pd
# from math import log2
# import statistics
# from sklearn.metrics import roc_curve

# def parse_vcd(file_path):
#     """Parse VCD file and extract signal transitions with improved glitch detection"""
#     signals = {}
#     current_time = 0
#     parsing_values = False
    
#     with open(file_path, 'r') as f:
#         for line in f:
#             line = line.strip()
            
#             if line.startswith('$enddefinitions'):
#                 parsing_values = True
#                 continue
                
#             if not parsing_values:
#                 continue
                
#             if line.startswith('#'):
#                 current_time = int(line[1:])
#                 continue
                
#             if line.startswith('b'):
#                 # Binary value change
#                 parts = line.split()
#                 if len(parts) >= 2:
#                     value = parts[0][1:]
#                     signal_id = parts[1]
#                     if signal_id not in signals:
#                         signals[signal_id] = []
#                     signals[signal_id].append((current_time, value))
#             elif line.startswith(('0', '1', 'x', 'z')):
#                 # Scalar value change
#                 value = line[0]
#                 signal_id = line[1:]
#                 if signal_id not in signals:
#                     signals[signal_id] = []
#                 signals[signal_id].append((current_time, value))
                
#     return signals

# def calculate_features(signal_data):
#     """Calculate features from signal transitions with improved glitch detection"""
#     if not signal_data or len(signal_data) < 2:
#         return None
    
#     times = [t for t, val in signal_data]
#     values = [val for t, val in signal_data]
    
#     # Basic time statistics
#     min_time = min(times)
#     max_time = max(times)
#     duration_ps = max_time - min_time
    
#     # Toggle count and rate
#     toggles = len(signal_data) - 1
#     toggle_rate = toggles / (duration_ps / 1e6) if duration_ps > 0 else 0  # in toggles/μs
    
#     # Value statistics
#     numeric_values = []
#     for val in values:
#         if val in ('x', 'z'):
#             continue
#         try:
#             numeric_values.append(int(val, 2))
#         except:
#             continue
    
#     avg_value = statistics.mean(numeric_values) if numeric_values else 0
    
#     # Improved glitch detection (rapid toggles within 1ns)
#     glitches = 0
#     time_diffs = []
#     glitch_threshold = 1000  # 1ns in ps
    
#     for i in range(1, len(times)):
#         time_diff = times[i] - times[i-1]
#         time_diffs.append(time_diff)
#         if time_diff < glitch_threshold:
#             glitches += 1
#             # For debugging glitch detection
#             # print(f"Glitch detected at {times[i]}ps: {values[i-1]} -> {values[i]} ({time_diff}ps)")
    
#     mean_time_diff = statistics.mean(time_diffs) if time_diffs else 0
    
#     # Jump analysis (value changes)
#     jumps = []
#     prev_val = None
#     for val in numeric_values:
#         if prev_val is not None:
#             jumps.append(abs(val - prev_val))
#         prev_val = val
    
#     avg_jump = statistics.mean(jumps) if jumps else 0
#     jump_std = statistics.stdev(jumps) if len(jumps) > 1 else 0
#     jump_max = max(jumps) if jumps else 0
    
#     # Bit flip rate
#     bit_flips = 0
#     total_bits = 0
#     for i in range(1, len(values)):
#         if len(values[i]) != len(values[i-1]):
#             continue
#         for b1, b2 in zip(values[i], values[i-1]):
#             if b1 != b2 and b1 in ('0', '1') and b2 in ('0', '1'):
#                 bit_flips += 1
#             if b1 in ('0', '1') and b2 in ('0', '1'):
#                 total_bits += 1
    
#     bitflip_rate = bit_flips / total_bits if total_bits > 0 else 0
    
#     # Entropy calculation
#     value_counts = {}
#     for val in values:
#         if val in ('x', 'z'):
#             continue
#         if val not in value_counts:
#             value_counts[val] = 0
#         value_counts[val] += 1
    
#     total = sum(value_counts.values())
#     entropy = 0
#     for count in value_counts.values():
#         p = count / total
#         entropy -= p * log2(p) if p > 0 else 0
    
#     return {
#         'min_time': min_time,
#         'max_time': max_time,
#         'duration_ps': duration_ps,
#         'toggles': toggles,
#         'toggle_rate': toggle_rate,
#         'avg_value': avg_value,
#         'glitches': glitches,
#         'mean_time_diff': mean_time_diff,
#         'avg_jump': avg_jump,
#         'jump_std': jump_std,
#         'jump_max': jump_max,
#         'bitflip_rate': bitflip_rate,
#         'entropy': entropy
#     }

# def find_optimal_threshold(feature_values, labels):
#     """Find optimal threshold for a feature using ROC analysis"""
#     if len(set(labels)) < 2:
#         return None, 0
    
#     # Calculate ROC curve
#     fpr, tpr, thresholds = roc_curve(labels, feature_values)
    
#     # Find threshold with maximum TPR - FPR
#     optimal_idx = np.argmax(tpr - fpr)
#     optimal_threshold = thresholds[optimal_idx]
    
#     return optimal_threshold, tpr[optimal_idx] - fpr[optimal_idx]

# def process_files(vcd_dir):
#     """Process all VCD files in directory and return feature dataframe"""
#     features_list = []
#     all_signals = []
#     all_labels = []
    
#     # Find all VCD files in directory
#     vcd_files = glob.glob(os.path.join(vcd_dir, '*.vcd'))
    
#     if not vcd_files:
#         print(f"No VCD files found in directory: {vcd_dir}")
#         return pd.DataFrame(), [], []
    
#     print(f"Found {len(vcd_files)} VCD files to process")
    
#     for file_path in vcd_files:
#         try:
#             filename = os.path.basename(file_path)
#             signals = parse_vcd(file_path)
            
#             # Determine label based on filename
#             is_trojan = 'trojan' in filename.lower()
#             label = 1 if is_trojan else 0
            
#             # Find the relevant signal
#             signal_id = '$' if is_trojan else '#'  # $ for trojan, # for trusted
#             if signal_id not in signals:
#                 # Try alternative signal identifiers
#                 for alt_id in ['&', '(']:
#                     if alt_id in signals:
#                         signal_id = alt_id
#                         break
            
#             if signal_id in signals:
#                 signal_data = signals[signal_id]
#                 features = calculate_features(signal_data)
#                 if features:
#                     features['file'] = filename
#                     features['signal'] = signal_id
#                     features['label'] = label
#                     features_list.append(features)
#                     all_signals.append(signal_data)
#                     all_labels.append(label)
#                     print(f"Processed {filename} (label: {label})")
#             else:
#                 print(f"Warning: No relevant signal found in {filename}")
                
#         except Exception as e:
#             print(f"Error processing {file_path}: {str(e)}")
    
#     if not features_list:
#         print("No valid signal data found in any files")
#         return pd.DataFrame(), [], []
    
#     df = pd.DataFrame(features_list)
    
#     # Focus on the most discriminative features
#     key_features = ['glitches', 'avg_jump', 'toggle_rate', 'jump_max', 'bitflip_rate']
    
#     # Determine optimal thresholds for key features
#     optimal_thresholds = {}
#     for feature in key_features:
#         if feature in df.columns:
#             threshold, score = find_optimal_threshold(df[feature], all_labels)
#             if threshold is not None:
#                 optimal_thresholds[feature] = (threshold, score)
#                 print(f"Optimal threshold for {feature}: {threshold:.4f} (score: {score:.4f})")
    
#     # Apply automatic labeling based on key features
#     if optimal_thresholds:
#         best_feature = max(optimal_thresholds.keys(), key=lambda k: optimal_thresholds[k][1])
#         print(f"\nBest discriminating feature: {best_feature}")
        
#         # Create a score based on how many thresholds are exceeded
#         df['trojan_score'] = 0
#         for feature, (threshold, _) in optimal_thresholds.items():
#             df['trojan_score'] += (df[feature] > threshold).astype(int)
        
#         # Label as trojan if score exceeds half the number of features
#         threshold_count = len(optimal_thresholds) // 2
#         df['auto_label'] = (df['trojan_score'] > threshold_count).astype(int)
    
#     return df

# # Main execution
# if __name__ == "__main__":
#     vcd_dir = 'vcd_dumps'  # Directory containing VCD files
    
#     # Process files and generate dataset
#     df = process_files(vcd_dir)
    
#     if not df.empty:
#         # Save to CSV
#         output_file = 'trojan_detection_dataset_new.csv'
#         df.to_csv(output_file, index=False)
#         print(f"\nDataset saved to {output_file}")

#         # Print dataset summary
#         print("\nDataset summary:")
#         print(f"Total samples: {len(df)}")
#         if 'auto_label' in df.columns:
#             print(f"Auto-labeled as trusted (0): {len(df[df['auto_label'] == 0])}")
#             print(f"Auto-labeled as trojan (1): {len(df[df['auto_label'] == 1])}")
#         print(f"Original labels - trusted (0): {len(df[df['label'] == 0])}")
#         print(f"Original labels - trojan (1): {len(df[df['label'] == 1])}")

#         # Show most important features
#         print("\nKey features analysis:")
#         key_features = ['glitches', 'avg_jump', 'toggle_rate', 'jump_max', 'bitflip_rate']
#         for feature in key_features:
#             if feature in df.columns:
#                 trusted_mean = df[df['label'] == 0][feature].mean()
#                 trojan_mean = df[df['label'] == 1][feature].mean()
#                 print(f"{feature}: Trusted {trusted_mean:.2f} vs Trojan {trojan_mean:.2f}")

#         # Show sample of the data
#         print("\nSample data:")
#         print(df.head())
#     else:
#         print("\nNo data to save. Check your VCD files and directory path.")



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