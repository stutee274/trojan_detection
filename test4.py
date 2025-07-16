import numpy as np
import pandas as pd
from math import log2
from collections import defaultdict
import glob
import os

def parse_vcd(filepath):
    """Enhanced VCD parser with better error handling"""
    signals = defaultdict(list)
    current_time = 0
    parsing_values = False
    signal_names = {}
    
    try:
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
                    try:
                        current_time = int(line[1:])
                    except ValueError:
                        continue
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
                    
    except Exception as e:
        print(f"Error parsing {filepath}: {str(e)}")
        return None, None
        
    return signals, signal_names

def identify_signal(signal_names, is_trusted):
    """Improved signal identification with multiple patterns"""
    target_patterns = {
        True: ['count_trusted', 'trusted_count', 'count[3:0]'],
        False: ['count_trojan', 'trojan_count', 'count_troj']
    }
    
    for symbol, parts in signal_names.items():
        parts_str = ' '.join(parts).lower()
        for pattern in target_patterns[is_trusted]:
            if pattern in parts_str:
                return symbol
                
    for symbol, parts in signal_names.items():
        if 'count' in ' '.join(parts).lower():
            return symbol
            
    return None

def calculate_features(times, values):
    """Feature calculation with validation"""
    try:
        samples = len(times)
        if samples < 2:
            return None
            
        times = np.array(times)
        values = np.array(values)
        min_t = np.min(times)
        max_t = np.max(times)
        duration = max_t - min_t
        
        ints = []
        for val in values:
            try:
                ints.append(int(val, 2))
            except:
                ints.append(0)
        ints = np.array(ints)
        
        diffs = np.diff(ints)
        abs_diffs = np.abs(diffs)
        time_diffs = np.diff(times)
        
        glitches = np.sum(time_diffs <= 500)
        bitflips = [bin(a ^ b).count('1') for a, b in zip(ints[:-1], ints[1:])]
        bitflip_rate = np.mean(bitflips) if bitflips else 0
        
        value_counts = {}
        for val in values:
            value_counts[val] = value_counts.get(val, 0) + 1
        probs = [count / samples for count in value_counts.values()]
        entropy = -sum(p * log2(p) for p in probs if p > 0)
        
        return {
            'samples': samples,
            'min_time': min_t,
            'max_time': max_t,
            'duration_ps': duration,
            'toggles': np.count_nonzero(diffs),
            'toggle_rate': np.count_nonzero(diffs) / (duration / 1e6) if duration > 0 else 0,
            'avg_value': np.mean(ints),
            'glitches': glitches,
            'mean_time_diff': np.mean(time_diffs),
            'avg_jump': np.mean(abs_diffs),
            'jump_std': np.std(abs_diffs),
            'jump_max': np.max(abs_diffs),
            'bitflip_rate': bitflip_rate,
            'entropy': entropy
        }
    except Exception as e:
        print(f"Error calculating features: {str(e)}")
        return None

def process_vcd_files(vcd_dir):
    """Main processing function with improved error handling"""
    all_data = []
    processed_files = 0
    
    for filepath in glob.glob(os.path.join(vcd_dir, '*.vcd')):
        filename = os.path.basename(filepath)
        
        signals, signal_names = parse_vcd(filepath)
        if not signals:
            continue
            
        is_trusted = 'trusted' in filename.lower()
        label = 0 if is_trusted else 1
        
        signal_id = identify_signal(signal_names, is_trusted)
        if signal_id is None or signal_id not in signals:
            print(f"Warning: Could not identify target signal in {filename}")
            continue
            
        try:
            times, values = zip(*signals[signal_id])
        except:
            print(f"Warning: No valid signal data in {filename}")
            continue
            
        features = calculate_features(times, values)
        if features:
            features.update({
                'file': filename,
                'signal': ' '.join(signal_names[signal_id]),
                'label': label
            })
            all_data.append(features)
            processed_files += 1
    
    print(f"\nProcessed {processed_files} VCD files successfully")
    if not all_data:
        return None
        
    df = pd.DataFrame(all_data)
    
    # Calculate dynamic thresholds
    thresholds = {
        'glitches': 0.5,  # Any glitch is suspicious
        'avg_jump': df[df['label'] == 0]['avg_jump'].quantile(0.95),
        'jump_std': df[df['label'] == 0]['jump_std'].quantile(0.95),
        'toggle_rate': df[df['label'] == 0]['toggle_rate'].quantile(0.95)
    }
    
    # Auto-labeling
    df['auto_label'] = 0
    trojan_conditions = (
        (df['glitches'] > thresholds['glitches']) |
        (df['avg_jump'] > thresholds['avg_jump']) |
        ((df['jump_std'] > thresholds['jump_std']) & 
         (df['toggle_rate'] > thresholds['toggle_rate']))
    )
    df.loc[trojan_conditions, 'auto_label'] = 1
    
    return df, thresholds

def main():
    vcd_dir = 'vcd_dumps'
    output_csv = 'trojan_dataset44.csv'
    
    print(f"🔍 Processing VCD files in '{vcd_dir}'...")
    result = process_vcd_files(vcd_dir)
    
    if result is not None:
        df, thresholds = result
        df.to_csv(output_csv, index=False)
        print(f"\n✅ Dataset saved to {output_csv}")
        
        print("\n📊 Threshold Values:")
        for k, v in thresholds.items():
            print(f"{k:>12}: {v:.4f}")
            
        print("\n📈 Label Distribution:")
        print(f"Original labels - Trusted: {sum(df['label'] == 0)}, Trojan: {sum(df['label'] == 1)}")
        print(f"Auto labels    - Trusted: {sum(df['auto_label'] == 0)}, Trojan: {sum(df['auto_label'] == 1)}")
        
        print("\n🔍 Sample Data:")
        print(df[['file', 'label', 'auto_label', 'glitches', 'avg_jump', 'toggle_rate']].head())
    else:
        print("⚠️ No valid VCD files found or processed.")

if __name__ == "__main__":
    main()