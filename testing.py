import numpy as np
import pandas as pd
from math import log2
from collections import defaultdict
import glob
import os
import warnings
warnings.filterwarnings('ignore')

def parse_vcd(filepath):
    """Enhanced VCD parser with robust error handling and signal extraction"""
    signals = defaultdict(list)
    current_time = 0
    parsing_values = False
    signal_names = {}
    var_types = {}
    
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
                            var_type = parts[1]
                            symbol = parts[3]
                            name = ' '.join(parts[4:])
                            signal_names[symbol] = name
                            var_types[symbol] = var_type
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
        print(f"⛔ Error parsing {os.path.basename(filepath)}: {str(e)}")
        return None, None, None
        
    return signals, signal_names, var_types

def identify_signal(signal_names, var_types, is_trusted):
    """Advanced signal identification with multiple matching strategies"""
    # Priority 1: Look for exact naming patterns
    target_patterns = {
        True: ['count_trusted', 'trusted_count', 'count[3:0]'],
        False: ['count_trojan', 'trojan_count', 'count_troj']
    }
    
    for symbol, name in signal_names.items():
        name_lower = name.lower()
        for pattern in target_patterns[is_trusted]:
            if pattern in name_lower and var_types.get(symbol) in ['wire', 'reg']:
                return symbol
                
    # Priority 2: Look for partial matches
    partial_patterns = {
        True: ['trusted', 'normal', 'clean'],
        False: ['trojan', 'malic', 'inject']
    }
    
    for symbol, name in signal_names.items():
        name_lower = name.lower()
        for pattern in partial_patterns[is_trusted]:
            if pattern in name_lower and var_types.get(symbol) in ['wire', 'reg']:
                return symbol
                
    # Priority 3: Any 4-bit counter signal
    for symbol, name in signal_names.items():
        if ('count' in name.lower() or 'cnt' in name.lower()) and var_types.get(symbol) in ['wire', 'reg']:
            if '[3:0]' in name or '4' in name:  # Check for 4-bit signals
                return symbol
                
    return None

def calculate_features(times, values):
    """Comprehensive feature calculation with validation and enhanced metrics"""
    try:
        samples = len(times)
        if samples < 2:
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
        
        # Enhanced glitch detection (300ps threshold)
        glitches = np.sum(time_diffs <= 300)
        
        # Advanced bit flip analysis
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
        probs = [count / samples for count in value_counts.values()]
        entropy = -sum(p * log2(p) for p in probs if p > 0)
        
        # Additional statistical features
        median_jump = np.median(abs_diffs) if len(abs_diffs) > 0 else 0
        max_time_diff = np.max(time_diffs) if len(time_diffs) > 0 else 0
        min_time_diff = np.min(time_diffs) if len(time_diffs) > 0 else 0
        
        return {
            'samples': samples,
            'min_time': min_t,
            'max_time': max_t,
            'duration_ps': duration,
            'toggles': np.count_nonzero(diffs),
            'toggle_rate': np.count_nonzero(diffs) / (duration / 1e6) if duration > 0 else 0,
            'avg_value': np.mean(ints),
            'median_value': np.median(ints),
            'value_range': np.max(ints) - np.min(ints),
            'glitches': glitches,
            'mean_time_diff': np.mean(time_diffs) if len(time_diffs) > 0 else 0,
            'min_time_diff': min_time_diff,
            'max_time_diff': max_time_diff,
            'avg_jump': np.mean(abs_diffs) if len(abs_diffs) > 0 else 0,
            'median_jump': median_jump,
            'jump_std': np.std(abs_diffs) if len(abs_diffs) > 1 else 0,
            'jump_max': np.max(abs_diffs) if len(abs_diffs) > 0 else 0,
            'bitflip_rate': bitflip_rate,
            'entropy': entropy,
            'unique_values': len(value_counts)
        }
    except Exception as e:
        print(f"⛔ Error calculating features: {str(e)}")
        return None

def calculate_thresholds(df):
    """Dynamic threshold calculation using multiple robust methods"""
    def get_threshold(feature, q_trusted=0.95, q_trojan=0.05):
        trusted = df[df['label'] == 0][feature].dropna()
        trojan = df[df['label'] == 1][feature].dropna()
        
        if len(trusted) == 0 or len(trojan) == 0:
            return None
            
        return max(
            trusted.quantile(q_trusted),
            (trusted.quantile(q_trusted) + trojan.quantile(q_trojan)) / 2
        )
    
    thresholds = {}
    features = [
        'glitches', 'avg_jump', 'jump_std', 'toggle_rate',
        'bitflip_rate', 'entropy', 'median_jump', 'max_time_diff'
    ]
    
    for feat in features:
        threshold = get_threshold(feat)
        if threshold is not None:
            thresholds[feat] = threshold
            
    # Special case for glitches - any occurrence is suspicious
    if 'glitches' in thresholds:
        thresholds['glitches'] = 0.5
        
    return thresholds

def auto_label(df, thresholds):
    """Comprehensive auto-labeling using multiple feature thresholds"""
    df['auto_label'] = 0  # Default to trusted
    
    # Primary detection conditions
    primary_conditions = (
        (df['glitches'] > thresholds.get('glitches', 0)) |
        (df['avg_jump'] > thresholds.get('avg_jump', np.inf)) |
        (df['jump_std'] > thresholds.get('jump_std', np.inf))
    )
    
    # Secondary detection conditions
    secondary_conditions = (
        (df['toggle_rate'] > thresholds.get('toggle_rate', np.inf)) &
        (df['bitflip_rate'] > thresholds.get('bitflip_rate', np.inf))
    )
    
    # Entropy-based detection
    entropy_condition = (
        df['entropy'] > thresholds.get('entropy', np.inf)
    )
    
    # Combine conditions
    trojan_conditions = (
        primary_conditions |
        secondary_conditions |
        entropy_condition
    )
    
    df.loc[trojan_conditions, 'auto_label'] = 1
    return df

def generate_detailed_report(df, thresholds):
    """Generate comprehensive analysis report"""
    report = {
        'thresholds': thresholds,
        'label_distribution': {
            'original': {
                'trusted': sum(df['label'] == 0),
                'trojan': sum(df['label'] == 1)
            },
            'auto': {
                'trusted': sum(df['auto_label'] == 0),
                'trojan': sum(df['auto_label'] == 1)
            }
        },
        'detection_metrics': {
            'true_positives': sum((df['label'] == 1) & (df['auto_label'] == 1)),
            'false_positives': sum((df['label'] == 0) & (df['auto_label'] == 1)),
            'true_negatives': sum((df['label'] == 0) & (df['auto_label'] == 0)),
            'false_negatives': sum((df['label'] == 1) & (df['auto_label'] == 0))
        },
        'feature_analysis': {}
    }
    
    # Feature importance analysis
    for feature in thresholds.keys():
        trusted_mean = df[df['label'] == 0][feature].mean()
        trojan_mean = df[df['label'] == 1][feature].mean()
        report['feature_analysis'][feature] = {
            'trusted_mean': trusted_mean,
            'trojan_mean': trojan_mean,
            'separation': trojan_mean - trusted_mean,
            'threshold': thresholds[feature]
        }
    
    return report

def process_vcd_files(vcd_dir):
    """Main processing pipeline with comprehensive data collection"""
    all_data = []
    processed_files = 0
    
    print(f"\n{'='*50}")
    print(f"📂 Processing VCD files in: {os.path.abspath(vcd_dir)}")
    print(f"{'='*50}\n")
    
    for filepath in glob.glob(os.path.join(vcd_dir, '*.vcd')):
        filename = os.path.basename(filepath)
        print(f"🔍 Analyzing: {filename}")
        
        signals, signal_names, var_types = parse_vcd(filepath)
        if not signals:
            continue
            
        is_trusted = 'trusted' in filename.lower()
        label = 0 if is_trusted else 1
        
        signal_id = identify_signal(signal_names, var_types, is_trusted)
        if signal_id is None or signal_id not in signals:
            print(f"⚠️ Warning: Could not identify target signal in {filename}")
            continue
            
        try:
            times, values = zip(*signals[signal_id])
        except Exception as e:
            print(f"⚠️ Warning: Signal extraction failed in {filename}: {str(e)}")
            continue
            
        features = calculate_features(times, values)
        if features:
            features.update({
                'file': filename,
                'signal': signal_names.get(signal_id, 'unknown'),
                'signal_type': var_types.get(signal_id, 'unknown'),
                'label': label
            })
            all_data.append(features)
            processed_files += 1
            print(f"✅ Processed: {filename} ({len(times)} samples)")
        else:
            print(f"⚠️ Warning: Feature extraction failed in {filename}")
    
    print(f"\n{'='*50}")
    print(f"📊 Processed {processed_files} VCD files successfully")
    print(f"{'='*50}\n")
    
    if not all_data:
        return None, None
        
    df = pd.DataFrame(all_data)
    
    # Calculate dynamic thresholds
    thresholds = calculate_thresholds(df)
    
    # Apply auto-labeling
    df = auto_label(df, thresholds)
    
    # Generate detailed report
    report = generate_detailed_report(df, thresholds)
    
    return df, thresholds, report

def save_results(df, thresholds, report, output_dir='output'):
    """Save all results to files"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save main dataset
    dataset_path = os.path.join(output_dir, 'trojan_dataset11.csv')
    df.to_csv(dataset_path, index=False)
    df.to_csv('trojan_dataset11.csv', index=False)
    # Save thresholds
    thresholds_path = os.path.join(output_dir, 'thresholds.json')
    pd.Series(thresholds).to_json(thresholds_path)
    
    # Save report
    report_path = os.path.join(output_dir, 'analysis_report.json')
    pd.Series(report).to_json(report_path)
    
    return dataset_path, thresholds_path, report_path

def main():
    # Configuration
    vcd_dir = 'vcd_dumps'
    output_dir = 'analysis_results'
    
    # Process files
    result = process_vcd_files(vcd_dir)
    
    if result is None:
        print("⛔ No valid data processed. Exiting.")
        return
        
    df, thresholds, report = result
    
    # Save results
    dataset_path, thresholds_path, report_path = save_results(
        df, thresholds, report, output_dir
    )
    
    # Print summary
    print(f"{'='*50}")
    print("📊 RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"\n💾 Dataset saved to: {os.path.abspath(dataset_path)}")
    print(f"⚖️ Thresholds saved to: {os.path.abspath(thresholds_path)}")
    print(f"📝 Report saved to: {os.path.abspath(report_path)}")
    
    print("\n🔍 Label Distribution:")
    print(f"Original: Trusted={report['label_distribution']['original']['trusted']}, "
          f"Trojan={report['label_distribution']['original']['trojan']}")
    print(f"Auto:     Trusted={report['label_distribution']['auto']['trusted']}, "
          f"Trojan={report['label_distribution']['auto']['trojan']}")
    
    print("\n📈 Detection Metrics:")
    print(f"True Positives:  {report['detection_metrics']['true_positives']}")
    print(f"False Positives: {report['detection_metrics']['false_positives']}")
    print(f"True Negatives:  {report['detection_metrics']['true_negatives']}")
    print(f"False Negatives: {report['detection_metrics']['false_negatives']}")
    
    print("\n📊 Feature Analysis (most discriminative):")
    feature_importance = sorted(
        [(f, data['separation']) for f, data in report['feature_analysis'].items()],
        key=lambda x: abs(x[1]), reverse=True
    )[:5]
    
    for feature, separation in feature_importance:
        data = report['feature_analysis'][feature]
        print(f"\n{feature}:")
        print(f"  Trusted mean: {data['trusted_mean']:.4f}")
        print(f"  Trojan mean:  {data['trojan_mean']:.4f}")
        print(f"  Separation:   {data['separation']:.4f}")
        print(f"  Threshold:    {data['threshold']:.4f}")

if __name__ == "__main__":
    main()