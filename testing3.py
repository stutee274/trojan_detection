import numpy as np
import pandas as pd
import os, glob, warnings
from collections import defaultdict
from math import log2

warnings.filterwarnings('ignore')

def parse_vcd(filepath):
    signals, signal_names, var_types = defaultdict(list), {}, {}
    parsing_values, current_time = False, 0

    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if '$enddefinitions' in line:
                    parsing_values = True
                    continue
                if not parsing_values and line.startswith('$var'):
                    parts = line.split()
                    if len(parts) >= 5:
                        var_types[parts[3]] = parts[1]
                        signal_names[parts[3]] = ' '.join(parts[4:])
                    continue
                if line.startswith('#'):
                    current_time = int(line[1:]) if line[1:].isdigit() else current_time
                elif line.startswith('b'):
                    val, sid = line[1:].split()
                    signals[sid].append((current_time, val))
                elif line and line[0] in '01xz':
                    signals[line[1:]].append((current_time, line[0]))
    except Exception as e:
        print(f"⛔ Error in {os.path.basename(filepath)}: {e}")
        return None, None, None

    return signals, signal_names, var_types

def identify_signal(signal_names, var_types, trusted):
    patterns = (['count_trusted', 'trusted'], ['count_trojan', 'trojan'])[not trusted]
    for sym, name in signal_names.items():
        if any(p in name.lower() for p in patterns) and var_types.get(sym) in ['wire', 'reg']:
            return sym
    return next((s for s, n in signal_names.items() if 'count' in n and var_types[s] in ['wire', 'reg']), None)

def calculate_features(times, values):
    if len(times) < 2: return None
    times, values = np.array(times), np.array(values)
    duration = times[-1] - times[0]
    ints = [int(v, 2) if all(c in '01' for c in v) else 0 for v in values]
    ints = np.array(ints)
    diffs, td = np.diff(ints), np.diff(times)
    glitches = np.sum(td <= 300)
    bitflips = [bin(int(values[i], 2) ^ int(values[i-1], 2)).count('1') for i in range(1, len(values)) if len(values[i]) == len(values[i-1])]
    entropy = -sum((p := v/len(values)) * log2(p) for v in pd.Series(values).value_counts())

    return {
        'samples': len(times), 'min_time': times.min(), 'max_time': times.max(), 'duration_ps': duration,
        'toggles': np.count_nonzero(diffs), 'toggle_rate': np.count_nonzero(diffs) / (duration / 1e6) if duration > 0 else 0,
        'avg_value': ints.mean(), 'median_value': np.median(ints), 'value_range': np.ptp(ints), 'glitches': glitches,
        'mean_time_diff': td.mean(), 'min_time_diff': td.min(), 'max_time_diff': td.max(),
        'avg_jump': np.abs(diffs).mean(), 'median_jump': np.median(np.abs(diffs)),
        'jump_std': np.std(diffs), 'jump_max': np.max(np.abs(diffs)),
        'bitflip_rate': np.mean(bitflips) if bitflips else 0, 'entropy': entropy, 'unique_values': len(set(values))
    }

def process_vcd_dir(vcd_dir):
    rows = []
    for f in glob.glob(os.path.join(vcd_dir, '*.vcd')):
        print(f"📁 {os.path.basename(f)}")
        signals, names, types = parse_vcd(f)
        if not signals: continue
        trusted = 'trusted' in f.lower()
        sid = identify_signal(names, types, trusted)
        if not sid or sid not in signals: continue
        times, values = zip(*signals[sid])
        features = calculate_features(times, values)
        if features:
            features.update({'file': os.path.basename(f), 'signal': names.get(sid, 'unknown'), 'signal_type': types.get(sid, 'unknown'), 'label': int(not trusted)})
            rows.append(features)
    return pd.DataFrame(rows)

def calculate_thresholds(df):
    def t(f): return max(df[df.label==0][f].quantile(0.95), (df[df.label==0][f].quantile(0.95)+df[df.label==1][f].quantile(0.05))/2)
    features = ['glitches', 'avg_jump', 'jump_std', 'toggle_rate', 'bitflip_rate', 'entropy', 'median_jump', 'max_time_diff']
    return {f: (0.5 if f == 'glitches' else t(f)) for f in features if df[f].notna().any()}

def auto_label(df, t):
    df['auto_label'] = 0
    df.loc[
        (df['glitches'] > t.get('glitches', 0)) |
        (df['avg_jump'] > t.get('avg_jump', np.inf)) |
        (df['jump_std'] > t.get('jump_std', np.inf)) |
        ((df['toggle_rate'] > t.get('toggle_rate', np.inf)) & (df['bitflip_rate'] > t.get('bitflip_rate', np.inf))) |
        (df['entropy'] > t.get('entropy', np.inf)),
        'auto_label'
    ] = 1
    return df

def save(df, t, out='output'):
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, 'trojan_dataset111.csv'), index=False)
    df.to_csv('trojan_dataset111.csv', index=False)
    pd.Series(t).to_json(os.path.join(out, 'thresholds.json'))
    print(f"✅ Saved to {out}")

def main():
    df = process_vcd_dir('vcd_dumps')
    if df.empty:
        print("⛔ No data extracted.")
        return
    thresholds = calculate_thresholds(df)
    df = auto_label(df, thresholds)
    save(df, thresholds)
    print(df[['file', 'label', 'auto_label'] + list(thresholds.keys())])

if __name__ == '__main__':
    main()
