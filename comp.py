import pandas as pd

# Load both CSVs (change filenames if needed)
df1 = pd.read_csv("trojan_dataset_23.csv")         # from script 1
df2 = pd.read_csv("trojan_dataset_combined.csv")         # from script 2

# Align filenames to ensure matching rows
df1['file'] = df1['file'].str.strip().str.lower()
df2['file'] = df2['file'].str.strip().str.lower()

# Merge on filename (common key)
merged = pd.merge(df1, df2, on='file', suffixes=('_s1', '_s2'))

# Define important features to compare
important_features = [
    'samples', 'toggles', 'toggle_rate',
    'glitches', 'avg_jump', 'jump_std',
    'bitflip_rate', 'entropy'
]

# Compute absolute differences for important features
for feat in important_features:
    merged[f'diff_{feat}'] = (merged[f'{feat}_s1'] - merged[f'{feat}_s2']).abs()

# Show only rows with large differences
thresholds = {
    'samples': 2,
    'toggles': 2,
    'toggle_rate': 5,
    'glitches': 1,
    'avg_jump': 1.0,
    'jump_std': 1.0,
    'bitflip_rate': 0.5,
    'entropy': 0.5
}

discrepancy_mask = False
for feat, th in thresholds.items():
    discrepancy_mask |= (merged[f'diff_{feat}'] > th)

# Show mismatched rows
discrepant = merged[discrepancy_mask]
print("\n⚠️ Discrepant Feature Values Between Scripts:\n")
print(discrepant[['file'] + [f'diff_{f}' for f in important_features]])

# Optional: Save differences to CSV
discrepant.to_csv("feature_discrepancy_report.csv", index=False)
print("\n📁 Saved discrepancy report to feature_discrepancy_report.csv")
