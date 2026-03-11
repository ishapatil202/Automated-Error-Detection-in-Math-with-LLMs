import pandas as pd

# Read the CSV as a single column (no header)
df = pd.read_csv('diversed-data.csv', header=None, names=['raw'])

# Extract the error label (after the last comma)
df['error_label'] = df['raw'].str.split(',').str[-1].str.strip()

# Count occurrences of each error type
label_counts = df['error_label'].value_counts()

print(label_counts)