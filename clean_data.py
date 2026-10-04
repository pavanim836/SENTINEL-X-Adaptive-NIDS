import pandas as pd
import numpy as np

file = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

# Load the dataset
df = pd.read_csv(file, low_memory=False)

print("Original dataset shape:", df.shape)

missing_values = df.isnull().sum().sum()
duplicate_rows = df.duplicated().sum()

print("Missing values:", missing_values)
print("Duplicate rows:", duplicate_rows)

# Clean column names and labels
df.columns = df.columns.str.strip()
df["Label"] = df["Label"].str.strip()

print("Original shape:", df.shape)

# Replace infinite values with missing values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remove rows with missing values
df.dropna(inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Reset row numbers
df.reset_index(drop=True, inplace=True)

print("Cleaned shape:", df.shape)
print("\nTraffic distribution:")
print(df["Label"].value_counts())

# Save cleaned data
df.to_csv("cleaned_ddos.csv", index=False)

print("\nCleaned dataset saved!")