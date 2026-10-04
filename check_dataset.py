import pandas as pd
import os

folder = "."

file = os.path.join(
    folder,
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

# Load the dataset
df = pd.read_csv(file, low_memory=False)

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Remove spaces from labels
df["Label"] = df["Label"].str.strip()

# Display dataset information
print("Dataset shape:", df.shape)

print("\nAttack and normal traffic:")
print(df["Label"].value_counts())

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())