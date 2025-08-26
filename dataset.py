import pandas as pd

# Load dataset with semicolon delimiter
df = pd.read_csv("data 4.csv", delimiter=";")

# Check first rows
print(df.head())

# Save it back with commas (optional)
df.to_csv("data_clean.csv", index=False)
