import pandas as pd

df = pd.read_csv("datasets/Retail Sales Dataset.csv")

small_df = df.sample(n=40, random_state=42)

small_df.to_csv("datasets/sales.csv", index=False)

print("Original rows:", len(df))
print("New rows:", len(small_df))
print("Small dataset created successfully!")