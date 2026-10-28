import pandas as pd

df = pd.read_csv("datasets/products.csv")

small_df = df.sample(n=40, random_state=42)

small_df.to_csv("datasets/inventory.csv", index=False)

print("Original rows:", len(df))
print("New rows:", len(small_df))
print("Small inventory dataset created successfully!")