import pandas as pd

# Read the original customer review dataset
df = pd.read_csv("datasets/sentiment-analysis.csv")

# Select 40 rows
small_df = df.sample(n=40, random_state=42)

# Save the smaller customer review dataset
small_df.to_csv("datasets/customer_reviews.csv", index=False)

print("Original rows:", len(df))
print("New rows:", len(small_df))
print("Small customer reviews dataset created successfully!")