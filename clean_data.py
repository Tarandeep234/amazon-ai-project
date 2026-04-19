import pandas as pd
import re

# Load dataset
df = pd.read_csv("data/raw/reviews.csv")

# Show original columns
print("Original columns:")
print(df.columns.tolist())

# Rename important columns
df = df.rename(columns={
    "asin": "product_id",
    "reviewText": "review_text",
    "overall": "rating"
})

# Keep only the useful columns
df = df[["product_id", "review_text", "rating"]]

print("\nSelected columns:")
print(df.head())

# Remove rows where review text or rating is missing
df = df.dropna(subset=["review_text", "rating"])

# Remove duplicate rows
df = df.drop_duplicates()

# Function to clean text
def clean_text(text):
    text = str(text).lower()                    # convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text) # remove punctuation/special characters
    text = re.sub(r"\s+", " ", text).strip()   # remove extra spaces
    return text

# Apply cleaning to review_text
df["review_text"] = df["review_text"].apply(clean_text)

# Remove reviews that are too short
df = df[df["review_text"].str.len() > 5]

# Save cleaned dataset
df.to_csv("data/processed/clean_reviews.csv", index=False)

print("\nCleaned data saved successfully.")
print("Final shape:", df.shape)
print(df.head())