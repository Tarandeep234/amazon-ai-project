import pandas as pd

# Load sentiment data
df = pd.read_csv("outputs/sentiment_results.csv")

# Convert sentiment to numeric
sentiment_map = {
    "positive": 1,
    "neutral": 0,
    "negative": -1
}
df["sentiment_numeric"] = df["sentiment_label"].map(sentiment_map)

# Group by product
product_summary = df.groupby("product_id").agg(
    average_rating=("rating", "mean"),
    average_sentiment=("sentiment_numeric", "mean"),
    review_count=("review_text", "count")
).reset_index()

print("\nProduct summary:")
print(product_summary.head())

# Normalize review count (0–1)
max_reviews = product_summary["review_count"].max()
product_summary["review_count_score"] = product_summary["review_count"] / max_reviews

# Final recommendation score (simple, explainable)
product_summary["recommendation_score"] = (
    product_summary["average_rating"] * 0.4 +
    product_summary["average_sentiment"] * 0.4 +
    product_summary["review_count_score"] * 0.2
)

# Sort by best products
product_summary = product_summary.sort_values(
    by="recommendation_score", ascending=False
)

# Save results
product_summary.to_csv("outputs/recommendation_results.csv", index=False)

print("\nTop recommended products:")
print(product_summary.head(10))