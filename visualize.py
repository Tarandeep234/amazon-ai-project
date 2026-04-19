import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("outputs/sentiment_results.csv")
product_df = pd.read_csv("outputs/recommendation_results.csv")

# Sentiment distribution
sentiment_counts = df["sentiment_label"].value_counts()

plt.figure()
sentiment_counts.plot(kind="bar")
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/sentiment_chart.png")
plt.close()

# Top products
top10 = product_df.head(10)

plt.figure()
plt.bar(top10["product_id"].astype(str), top10["recommendation_score"])
plt.xticks(rotation=45)
plt.title("Top 10 Recommended Products")
plt.xlabel("Product ID")
plt.ylabel("Recommendation Score")
plt.tight_layout()
plt.savefig("outputs/top_products.png")
plt.close()

print("Charts created successfully!")