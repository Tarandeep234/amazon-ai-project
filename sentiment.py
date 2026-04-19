import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load cleaned data
df = pd.read_csv("data/processed/clean_reviews.csv")

# Create analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to analyze sentiment
def get_sentiment(text):
    score = analyzer.polarity_scores(str(text))["compound"]
    
    if score >= 0.05:
        label = "positive"
    elif score <= -0.05:
        label = "negative"
    else:
        label = "neutral"
        
    return score, label

# Apply function
df[["sentiment_score", "sentiment_label"]] = df["review_text"].apply(
    lambda x: pd.Series(get_sentiment(x))
)

# Save results
df.to_csv("outputs/sentiment_results.csv", index=False)

print("Sentiment analysis complete!")
print(df.head())