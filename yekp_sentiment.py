import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# file = 'yelp_review_business_sample_cleaned.csv'
# df = pd.read_csv(file)
# df_sample = df.sample(n=2000, random_state=42)
#
# # apply lexicon-based analyzer
# analyzer = SentimentIntensityAnalyzer()
# df['Sentiment_Compound_Score'] = [analyzer.polarity_scores(row)['compound'] for row in df['text']]
# df['Sentiment_Positive_Score'] = [analyzer.polarity_scores(row)['pos'] for row in df['text']]
# df['Sentiment_Neutral_Score'] = [analyzer.polarity_scores(row)['neu'] for row in df['text']]
# df['Sentiment_Negative_Score'] = [analyzer.polarity_scores(row)['neg'] for row in df['text']]
#
# print(df['Sentiment_Compound_Score'])
# print(df['Sentiment_Positive_Score'])
# print(df['Sentiment_Neutral_Score'])
# print(df['Sentiment_Negative_Score'])
#
# df.to_csv('yelp_review_sentiment.csv', index=False)

# I comment the code above to run the code below
file = 'yelp_review_sentiment.csv'
df = pd.read_csv(file)

def classify_sentiment(score):
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Apply to dataframe
df['Sentiment_Label'] = df['Sentiment_Compound_Score'].apply(classify_sentiment)

# See distribution
print("Sentiment distribution:")
print(df['Sentiment_Label'].value_counts(normalize=True).round(3))
print(df['Sentiment_Label'].value_counts())

df.to_csv('yelp_review_sentiment_labeled.csv', index=False)