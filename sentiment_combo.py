import requests
from bs4 import BeautifulSoup
import re
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import opinion_lexicon
from nltk.tokenize import word_tokenize
import nltk
import numpy as np

nltk.download('opinion_lexicon')
nltk.download('punkt')
nltk.download('vader_lexicon')

def analyze_sentiment(url):
    # Send GET request to the URL
    response = requests.get(url)

    # Parse the HTML content of the response
    soup = BeautifulSoup(response.content, "html.parser")

    # Find all the text in the page
    all_text = soup.get_text()

    # Tokenize the text
    tokens = word_tokenize(all_text)

    # Perform sentiment analysis on the tokens
    sia = SentimentIntensityAnalyzer()
    positive_count = 0
    negative_count = 0

    for token in tokens:
        if token.lower() in sentiment_lexicon:
            score = sia.polarity_scores(token)
            if score["compound"] > 0:
                positive_count += 1
            elif score["compound"] < 0:
                negative_count += 1

    # Calculate the sentiment ratio
    if negative_count == 0:
        sentiment_ratio = positive_count
    else:
        sentiment_ratio = round(positive_count / negative_count, 2)

    return sentiment_ratio

# Prompt user for stock ticker
ticker = input("Please enter a stock ticker to analyze: ").upper()

# Load the Financial Phrasebank
with open("stock/sentiment/Sentences_50Agree.txt") as f:
    phrasebank = f.read().splitlines()

# Combine the Financial Phrasebank with the opinion lexicon for better sentiment analysis
sentiment_lexicon = opinion_lexicon.words() + phrasebank

# Build URLs for the specific stock ticker
urls = {
    "GURU": f"https://www.gurufocus.com/stock/{ticker}/article",
    "MARKETWATCH": f"https://www.marketwatch.com/investing/stock/{ticker}",
    "YAHOO": f"https://finance.yahoo.com/quote/{ticker}"
}

# Analyze sentiment for each website
sentiments = {}
for site, url in urls.items():
    sentiment = analyze_sentiment(url)
    sentiments[site] = sentiment
    print(f"{site}: {sentiment}")

# Calculate the average (mean) and median of sentiment values
sentiment_values = list(sentiments.values())
average_sentiment = np.mean(sentiment_values)
median_sentiment = np.median(sentiment_values)

# Print the average and median sentiment values
print(f"Average sentiment: {average_sentiment:.2f}")
print(f"Median sentiment: {median_sentiment:.2f}")
