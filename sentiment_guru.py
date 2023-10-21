import requests
from bs4 import BeautifulSoup
import re
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import opinion_lexicon
from nltk.tokenize import word_tokenize

# Prompt user for stock ticker
ticker = input("Please enter a stock ticker to analyze: ").upper()

# Build URL for the specific stock ticker
url = f"https://www.gurufocus.com/stock/{ticker}/article"

# Send GET request to the URL
response = requests.get(url)

# Parse the HTML content of the response
soup = BeautifulSoup(response.content, "html.parser")

# Find all the text in the page
all_text = soup.get_text()

# Load the Financial Phrasebank
with open("stock/sentiment/Sentences_50Agree.txt") as f:
    phrasebank = f.read().splitlines()

# Combine the Financial Phrasebank with the opinion lexicon for better sentiment analysis
sentiment_lexicon = opinion_lexicon.words() + phrasebank

# Tokenize the text
tokens = word_tokenize(all_text)

# Perform sentiment analysis on the tokens
sia = SentimentIntensityAnalyzer()
positive_count = 0
negative_count = 0
with open("stock/sentiment/analyze.txt", "w", encoding="utf-8") as f:
    for token in tokens:
        if token.lower() in sentiment_lexicon:
            score = sia.polarity_scores(token)
            if score["compound"] > 0:
                positive_count += 1
            elif score["compound"] < 0:
                negative_count += 1
            f.write(token + "\n")

# Calculate the sentiment ratio
if negative_count == 0:
    sentiment_ratio = positive_count
else:
    sentiment_ratio = round(positive_count/negative_count, 2)

# Print the sentiment ratio
print(f"{ticker} has a sentiment of {sentiment_ratio}")
