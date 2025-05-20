# -*- coding: utf-8 -*-
"""
Created on Fri May 16 12:54:05 2025

@author: rishabh
"""
# import mecessary labraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
# convert zip file into read.csv
import zipfile
import os

# Set your local path
zip_path = r"C:\Users\risha\Downloads\archive.zip"  # use raw string (r"...") to avoid escape issues
extract_dir = r"C:\Users\risha\Downloads\social_media_dataset"

# Extract the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# Check the extracted files
extracted_files = os.listdir(extract_dir)
print(extracted_files)
#load my csv file
csv_path = os.path.join(extract_dir, "sentimentdataset.csv")
df = pd.read_csv(csv_path)
df.head()
#Basic EDA(Exploratory Data Analysis)
print(df.info())
print(df['Sentiment'].value_counts())  # e.g., Positive, Negative, Neutral
#Plot Sentiment Distribution
sns.countplot(data=df, x='Sentiment')
plt.title("Sentiment Distribution")
plt.show()
# Optional NLP Enhancement(Sentiment Polarity Score)
df['Polarity'] = df['Text'].apply(lambda text: TextBlob(str(text)).sentiment.polarity)
# Check if 'Text' and 'Sentiment' columns exist
print(df.columns)

# Optional: Rename columns if needed
# df.rename(columns={'your_actual_text_column': 'Text', 'your_actual_sentiment_column': 'Sentiment'}, inplace=True)

# View unique sentiment values
print(df['Sentiment'].unique())

# Try using a sentiment that actually exists
filtered_df = df[df['Sentiment'] == 'Joy']  # Replace 'Joy' with a real category if needed

# Join text only if not empty
if not filtered_df.empty:
    text = " ".join(filtered_df['Text'].dropna().astype(str))

    if text.strip():  # Check if text is not just spaces
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title("Word Cloud for 'Joy'")
        plt.show()
    else:
        print("⚠️ No valid words found in 'Text' for the selected sentiment.")
else:
    print("⚠️ No rows found for the sentiment you specified.")
    


df.rename(columns={'YourActualTimeColumn': 'Timestamp'}, inplace=True)
df['Date'] = pd.to_datetime(df['Timestamp'], errors='coerce')  # coerce = convert invalid to NaT
df = df.dropna(subset=['Date'])  # Drop rows where Timestamp couldn't be parsed
df.groupby(df['Date'].dt.date)['Sentiment'].value_counts().unstack().plot(kind='line', figsize=(10,5))
plt.title("Sentiment Trends Over Time")
plt.xlabel("Date")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

