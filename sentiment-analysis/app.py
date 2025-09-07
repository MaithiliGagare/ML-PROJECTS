import pandas as pd
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.sentiment import SentimentIntensityAnalyzer

# Step 1: Load dataset
df = pd.read_csv(r'C:\Users\juili\Downloads\data_analysis\data_analysis.csv')  # Make sure the file is in the same directory

# Step 2: Clean the data
df.dropna(subset=['tweet'], inplace=True)  # Drop rows with missing tweets

def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    return text

df['cleaned_tweet'] = df['tweet'].apply(clean_text)

# Step 3: Label the sentiment (positive/negative) using VADER
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = sia.polarity_scores(text)['compound']
    return 'positive' if score >= 0.05 else 'negative'

df['sentiment'] = df['cleaned_tweet'].apply(get_sentiment)

# Step 4: Prepare training data
X = df['cleaned_tweet']
y = df['sentiment']

vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Step 5: Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Predict and evaluate
y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))
