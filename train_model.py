import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pickle

# Sample dataset (small but enough for demo)
data = {
    "text": [
        "Win a free iPhone now",
        "Urgent! click this link",
        "Meeting scheduled tomorrow",
        "Please find the attached report",
        "You won a lottery prize",
        "Project discussion at 3 PM"
    ],
    "label": [1, 1, 0, 0, 1, 0]  # 1 = Spam, 0 = Not Spam
}

df = pd.DataFrame(data)

# Convert text to numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Train model
model = MultinomialNB()
model.fit(X, y)

# Save model & vectorizer
pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained and saved")
