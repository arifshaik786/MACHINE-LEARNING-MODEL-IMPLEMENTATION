# 1. Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Load Dataset (Using a public SMS Spam Dataset)
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

# 3. Encode Labels
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

# 4. Vectorize Text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['message'])
y = df['label_num']

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Model Training
model = MultinomialNB()
model.fit(X_train, y_train)

# 7. Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 8. Test New Message
def predict_spam(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)
    return "Spam" if pred[0] == 1 else "Ham"

print(predict_spam("Congratulations! You won a $1000 gift card. Call now!"))
print(predict_spam("Hi, are we still on for the meeting at 3?"))
