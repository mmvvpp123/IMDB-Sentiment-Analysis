from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("./IMDB_Dataset.csv")

X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.3)


vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)


model = LogisticRegression()
model.fit(train_vectors, y_train)
prediction = model.predict(test_vectors)

print(model.score(test_vectors, y_test))
