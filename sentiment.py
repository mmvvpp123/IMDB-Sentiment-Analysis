from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd

'''
Load our data. Since this data has already 
been formated in a way that allows us to load 
it, I have not taken the liberty to scrub it.
'''
df = pd.read_csv("./IMDB_Dataset.csv")

# Split the data 70 training and 30 testing
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.3)

# Vectorize the reviews
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(train_vectors, y_train)

prediction = model.predict(test_vectors)

# 0.89493
print(model.score(test_vectors, y_test))
# Results can further be improved by data scrubbing which
# would remove stop words, and other unnecessary details.