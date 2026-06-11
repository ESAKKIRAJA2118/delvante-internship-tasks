from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample data
texts = ["I love this product", "This is bad", "Very good", "Worst experience"]
labels = ["positive", "negative", "positive", "negative"]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

# Test
test = ["I love it"]
test_vec = vectorizer.transform(test)

prediction = model.predict(test_vec)
print("Sentiment:", prediction[0])
