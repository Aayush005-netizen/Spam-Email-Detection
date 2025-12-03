import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
import joblib

# Dummy data
data = {
    'Emails': ['free money now', 'hello friend', 'win lottery', 'meeting tomorrow'],
    'Class': [1, 0, 1, 0] # 1 for spam, 0 for ham
}
df = pd.DataFrame(data)

X = df['Emails']
y = df['Class']

# Train Logistic Regression (model.joblib)
clf_lr = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression())])
clf_lr.fit(X, y)
joblib.dump(clf_lr, 'model.joblib')
print("Created model.joblib")

# Train BernoulliNB (modelNB.joblib)
clf_nb = Pipeline([('tfidf', TfidfVectorizer()), ('clf', BernoulliNB())])
clf_nb.fit(X, y)
joblib.dump(clf_nb, 'modelNB.joblib')
print("Created modelNB.joblib")
