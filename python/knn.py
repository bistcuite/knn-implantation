import numpy as np
from scipy.spatial.distance import euclidean

class KNearestNeighbors(object):
    def __init__(self,k=3):
        self.k = k
    
    def fit(self,X,y):
        self.X = X
        self.y = y

    def predict(self,X):
        y_pred = [self._single_predict(X_row) for X_row in X]
        return np.array(y_pred)
    
    def _single_predict(self,x):
        distances = [euclidean(x, X_row) for X_row in self.X]

        # get indices of k-nearest neighbors(k-smallest distances)
        k_idx = np.argsort(distances)[:self.k]

        k_labels = [self.y[idx] for idx in k_idx] # labels of k-nearest neighbors
        return np.argmax(np.bincount(k_labels)) # most-common label
    
    def accuracy(self,y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
    
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
X,y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42,test_size=0.2)

scores = []
cv = KFold(n_splits=5, shuffle=True, random_state=1)
for fold, (idx_train, idx_valid) in enumerate(cv.split(X)):
    # split train and validation data
    X_train, y_train = X[idx_train], y[idx_train]
    X_valid, y_valid = X[idx_valid], y[idx_valid]

    clf = KNearestNeighbors(k=4)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_valid)

    score = clf.accuracy(y_valid, predictions)
    scores.append(score)
    
# print average accuracy over all folds
print(f"Scores: \n{scores}\n")
print(f"Mean Accuracy: {np.mean(scores)}")