# %%


from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()
X, y = iris.data[:, :2], iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy_score(y_test, y_pred)


# %%
import pandas as pd
from pandas.util import hash_pandas_object
# hash
# pd.util.hash_pandas_object


A = pd.DataFrame(iris["data"])
a = hash_pandas_object(A)

args_hash = (*tuple(a), "3")
tuple(list(kwargs.items()))


def learn
