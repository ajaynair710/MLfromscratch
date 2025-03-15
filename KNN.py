from collections import Counter
import numpy as np


def euclidean_distance(x1, x2):
    """
    Computes the Euclidean distance between two points.
    :param x1: First point (numpy array)
    :param x2: Second point (numpy array)
    :return: Euclidean distance (float)
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    """
    K-Nearest Neighbors (KNN) classifier implementation.
    """
    def __init__(self, k=3):
        """
        Initializes the classifier with the specified number of neighbors (k).
        :param k: Number of nearest neighbors to consider (int)
        """
        self.k = k

    def fit(self, X, y):
        """
        Stores the training data.
        :param X: Training feature dataset (numpy array)
        :param y: Training labels (numpy array)
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predicts labels for the given test dataset.
        :param X: Test feature dataset (numpy array)
        :return: Predicted labels (numpy array)
        """
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        """
        Predicts a single data point's class based on its k nearest neighbors.
        :param x: Data point (numpy array)
        :return: Predicted class label (int)
        """
        # Compute distances between the input point and all training points
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Get the indices of the k nearest training samples
        k_idx = np.argsort(distances)[: self.k]
        # Retrieve the labels of the k nearest training samples
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        # Determine the most common label among neighbors
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]


if __name__ == "__main__":
    from matplotlib.colors import ListedColormap
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

    def accuracy(y_true, y_pred):
        """
        Computes accuracy of predictions.
        :param y_true: True labels (numpy array)
        :param y_pred: Predicted labels (numpy array)
        :return: Accuracy score (float)
        """
        return np.sum(y_true == y_pred) / len(y_true)

    # Load the Iris dataset
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    # Initialize and train the KNN classifier
    k = 3
    clf = KNN(k=k)
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = clf.predict(X_test)

    # Print classification accuracy
    print("KNN classification accuracy:", accuracy(y_test, predictions))
