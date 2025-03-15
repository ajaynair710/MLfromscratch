import numpy as np

class NaiveBayes:
    """
    Naive Bayes classifier for continuous data using Gaussian likelihood.
    """
    def fit(self, X, Y):
        """
        Trains the Naive Bayes classifier by computing class priors, means, and variances.
        :param X: Training feature dataset (numpy array)
        :param Y: Training labels (numpy array)
        """
        n_samples, n_features = X.shape
        self._classes = np.unique(Y)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for c in self._classes:
            X_c = X[Y == c]
            self._mean[c, :] = X_c.mean(axis=0)
            self._var[c, :] = X_c.var(axis=0)
            self._priors[c] = X_c.shape[0] / float(n_samples)
    
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
        Computes posterior probabilities for each class and returns the most probable class.
        :param x: Data point (numpy array)
        :return: Predicted class label (int)
        """
        posteriors = []
        
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]
    
    def _pdf(self, class_idx, x):
        """
        Computes the probability density function of a Gaussian distribution.
        :param class_idx: Index of the class
        :param x: Data point (numpy array)
        :return: Probability density function value (numpy array)
        """
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    import matplotlib.pyplot as plt

    def accuracy(y_true, y_pred):
        """
        Computes accuracy of predictions.
        :param y_true: True labels (numpy array)
        :param y_pred: Predicted labels (numpy array)
        :return: Accuracy score (float)
        """
        return np.sum(y_true == y_pred) / len(y_true)

    # Generate synthetic dataset
    X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Initialize and train the Naive Bayes classifier
    clf = NaiveBayes()
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = clf.predict(X_test)

    # Print classification accuracy
    print("Naive Bayes classification accuracy:", accuracy(y_test, predictions))
