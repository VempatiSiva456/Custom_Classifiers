import time
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from collections import Counter

# creata an optimal knn classifier

inference_time = []

class OptimalKNNClassifier:
    
    # parameters are encoder, k, distance metric
    def __init__(self, encoder=1, k=3, distance_metric='euclidean', datasize_vs_time_optimal=[], consider_times=0):
        self.encoder = encoder
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
        self.datasize_vs_time_optimal = datasize_vs_time_optimal
        self.consider_times = consider_times

    # fit method
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # calculate distance based on the distance metric
    def calculate_distance(self, x1, x2):
        x1 = np.array(x1)
        x2 = np.array(x2)
        # print(x1.shape,x2.shape)
        if self.distance_metric == 'euclidean':
            return np.linalg.norm(x2-x1, axis=1)
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x2 - x1), axis=1)
        elif self.distance_metric == 'cosine':
            cosine_similarity = np.dot(
                x1, x2) / (np.linalg.norm(x1, axis=1) * np.linalg.norm(x2))
            cosine_distance = 1 - cosine_similarity
            return cosine_distance

    # method for finding k nearest neighbors (returns indices of k nearest neighbors)
    def find_k_nearest_neighbors(self, x):
        distances = self.calculate_distance(self.X_train, x)
        arr = distances.argsort()
        return arr[: self.k]

    # method for returning the inference (prediction)
    # calculates the most common label among the neighbors and adds it as predicted label for datapoint 'x'
    def predict(self, X):
        predictions = []
        for x in X:
            k_nearest_indices = self.find_k_nearest_neighbors(x)
            k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]
            most_common_label = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common_label)
        return np.array(predictions)

    # method for evaluating scores
    def evaluate(self, X_val, y_val):
        start_time = time.time()
        y_pred = self.predict(X_val)
        end_time = time.time()
        if ((len(inference_time) == 0)):
            inference_time.append(end_time-start_time)
        elif self.consider_times==1:
            self.datasize_vs_time_optimal.append(end_time-start_time)
        # else:
        #     self.datasize_vs_time_optimal.clear()
        #     self.datasize_vs_time_optimal.append(end_time-start_time)
        f1 = f1_score(y_val, y_pred, average='weighted')
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(
            y_val, y_pred, average='weighted', zero_division=1)
        recall = recall_score(
            y_val, y_pred, average='macro', zero_division=1)
        return {'f1_score': f1, 'accuracy': accuracy, 'precision': precision, 'recall': recall}

    # method for splitting the data into train and val subsets and then evaluate
    def split_and_evaluate(self, X, y, test_size=0.2):
        X = X[:, self.encoder]
        X = np.array([x[0] for x in X])
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42)
        self.fit(X_train, y_train)
        return [self.evaluate(X_val, y_val), inference_time, self.datasize_vs_time_optimal]
