from optimalKNN import OptimalKNNClassifier
import numpy as np
import sys

if len(sys.argv) != 2:
    print("Usage: python script.py <filename>")
    exit()

filename = sys.argv[1]
print("Filename:", filename)


data = np.load('data.npy', allow_pickle=True)
test_data = np.load(filename, allow_pickle=True)
encoder_type = 1

X_test = test_data[:, encoder_type]
# X_test = X_test[-300:]
X_test = np.array([x[0] for x in X_test])
y_test = test_data[:, 3]
# y_test = y_test[-300:]

X = data
y = data[:, 3]


knn = OptimalKNNClassifier(encoder=encoder_type, k=12,
                           distance_metric='manhattan',consider_times=0)
X = X[:, encoder_type]
# X = X[:300]
# y = y[:300]
X = np.array([x[0] for x in X])

knn.fit(X, y)
evaluation_results = knn.evaluate(X_test, y_test)

print("Evaluation Results:")
print("Accuracy:", evaluation_results['accuracy'])
print("F1 Score:", evaluation_results['f1_score'])
print("Precision:", evaluation_results['precision'])
print("Recall:", evaluation_results['recall'])
