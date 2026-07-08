import numpy as np

from sklearn.preprocessing import StandardScaler

# from sklearn.linear_model import RidgeClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import SGDClassifier

# from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)


SOURCE_FILE = 'digits-responses-3.npz'

if __name__ == '__main__':
    data = np.load(SOURCE_FILE)
    x_data, y_data = data['response'][:, 0], data['label']

    x_data = StandardScaler().fit_transform(x_data)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, shuffle=True)

    classifier = SVC()
    classifier = GridSearchCV(
        classifier,
        param_grid={
            'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
            'C': [1, 10],
            'gamma': ('scale', 'auto'),
            'degree': [1, 10],
            'decision_function_shape': ('ovo', 'ovr')
        },
        cv=10
    )
    classifier.fit(x_train, y_train)

    scores = cross_val_score(classifier, x_test, y_test, cv=10)

    print(f'Averaged Score: {np.mean(scores)}')
