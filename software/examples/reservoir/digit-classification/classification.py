import numpy as np

import matplotlib

from sklearn.linear_model import LogisticRegression

# from sklearn.svm import SVC

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SOURCE_FILE = 'digits-responses-1.npz'

if __name__ == '__main__':
    matplotlib.use('TkAgg')

    data = np.load(SOURCE_FILE)

    # The last column response from the reservoir is stored in the first row.
    x_data_reservoir = data['response'][:, 0]

    # The last column from the actual image.
    x_data_baseline = data['data'][:, 7]

    y_data = data['label']

    reservoir_splits = train_test_split(
        x_data_reservoir, y_data, train_size=0.8, random_state=42, shuffle=True
    )

    baseline_splits = train_test_split(
        x_data_baseline, y_data, train_size=0.8, random_state=42, shuffle=True
    )

    for splits, experiment in [
        (baseline_splits, 'baseline'), (reservoir_splits, 'reservoir')
    ]:
        x_train, x_test, y_train, y_test = splits

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            # ('classifier', SVC(kernel='linear', C=0.5, random_state=42))
            ('classifier', LogisticRegression(
                C=0.5,
                # class_weight='balanced',
                random_state=42
            ))
        ])

        cv_scores = cross_val_score(
            pipeline, x_train, y_train, cv=10, scoring='accuracy'
        )

        print(
            f'10-Fold CV Mean Accuracy ({experiment.capitalize()}): '
            f'{np.mean(cv_scores): .4f} (± {np.std(cv_scores): .4f})'
        )

        pipeline.fit(x_train, y_train)
        test_score = pipeline.score(x_test, y_test)

        print(f'Test Accurac ({experiment.capitalize()})y: {test_score: .4f}')
