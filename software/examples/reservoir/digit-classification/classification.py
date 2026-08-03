import numpy as np

import matplotlib

import reservoirpy as rpy

from sklearn.linear_model import LogisticRegression

# from sklearn.svm import SVC

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _simulate_reservoir(images: np.ndarray, nodes: int = 32, **kwargs) -> np.ndarray:
    states = []

    # Initializing a reservoir with standard parameters.
    reservoir = rpy.nodes.Reservoir(
        nodes,
        lr=kwargs.get('lr', 0.5),
        sr=kwargs.get('sr', 0.9),
        **kwargs
    )

    for image in images:
        states.append(
            reservoir.run(
                (image.T > 0.5).astype(np.float32)
            )[-1, :]
        )

        _ = reservoir.reset()

    return np.vstack(states)


SOURCE_FILE = 'digits-responses-1.npz'

if __name__ == '__main__':
    matplotlib.use('TkAgg')

    rpy.set_seed(42)

    data = np.load(SOURCE_FILE)

    # The last column response from the reservoir is stored in the first row.
    x_data_reservoir = data['response'][:, 0]

    # The last column from the actual image.
    x_data_baseline = data['data'][:, 7]

    # The states collected from the simulated digital reservoir.
    x_data_digital = _simulate_reservoir(images=data['data'])

    y_data = data['label']

    reservoir_splits = train_test_split(
        x_data_reservoir, y_data, train_size=0.8, random_state=42, shuffle=True
    )

    baseline_splits = train_test_split(
        x_data_baseline, y_data, train_size=0.8, random_state=42, shuffle=True
    )

    digital_splits = train_test_split(
        x_data_digital, y_data, train_size=0.8, random_state=42, shuffle=True
    )

    for splits, experiment in [
        (baseline_splits, 'baseline'),
        (reservoir_splits, 'reservoir'),
        (digital_splits, 'digital')
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

        print(f'Test Accuracy ({experiment.capitalize()}): {test_score: .4f}')
