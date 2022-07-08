# Forest Self-Attention

## Installation

For the package installation, first install all the requirements and then install the **sa_forst** package.
```
$ pip install -r requirements.txt
$ python setup.py install
```

## Usage

Self-Attention Forest model has a scikit-learn like interface.
It is extended with `optimize_weights` method which can be executed with the same training data as used for an underlying forest training, or with a new data set.

Code example for model instantiation:
```
from sa_forest import (
    SAFParams,
    SelfAttentionForest,
    ForestKind,
    TaskType,
)

model = SelfAttentionForest(
    SAFParams(
        kind=ForestKind.EXTRA,
        task=TaskType.REGRESSION,
        eps=0.9,
        tau=1.0,
        gamma=0.9,
        sa_tau=1.0,
        sa_dist='y',
        forest=dict(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=5,
            random_state=12345,
        ),
    )
)
```

After the underlying forest should be trained:
```
model.fit(X_train, y_train)
```

And then weights are optimized:
```
model.optimize_weights(X_train, y_train)
```

In order to estimate weights optimization impact scores for `model.predict_original(X_val)` and `model.predict(X_val)` could be compared.

