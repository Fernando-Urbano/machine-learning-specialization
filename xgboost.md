# XGBoost
XGbBoost is an optimized distributed gradient boosting library.
Stands for "Extreme Gradient Boosting", where the term "Gradient Boosting" originates from the paper *Greedy Function Approximation: A Gradient Boosting Machine*.

```
from xgboost import XGBClassifier
bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
bst.fit(X_train, y_train)
preds = bst.predict(X_test)
```

### Objective Function
Consists in training loss and regularization.

$obj(theta) = L(theta) + omega(theta)$

Regularization control the complexity of the model to avoid overfitting.

<img width="662" alt="image" src="https://user-images.githubusercontent.com/99626376/201695611-b021cf07-05fe-4522-8dfa-e860ee9d0c62.png">

Bias-variance tradeoff: general principle is that we want both a *simple and predictive model*

### Additive Training
The first thing to do in a tree based model is to find good parameters. Learning tree structure is much harder than traditional optimization because it cannot use a simple gradient. The interaction has to be done considering all the trees at once. Because of that, it is necessary to use an additive strategy: "fiz what we have learned, and add one new tree at a time."

### Regularization and Model Complexity
The regularization of the tree is defined by the "model complexity": the model complexity of the tree is represented by $digamma$
