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

$obj(\theta) = L(\theta) + \Omega(\theta)$

Regularization control the complexity of the model to avoid overfitting.

<img width="662" alt="image" src="https://user-images.githubusercontent.com/99626376/201695611-b021cf07-05fe-4522-8dfa-e860ee9d0c62.png">

Bias-variance tradeoff: general principle is that we want both a *simple and predictive model*

### Additive Training
The first thing to do in a tree based model is to find good parameters. Learning tree structure is much harder than traditional optimization because it cannot use a simple gradient. The interaction has to be done considering all the trees at once. Because of that, it is necessary to use an additive strategy: "fiz what we have learned, and add one new tree at a time."

### Regularization and Model Complexity
The regularization of the tree is defined by the "model complexity": the model complexity of the tree is represented by $\omega$:

<img width="213" alt="image" src="https://user-images.githubusercontent.com/99626376/201698407-5cd66af3-ef51-44d2-8c78-976f860f5e6d.png">

- $T$ is the number of leaves.
- $w$ is the vector of scores on leaves.

Most of the times, people focus too much on the inpurity of the model and forget the regularization term.

What will happen in a general way is that if the $\gamma$ term is bigger than the gain, we will end up not adding that branch. That is a pruning technique.

### XGBoost Parameters
There are three types of parameters:
- general parameters: boosting, commonly tree, linear model.
- booster parameters: depents on which booster was chosen.
- learning task parameters: decide on the learning scenario.

**General Parameters**

`booster`: default=`gbtree`
`validate_parameters`: [default=`False`] when true, XGBoost will perform validation of input parameters to check whether a parameter is used or not.
`nthread`: [default to maximum number available]

**Parameters for Tree Booster**
- `eta`: step size shrinkage used to update to prevents overfitting. After each boosting step, we ca directly get the weights of new features, and `eta` shrinks the feature weights to make the boosting process more conservative.
- `gamma`: [default=0, max=infinite] minimum loss reduction required to make a further partition on a leaf node of the three. The larger the gamma, the more conservative the algorithm will be.
- `max_depth`: maximum depth of the tree.
- `min_child_weights`: [default=1] minimum sum of instance weight needed in a child. The larger it is the more conservative the algorithm will be.
- `subsample`: [default=1] subsample ratio of the training instance. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. Used to prevent overfitting.
- `sampling_method`: [default=`uniform`] regards the method used to sample the training instances.
    - `uniform`: each training instance has an equal probability of being selected.
    - `gradient_based`: the selection probability for each training instance is proportional to the regularized absolute value of gradients.
- `lambda`: [default=1, alias: reg_lambda] increase the value will make model more conservative.
- `alpha`: [default=0, alias: reg_alpha] increase the value will make model more conservative. The alpha seems to transform a RIDGE in an Elastic Net or in a LASSO.






