# Cross-Validation
Learning the parameters of a prediction function and testing it on the same data is a methodological mistake. To make better machine learning predictions:

<img width="443" alt="image" src="https://user-images.githubusercontent.com/99626376/201743101-b2dd50a7-7ed7-4f6d-878a-fbcd949feb28.png">

The following example uses a SVM to fit a linear support vector machine where X is the matrix of features and y is the target.

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
```
```
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)
```
As we saw in the "Advanced Algorithms" session, the idea of tuning the hyperparameters of the model directling looking at the testing set gives a unfair advantage to the testing set vs other generalizations. That is the reason why the "validation-set" is so important!

Because we might reduce drastically the number of observations we have to train, separate a validation set might not be a good idea. Therefore, using a cross-validation can be better.
Therefore, we should:
- separate the data in training and testing set.
- separate the training data in many sets (k-fold CV: training set is split into "k" smaller sets)
- a model is training using k-1 of the foldsas training data and validated on the remaining part of the data (used as a test set to compute a performance measure such as accuracy). the performance measure reported by the k-fold cross-validation is then the average of the values computed in the loop.

<img width="464" alt="image" src="https://user-images.githubusercontent.com/99626376/201744856-2713a456-3c39-486d-ba91-837edb32a564.png">

### Computing Cross-Validated Metrics
The simplest way to use cross-validation is to call the `cross_val_score` helper function on the estimator and the dataset.

The following example shows how to do it for 5 examples:

```
from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(clf, X, y, cv=5)
scores
```

```
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
````

By default, the score computed at each CV iteration is the `score` method of the estimator. It is possible to change it by specifying the `scoring`parameter.

The scoring function can be find in this: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter.
There are many scoring functions, but generally the most useful one is "f1_score" when dealing with binary targets.

The folds can be done by using "KFold" or "StratifiedKFold".

Other strategies are possible as well:

```
from sklearn.model_selection import ShuffleSplit
n_samples = X.shape[0]
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
cross_val_score(clf, X, y, cv=cv)
```

### cross_validate vs. cross_val_score
The `cross_validade`differs from the cross_val_score because it computes the  fit-times, score-times and optionally training scores and fitted estimators.

```
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
scoring = ['precision_macro', 'recall_macro']
clf = svm.SVC(kernel='linear', C=1, random_state=0)
scores = cross_validate(clf, X, y, scoring=scoring)
sorted(scores.keys())
```


