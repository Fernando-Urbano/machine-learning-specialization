# XGBoost
XGbBoost is an optimized distributed gradient boosting library.
Stands for "Extreme Gradient Boosting", where the term "Gradient Boosting" originates from the paper
```
from xgboost import XGBClassifier
bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
bst.fit(X_train, y_train)
preds = bst.predict(X_test)
```
