# SHAP (SHapley Additive exPlanations
Game theory approavh to explain the output of any machine learning model.
We can build a solid understanding of how to compute and interpret the Shapley-bassed explanations of machine learning models.

### Linear Regression
In linear regression, coefficients are great for telling us what will happen when we change the value of an input feature, but they do not measure the overall importance of a feature.

```
import pandas as pd
import shap
import sklearn

# a classic housing price dataset
X,y = shap.datasets.california(n_points=1000)

X100 = shap.utils.sample(X, 100) # 100 instances for use as the background distribution

# a simple linear model
model = sklearn.linear_model.LinearRegression()
model.fit(X, y)import pandas as pd
import shap
import sklearn

# a classic housing price dataset
X,y = shap.datasets.california(n_points=1000)

X100 = shap.utils.sample(X, 100) # 100 instances for use as the background distribution

# a simple linear model
model = sklearn.linear_model.LinearRegression()
model.fit(X, y)
```

### Partial dependency
Partial dependency between variables help us understand the distribution of the feature and how is the dependency between the feature and the target.

```
shap.partial_dependence_plot(
    "MedInc", model.predict, X100, ice=False,
    model_expected_value=True, feature_expected_value=True
)
```

![image](https://user-images.githubusercontent.com/99626376/201417296-079ac4f3-24c6-4e9a-9034-8f34a2e9be29.png)

The values on the y-axis show the expected value of the target for each value of the feature.

The center of the dashed lines is the center of the partial dependency.

The idea of game theory inside Shap is based on the idea that a feature can or not join a certain model (we know or do not know the value of that feature).

The SHAP value for a specific feature "i" is just the difference between the expected model output and the partial dependence plot at the feature's value.

```
# compute the SHAP values for the linear model
explainer = shap.Explainer(model.predict, X100)
shap_values = explainer(X)

# make a standard partial dependence plot
sample_ind = 20
shap.partial_dependence_plot(
    "MedInc", model.predict, X100, model_expected_value=True,
    feature_expected_value=True, ice=False,
    shap_values=shap_values[sample_ind:sample_ind+1,:]
)
```

![image](https://user-images.githubusercontent.com/99626376/201417915-f770a2bd-0533-41db-94bb-ad9de6999bb2.png)

The same is true for other models, nevertheless, we can not derive the shap value with partial dependency for other models.

### The additive nature of Shapley values
The Shapley value always sum up to the difference between the game outcome when all players are present and the game outcome when no players are present.
The SHAP values of all the input features will always sum up to the difference between baseline (expected) model output and the current model output for the prediction being explained. Clear example can be viewed in a waterfall:

```
# the waterfall_plot shows how we get from shap_values.base_values to model.predict(X)[sample_ind]
shap.plots.waterfall(shap_values[sample_ind], max_display=14)
```

![image](https://user-images.githubusercontent.com/99626376/201422203-8e61b348-6898-4bf6-b887-ad5ce0410a7f.png)

### Explaining an additive regression model
In regression linear model each feature in the model is handled independently of every other. The fact that a certain feature "a" is at a determinated point steady of another does not change the effect of feature "b" in the target. That is the reason why partial dependency tells well the effect of a certain feature.

In Generalized Additive Models (GAM), the relationship and importance of the feature to the target cannot be found in a "partial dependency".

```
# fit a GAM model to the data
import interpret.glassbox
model_ebm = interpret.glassbox.ExplainableBoostingRegressor(interactions=0)
model_ebm.fit(X, y)

# explain the GAM model with SHAP
explainer_ebm = shap.Explainer(model_ebm.predict, X100)
shap_values_ebm = explainer_ebm(X)

# make a standard partial dependence plot with a single SHAP value overlaid
fig,ax = shap.partial_dependence_plot(
    "MedInc", model_ebm.predict, X100, model_expected_value=True,
    feature_expected_value=True, show=False, ice=False,
    shap_values=shap_values_ebm[sample_ind:sample_ind+1,:]
)
```

![image](https://user-images.githubusercontent.com/99626376/201429645-88fc63d6-110b-49a7-9fce-a699212067e3.png)

```
# the waterfall_plot shows how we get from explainer.expected_value to model.predict(X)[sample_ind]
shap.plots.waterfall(shap_values_ebm[sample_ind])
```

The following graphic shows the contribution for a specific target point.
![image](https://user-images.githubusercontent.com/99626376/201429690-df9eeb86-3311-406e-9e62-943cc5eba987.png)

In GAM models, the impact of the feature can change from negative to positive and the opposite as well.

```
# the waterfall_plot shows how we get from explainer.expected_value to model.predict(X)[sample_ind]
shap.plots.beeswarm(shap_values_ebm)
```

![image](https://user-images.githubusercontent.com/99626376/201429974-2bfdc3fc-39e0-4516-9ac0-511899eb6cd8.png)

The first feature can clear define that the impact is positive for high values and negative for small values, but some other features do not have the same clear definition

### Train a a XGBoost
```
import xgboost 

# train XGBoost model
model = xgboost.XGBClassifier(n_estimators=100, max_depth=2).fit(X_adult, y_adult*1, eval_metric="logloss")
```

In the following, `backgroup_adult` is a sample from the the data:

```
# compute SHAP values
explainer = shap.Explainer(model, background_adult)
shap_values = explainer(X_adult)

# set a display version of the data to use for plotting (has string values)
shap_values.display_data = shap.datasets.adult(display=True)[0].values
```

The bar plot shows the mean absolute value of each feature.

```
shap.plots.bar(shap_values)
```

![image](https://user-images.githubusercontent.com/99626376/201431056-f2ba0f7a-194b-493c-bf3a-5c2cb39d883f.png)

Alternativilly, we can take the max variation of each feature. This will favor features that occasionally have a huge effect on the target (but, on average, have smaller effect). In other words, favors outliers.

```
shap.plots.bar(shap_values.abs.max(0))
```

![image](https://user-images.githubusercontent.com/99626376/201431222-4ee2c7ac-2efe-4188-a46a-9f71af08b1f2.png)

To view a more amplified description:

```
shap.plots.beeswarm(shap_values)
```

![image](https://user-images.githubusercontent.com/99626376/201434429-17ca33e8-aa13-45ea-84d0-e06f79494227.png)

By taking the "abs" of the values, we understand the importance of each feature in each case:

```
shap.plots.beeswarm(shap_values.abs, color="shap_red")
```

![image](https://user-images.githubusercontent.com/99626376/201434590-cb6e89c9-a78c-4c96-9dfa-4ee146dcbb9b.png)

### Explaining quantitative measures of fairness
With SHAP we can decompose measures of fairness and allocate responsibility for any observed disparity among each of the model's input features. Nevertheless, the tutorial emphasizes that the black-box fairness does not agree with a human perception of fairness: fairness relies on context-dependent value and is dangerous to treat quantitative fearness as the solution because it obscure important value judgemnts.

### XGBoost Example
```
import xgboost
import numpy as np
import shap
```
Create the features and target:
```
# simulate some binary data and a linear outcome with an interaction term
# note we make the features in X perfectly independent of each other to make
# it easy to solve for the exact SHAP values
N = 2000
X = np.zeros((N,5))
X[:1000,0] = 1
X[:500,1] = 1
X[1000:1500,1] = 1
X[:250,2] = 1
X[500:750,2] = 1
X[1000:1250,2] = 1
X[1500:1750,2] = 1
X[:,0:3] -= 0.5
y = 2*X[:,0] - 3*X[:,1]

# ensure the variables are independent
np.cov(X.T)
```
Running a single decisiion tree:
```
# train a model with single tree
Xd = xgboost.DMatrix(X, label=y)
model = xgboost.train({
    'eta':1, 'max_depth':3, 'base_score': 0, "lambda": 0
}, Xd, 1)
print("Model error =", np.linalg.norm(y-model.predict(Xd)))
print(model.get_dump(with_stats=True)[0])
```








