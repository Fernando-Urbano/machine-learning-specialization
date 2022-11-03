# Machine Learning Specialization - Deep Learning

# Overview of Machine Learning
## Applications of Machine Learning
Machine learning is a subfield of AI.
AGI (Artificial general intelligence): building machines as intelligent as human beings.

# 1) Supervised Machine Learning: Regression and Classification
## 1.1) What is machine learning?
"Field of study that gives computers the ability to learn without being explicity programmed."
### Supervised Learning (course 1 and 2)
Most used in real world problems. Had the most rapid advancements.
Learns from x to y (input to output. In this case you give the y.
- e-mail > spam (0 or 1)
- audio > text transcption
- etc...

#### Regression and Classification
Are the most common supervised learning algorithmics. Classification can be done in group or just classifying as True or False (0 or 1).
It is different because it tries to predict some discrete result.
One way to define the problem when classifying in two groups based on one variable is with a line.

<img width="794" alt="image" src="https://user-images.githubusercontent.com/99626376/199839693-b510fbcf-d74a-4043-977c-6b976a06b513.png">

Remember: in classification, the class = category.
With more variables, the previous line becomes a graphic.

<img width="469" alt="image" src="https://user-images.githubusercontent.com/99626376/199839793-ad2ab7f5-b6ed-4065-b8c6-ef74812dca87.png">

### Unsupervised Learning (course 3)
The second most used king of machine learning.
There is no label in this case.

<img width="806" alt="image" src="https://user-images.githubusercontent.com/99626376/199839841-4fe0ff1f-1c46-40a5-8ad8-396c17952557.png">

In this case, for the same example as before, the models cannot define what is what, but it can be able to find interesting patterns (structure in the data).

#### Clustering
A clustering algorithmic can divide the dataset in two or more groups based on the feaatures of the dataset.

<img width="644" alt="image" src="https://user-images.githubusercontent.com/99626376/199839919-9a85c366-40a2-40be-848b-379d66a99d2b.png">

The previous clustering algorithmic uses some words to define what is and what is not similar to the first article.
The next example divides people into different groups by the activity of their genes in their DNA.

<img width="791" alt="image" src="https://user-images.githubusercontent.com/99626376/199839968-d0453fd4-d2c9-49fc-8c78-414128b7a78a.png">

#### Anomaly Detection
Find unusual data points (like in the financial system with usual transactions).

#### Dimensionality Reduction
Reduce the number of dimensions in the dataset.

## 1.2) Regression Model

<img width="693" alt="image" src="https://user-images.githubusercontent.com/99626376/199840130-ed9fcf1d-91bf-49a6-aa4d-c4e439d29f33.png">

Training set: used to create the model.
Testing set: used to validade the model.

<img width="410" alt="image" src="https://user-images.githubusercontent.com/99626376/199840162-151f9814-a190-4d02-9566-a5bbe0708720.png">

The regression generates a funcion which generates a "y hat", or estimate/prediction for "y".

The key point here is how to represent "f"?
Can be linear, can be non-linear funcion, etc...

<img width="459" alt="image" src="https://user-images.githubusercontent.com/99626376/199840191-15fed035-a6fe-4031-b1c7-e6260892b8fb.png">

#### Cost Function
The cost function shows how well the model is doing (compares y hat and y). With linear regression, you want to choose a line that defines well the data.

<img width="368" alt="image" src="https://user-images.githubusercontent.com/99626376/199840247-ac7003e2-2711-43ae-9222-09f1c28b3f84.png">

Example of cost functions: MSE, MAE, RMSE, RMAE, etc...

<img width="606" alt="image" src="https://user-images.githubusercontent.com/99626376/199840273-8966f631-d737-4cb5-9c16-8e618ee21490.png">

The squared error cost function (MSE) is the most common and most used kind of cost function.

<img width="495" alt="image" src="https://user-images.githubusercontent.com/99626376/199840303-bebf9aa0-db93-4861-bbc1-afb869f9d425.png">

The cost function using two parameters:

<img width="803" alt="image" src="https://user-images.githubusercontent.com/99626376/199840342-952a16db-8666-4754-bca5-ff427ddd7ecf.png">

In this case, w and b are two parameters for us to choose to minimize the cost function. A simpler way to look at that is to use horizontal slides:

<img width="357" alt="image" src="https://user-images.githubusercontent.com/99626376/199840375-0a82391c-886e-4f56-9731-d38b41e4e8df.png">

#### Gradient Descent
Is a algorithmic to define the best w and b. The gradient descent is used to minimize any function. The gradient descent sometimes require a initial guess.

Not all functions have only one minimum. Sometimes, the function can have a local minimum and global minimum.

<img width="720" alt="image" src="https://user-images.githubusercontent.com/99626376/199840415-ad5d1349-3b55-47a6-8e5e-186e362cd89e.png">

Gradient descent chooses the direction which has the biggest inclination downwards.

<img width="747" alt="image" src="https://user-images.githubusercontent.com/99626376/199840458-6b9aa52f-36ab-4508-b5fa-6cbbfb40d033.png">

The problem with gradient descent is that it can lead you to one minimum. The final point (which local minimun is chosen), depends of the initial position chosen.
The gradient descent is updated everytime using an "alpha".

<img width="362" alt="image" src="https://user-images.githubusercontent.com/99626376/199840502-8edf706f-5c79-4cab-aab9-439a6ce69fdb.png">

The gradient descent is updated everytime using an "alpha".
The "alpha" is the learning rate. The learning rate is a number between 0 and 1. It controls how large are the steps.
Each of the paratemers needs to be substracted by the learning rate.

<img width="565" alt="image" src="https://user-images.githubusercontent.com/99626376/199840540-df9b1cdb-4b8a-4f48-9977-78500e66fb45.png">

Both parameters have to be updated at the same time!

<img width="833" alt="image" src="https://user-images.githubusercontent.com/99626376/199840562-fe55904a-62ef-48cd-b55a-fb54bf1719f2.png">

In incorrect form, the update of b is done after the update of w is done. In this way, b considers the variation wrongly.
Gradient descent might work without simultaneous update, but is not the correct way to do it.

#### Choice of Learning Rate
If the learning rate is chosen porly, than the algorith might not work well.
The learning rate will influence the parameter also considering the derivative (how much change there is a in the cost function).

##### If the learning rate is two small...
The decrease of cost function will take way more time. It will work anyway.

##### If the learning rate is two large...
The local minimun might not be found. It may overshoot, or may fail to converge.

Attention: in cases like that, if we find a local minimun, the algorith will not look for a global minimun because the derivate is 0. Therefore, independently of "alpha" the change in the parameter will always be 0.

Near a local minimun, because generally the derivate is smaller the steps also get smaller.

For linear regression, the following steps can be taken to estimate the values of the parameters:

<img width="537" alt="image" src="https://user-images.githubusercontent.com/99626376/199840625-3c25eb2f-2f9b-4690-b966-640c14037595.png">

##### Convex function
In a convex function there is only global minimun. The gradient descent will always work in this case.

#### "Batch" gradient descent
Each step of the gradient descent uses all the training examples.
Other algorithms use subsets of the complete training sample.

## 1.3) Multiple Linear Regression
When there are multiple features, the model is generally written in vector format:
![image-3.png](attachment:image-3.png)

#### Vectorization
Makes the model faster and the code shorter.
Vectorize code will allow us to take advantage of modern numerical linear algebra. The GPU is used when we vectorize code.

Numpy: numerical linear algebra for python. In this case, the we must index beginning at 0.
The model without vetorization would be more inneficient.

Model with vetorization:

<img width="462" alt="image" src="https://user-images.githubusercontent.com/99626376/199840952-e386dc1e-5de0-45b0-82ca-244bd97944b5.png">

<img width="474" alt="image" src="https://user-images.githubusercontent.com/99626376/199840980-4533aad0-6e5b-4012-8e33-20a2986d3d18.png">

The function makes the "dot" product between the two vectors.
Especially when "n" is large, it will run much faster.
Numpy function is capable of using parallel hardware in my computer.

Model without vetorization:

<img width="474" alt="image" src="https://user-images.githubusercontent.com/99626376/199840980-4533aad0-6e5b-4012-8e33-20a2986d3d18.png">

A model running with vectorization runs all the elements at the same time.

<img width="954" alt="image" src="https://user-images.githubusercontent.com/99626376/199841004-5010e15e-8614-403b-8323-4e5f58a0c1d8.png">

In the case of multiple linear the derivative * alpha will happen at the same time for all the parameters.
The code can go from hours to one or two minutes to complete...

## 1.4) Gradient Descent in Practice

#### Featuring Scale
Helps the algorith to run much faster. The feature scale gets the range to be the same for all features. Without scale, the parameter optimatization leads to parameters that are very different.
When the range is too different, the features will be really different and the cost function will be formed by ovals because a small variation in one of the parameters will generate a big difference in the cost function while for another parameter the same variation will generate a small difference in the cost function.

<img width="764" alt="image" src="https://user-images.githubusercontent.com/99626376/199841176-dc7ca160-7665-4ab2-8e68-8c5482039d1e.png">

In this case, the gradient descent will probably take a longer time to find the best parameters and will end up going back and forward more than before.

<img width="424" alt="image" src="https://user-images.githubusercontent.com/99626376/199841214-27e8c5a3-7f4c-4746-87e2-a7414da9bfa7.png">

Reescaling makes all the values range from 0 to 1. In this way, the gradient descent can find a more direct path to the global minimum.

<img width="345" alt="image" src="https://user-images.githubusercontent.com/99626376/199841248-5390b5bc-d7cc-4b13-b930-47c7dd6910d2.png">

How to scale?

#### Simple scale
Divide every value by the maximum value.

<img width="450" alt="image" src="https://user-images.githubusercontent.com/99626376/199841274-54b779bc-b9e0-428c-8888-33c0f2458b50.png">

#### Mean normalization
Find the mean of the feature.
Subtract the mean from the value of the features and divide the values of the feature by the range.

<img width="461" alt="image" src="https://user-images.githubusercontent.com/99626376/199841324-0d8c221e-91b5-48eb-baa4-dbeb1c566374.png">

#### Z-Score normalization
Find the mean of the feature.
Subtract the mean from the value of the features and divide by the standard deviation of the feature.

<img width="395" alt="image" src="https://user-images.githubusercontent.com/99626376/199841341-4d0a01bb-c3fe-4f58-a8fd-89cd0807af50.png">

#### What should be used?
Aim to get a scale between -1 and +1. But a scale that goes from -0.3 to +0.3 or -3 to +3 is also ok!
...It can also be between -0.5 and +2, 0 and 3, etc...
The important point is to get values that do not differ much in range from 10e-1 to 10e1.

<img width="594" alt="image" src="https://user-images.githubusercontent.com/99626376/199841374-07598e0d-2386-47bc-b358-424e5a4ff487.png">

#### Checking Gradient Descent for Convergence
1) Plot the graphic of J against the number of iterations.
2) Check that the J is converging. This is called a learning curve:

<img width="431" alt="image" src="https://user-images.githubusercontent.com/99626376/199841398-99fbeeff-f37b-4aa7-b841-05a2bd7ad988.png">

If J increases, that means that alpha was chosen poorly. Also, the number of iterations necessary for conversion is unknown. A good way to know if the convergence is at the end or at the beginning is to plot the convergence curve along the process.
#### Automatic convergence test

<img width="297" alt="image" src="https://user-images.githubusercontent.com/99626376/199841452-0a71cafc-adb3-4ff9-a5dd-3a3aca1abf2b.png">

Convergence: found the parameters w and b to get close to global minimum.

#### Choosing the Learning Rate
Means that the alpha is to big:

<img width="329" alt="image" src="https://user-images.githubusercontent.com/99626376/199841503-5b87e071-b529-46d9-8bb4-8cece4d3cfdd.png">

This happen because:

<img width="331" alt="image" src="https://user-images.githubusercontent.com/99626376/199841531-fa0419a1-990a-49bc-a519-c213c23f4b02.png">

A smaller learning rate will fix the problem.
But remember to check for bugs as well!
Good practice: start with a small alpha and increase until before having problems with the cost function.

#### Feature Engineering
How to choose the best feature for the model?
Feature engineering: using intuition to design new features, by transforming or combinin original features.

#### Polinomial Regression
Polinomial regression will just give possibilities to work with squares or cubes of the data.

<img width="817" alt="image" src="https://user-images.githubusercontent.com/99626376/199841579-49862481-4559-4514-aee4-6f7d3e0cdb0c.png">

## 1.5) Classification with Logistic Regression
The idea is that we have a discrete amount of possibilities.

#### Binary Classification
Only has two possibilities (P(y=0) + P(y=1) = 1)
In this case, category = class. We often designate the answers in this case as No or Yes, False or True, or 0 and 1. Generally, we refer as "negative class" vs. "positive class".

We can visualize as:

<img width="907" alt="image" src="https://user-images.githubusercontent.com/99626376/199843787-b18a523d-f014-4137-8cfa-a734c12a54ac.png">

Linear regression will lead to numbers below 0 or above 1:

<img width="733" alt="image" src="https://user-images.githubusercontent.com/99626376/199843807-d15ddc0c-9eea-470c-8c7e-d3ee443503ed.png">

In this case, it is interesting:
- consider a threshold to define if the model predicts as 0 or 1.
- consider that outliers will shift the decision making when regression is used:

<img width="950" alt="image" src="https://user-images.githubusercontent.com/99626376/199843853-62f9d425-3981-4d09-aad2-74b34d0d9673.png">

- The outliers and problem below 0 and above 1 can be solved by using a logistic regression.

#### Logistic Regression
The logistic regression fits a curve that looks like this:

<img width="555" alt="image" src="https://user-images.githubusercontent.com/99626376/199843908-4a92913e-5adc-467e-9315-3452db9f1427.png">

The output is never not 0 or 1, but the answer of the model might be. The logistic regression is based on the sigmoid function.

#### Sigmoid Function
The sigmoid funciton uses z. When z is very large, the nunber tends to function g(z) goes towards 1.

<img width="441" alt="image" src="https://user-images.githubusercontent.com/99626376/199843957-32bbf3dd-8a0d-4fc4-bbe2-b2a351e4f44c.png">

When z is too small, g(z) goes toward 0.
Then we try to predict z as the result variable of the regression in the same way as before. z will be always 0 or 1.

<img width="282" alt="image" src="https://user-images.githubusercontent.com/99626376/199843981-ce47f8af-2ea5-4afb-a5bd-4885b810ca96.png">

Considering that, the logistic regression model considers:

<img width="496" alt="image" src="https://user-images.githubusercontent.com/99626376/199844018-395cddb9-b090-467a-90ff-57d5f807e484.png">

The logistic regression will give a "probability" of being True.

#### Decision Boundary
Not necessary the decision boundary has to be 0.5. It is generally so, but not always true.
It is interesting to think abou the sigmoid function...
When is g(z) >= 0.5?
Whenever z is greater 0!
Therefore, it is greater than 0 and g(z) greater than 0.5 when wx + b is bigger than 0.

<img width="525" alt="image" src="https://user-images.githubusercontent.com/99626376/199844078-014a5150-ead7-4d27-b803-173947a3be94.png">

The decision boundary:

<img width="435" alt="image" src="https://user-images.githubusercontent.com/99626376/199844122-74212f75-6906-4017-bf89-98e9844de005.png">

The decision boundary can be visualized when we look at some variables.

<img width="849" alt="image" src="https://user-images.githubusercontent.com/99626376/199844151-d575abb3-5782-43fc-8f41-511927733468.png">

The decision boundary is not always a straight line:

<img width="266" alt="image" src="https://user-images.githubusercontent.com/99626376/199844181-6c00814d-8329-4636-bf33-c67d56b5d750.png">

In this previous case, a good polinomial regression would be the following:

<img width="425" alt="image" src="https://user-images.githubusercontent.com/99626376/199844197-77786a93-0fd1-4e31-8565-54902d44e819.png">

Even more complex polinomials can be used to define the decision boundary.

#### Cost Function for Logistic Regression
The minimization of the quadratic of the the errors might not be the best cost function for the logistic regression.
For linear regression:

<img width="551" alt="image" src="https://user-images.githubusercontent.com/99626376/199844245-aeaeaf4d-12e2-4b91-a29c-04e6f8a44baf.png">

For logistic regression, the same cost function would lead to costs like this:

<img width="375" alt="image" src="https://user-images.githubusercontent.com/99626376/199844267-47819cfa-e793-4e46-af3c-0969c25050cc.png">

...This would lead to a non convex function.
Therefore, we define a new cost function.
The L is the image below is the loss given 1 sample:

<img width="683" alt="image" src="https://user-images.githubusercontent.com/99626376/199844302-d181ae94-325f-44d5-a463-bd39bc758360.png">

The loss function gives you how well you are doing on one training example, while the cost function is a measure of all training examples (the mean of the loss functions).
With this new function we get the loss function to be:

<img width="370" alt="image" src="https://user-images.githubusercontent.com/99626376/199844336-5d2ebcae-cfa7-4f88-a5a0-adb17b239430.png">

This loss function is really big when the algorithm predicts a number close to zero when the right answer is one or when it predicts a number close to one when the right answer is zero.

<img width="641" alt="image" src="https://user-images.githubusercontent.com/99626376/199844363-3681995e-b540-4543-8b47-7b4b4c5f6a1e.png">

With this new choice, the cost will be convex.
The loss function above can be rewritten to be easier to implement.
    $$loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = (-y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)$$
  
This is a rather formidable-looking equation. It is less daunting when you consider $y^{(i)}$ can have only two values, 0 and 1. One can then consider the equation in two pieces:  

when

$$ y^{(i)} = 0$$

the left-hand term is eliminated:

$$
\begin{align}
loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), 0) &= (-(0) \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - 0\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) \\
&= -\log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)
\end{align}
$$

and when

$$ y^{(i)} = 1$$

the right-hand term is eliminated:

$$
\begin{align}
  loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), 1) &=  (-(1) \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - 1\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)\\
  &=  -\log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)
\end{align}
$$

OK, with this new logistic loss function, a cost function can be produced that incorporates the loss from all the examples.

#### Simplified Cost Function for Logistic Regression
The cost function can be written in the following format:

<img width="894" alt="image" src="https://user-images.githubusercontent.com/99626376/199844682-ffa2dcd7-ff17-4023-9aa8-b18a2f5902dc.png">

With that, the simplification happens because the vale can only have value 0 or 1.
At the end:

<img width="864" alt="image" src="https://user-images.githubusercontent.com/99626376/199844706-9c53c766-c647-456d-945e-d059688d3756.png">

Why we use that cost function? Because it is close to the maximun likelyhood.

#### Gradient Descent for Logistic Regression
We use the gradient descent again to minimize the cost function in matter in the same way:

<img width="849" alt="image" src="https://user-images.githubusercontent.com/99626376/199844810-0a6f68c5-61db-416d-b305-14c2edd7de16.png">

<img width="918" alt="image" src="https://user-images.githubusercontent.com/99626376/199844844-4d34a2a6-ffb6-4b73-bc9c-c639cddf6fb2.png">

The parameters will be again derived from the function of cost. The difference is in the function of cost:

<img width="636" alt="image" src="https://user-images.githubusercontent.com/99626376/199844863-4830b4d1-c1d2-4cdb-83b6-0a2bb47b8b93.png">

Vectorization and feature scaling are also useful in this case to reducethe time spent to find the best parameters.

## 1.6) The Problem of Overfitting
Under bias: relates to underfit.
Generalization: capacity of the model to generalize to data it has never seen before.

<img width="904" alt="image" src="https://user-images.githubusercontent.com/99626376/199844953-e18836d5-557e-46c9-ab94-40e856142ea8.png">

High variance: relates to overfit. The high variance will try to find patterns that actually do not exist.

The same works for the for classification:

<img width="902" alt="image" src="https://user-images.githubusercontent.com/99626376/199844982-b68ea4ee-2b2a-42d0-a6f0-6b5c972ac5fc.png">

#### Addressing Overfitting
- Collect more training date
- Select features to include and features to exclude (a lot of features but not enought training data will lead to overfitting). Also called feature selection (will be more addressed in course two).
- Regularization

####  Regularization
Obbligate the parameters to be smaller:

<img width="778" alt="image" src="https://user-images.githubusercontent.com/99626376/199845055-0b44f6ae-db1e-47da-8a30-8ea0f2f2e68f.png">

Prevents the features to have a too big effect.

#### Cost Function with Regularization
With regularization, the model penalizes by the square of the parameter or the absolute value of the parameter.
This will lead to function which will still have the same amount of parameters, but they will have a smaller value.

To make that, we add lambda:

<img width="742" alt="image" src="https://user-images.githubusercontent.com/99626376/199845118-bb2c7fbb-0fb0-4a59-adcc-8ee39d10626f.png">

Scaling lambda by "2m" will help to find better lambdas.

Attention: generally it does not make sense to penalize the "b" parameter (intercept).

<img width="733" alt="image" src="https://user-images.githubusercontent.com/99626376/199845161-dc63c71f-d89d-48a8-b400-7bd94b7c30cd.png">

The first term encourages to make the best fit in the sample.
The second term encourages smaller parameters.

Attention: too big lambdas will underfit and too small lambdas will overfit.

#### Regularized Linear Regression
In the case of regression:

<img width="753" alt="image" src="https://user-images.githubusercontent.com/99626376/199845220-8549bd82-6bdd-4c7a-8292-d211bb285615.png">

The gradient descent will work as before, but it will now have to look for another term:

<img width="506" alt="image" src="https://user-images.githubusercontent.com/99626376/199845245-e6787221-dbb0-4a76-bf7f-4225c5e685c3.png">

Because we do not regularize "b", we do not have to input a term of lambda there.

<img width="669" alt="image" src="https://user-images.githubusercontent.com/99626376/199845272-f7d15bcf-c48f-428e-9f9d-5b370e063579.png">

In this case, we can define "alpha" as the weight of regularization vs. the weight of fitting the data. Bigger "alpha" leads to bigger regularization.

#### Regularized Logistic Regression
In the case of logistic regression, the cost function is different, but the idea is the same:

<img width="1043" alt="image" src="https://user-images.githubusercontent.com/99626376/199845315-1767b506-c3bb-44df-ba1b-04477379cd8c.png">

In the same way, we do not expect to regularize "b".

