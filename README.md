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
