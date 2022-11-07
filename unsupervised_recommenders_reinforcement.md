# 3) Unsupervised Learning, Recommender and Reinforcement Learning
## 3.1) Unsupervised Learning: Clustering

What is clustering? Finds similar datapoints.
When clustering, we do not have "y".
Some utilities for clustering is to analyze DNA data, make groups of people.

#### K-Means Intuition

IMAGE

The k-means will initially take random guesses on what might be the centers of the clusters.

It will randomly pick two points.

IMAGE

After picking a cluster centroid, it will assign each of the points to a cluster centroid.

After that, the k-means will move the cluster centroid to the average place of the dots that belong to that group.

IMAGE

The k-means will continue to do the same as before again and again...

IMAGE

With that, points will change colors (clusters).
IMAGE

With time, no more points will change. When that happens, that means that the algorithm has converged.

IMAGE

### K-Means Algorithm
1) Randomly initialize K cluster centroids
- Each centroid will have the same amount of dimensions as all the data in consideration.
2) Assign a cluster to each of the dots based on the dot proximity to the clusters centroids. The distance is based on |x - m|.
3) Move the cluster centroids to the middle point of all x's.
4) Repeat steps 2 and 3 until there are no more changes in points.

IMAGE

Sometimes, a cluster may end up with no dots and the mean of 0 values is non defined... In that case, we can:
- change the cluster division to k-1 clusters.
- restart the clustering based on different randomly initialized points.

### K-Means Optimization Objective
IMAGE

Attention: generally, we use the squared difference between every point.

IMAGE

The cost function is all the squared differences.
The cost function is called the "Distortion function".
The function will tend to move the center towards a center penalized more by outliers due to squared differences.
The cost function should never go up.

### Initializing K-Means
How do we take the random guess for the cluster centroids in the first attempt?
The most common way to "randomly" select data in the some amount as the number of clusters. This will lead to a first grouping in clusters with cluster centroids being the samples selected.

IMAGE

Neverthless, the initial randomly selected cluster will start at bad points... With that, the local optimum might not be global optimum.

IMAGE

The global optimum would be:

IMAGE

The best way to solve this problem would be to run the k-means multiple times and find the best cost function.
Pick the set of cluster that gave the lowest cost "j".

IMAGE

Trying 50 or even 100 initialization will give the best results!

### Choosing the Numbers of Cluster
IMAGE

#### Elbow Method
How many clusters should we choose?
"Elbow Method": K-Means runs k-means with a variaty of clusters. After that, we plot the cost function for each number of k-means.

IMAGE

The idea is to find the "elbow": the point where the cost function starts to decrease less rapidly. Neverthless, not every plot will show an elbow.

#### Choose based on metric for how well it performs for the later purpose
Example: when choosing the number of sizes for t-shirts, should we choose 3 sizes or 5 sizes?
![image-18.png](attachment:image-18.png)
The first would lead to a bigger cost function, but we would have a smaller cost in production.

## 3.1) Unsupervised Learning: Anomaly Detection
#### Find Unusual Events
Detect if an aircraft engine is functioning well.

IMAGE

After the algorithm has seen the examples, the anomaly detection will try to understand if the engine looks different or similar to the previous example.
If the new data point is different from the previous ones we can define it as an anomaly.
#### Density Estimation
We build a model to see which values are more likely to happen and what are the values which are unlikely to happen.

IMAGE

After that, we define a point "e" that is the separation between normal or anomaly.

IMAGE

#### Example
- fraud detection (websites, banks, etc...)
If we find an anomaly, we generally make a new step for security.
### Gaussian (Normal) Distribution
Say x is a number. If x is a distributed Gaussian with mean and variance.

IMAGE
IMAGE

In the training set, we will estimate the parameters of mean and sigma square:

IMAGE
IMAGE

What we are doing is finding the maximum likelyhood for mu and sigma.
For pratical detection algorithms we generally have many features.
### Anomaly Detection Algorithm
Training set with n features

IMAGE

We will then build a density estimation.
We will than understand the probability of a point for each variable:

IMAGE

The probabilities do not have to be independent for this algorithm to work (really?)

IMAGE

It will flag as anomaly if one or more of the features is very big or very small.

IMAGE

### Developing and Evaluating an Anomaly Detection System
Real-number evaluation: making decision with a way of evaluating our learning algorithm.
The real-number evaluation is important to define the feature engineering, choosing features, etc...

IMAGE

To make a real-number evaluation we define some data as anomalous data and some data as non anomalous data.

Cross validation sets:
will have some anomalous and not anomalous observations in each validation set.
After the cross validation we have also a testing set:

IMAGE

We can then train the algorithm in the training set and use the test set to understand how well the algorithm is doing on the out-of-sample.

This is a good way to tune the parameter "e": the best parameter can be done by getting the same result as the training.

IMAGE

What is the "e" that minimizes the amount of error comparing to the regression.
Evaluation metrics for algorithms is different for datas that are skewed.

IMAGE

Therefore, a small number of labeled examples can help very much in the process of detecting anomalies. ...But why should we use anomaly detection when we have the possibility of using supervised learning?
Answer next:

#### Anomaly Detection vs. Supervised Learning
The anomaly detection is better when we have a very small number of positive examples (less than 20) and a large number of negative examples.
