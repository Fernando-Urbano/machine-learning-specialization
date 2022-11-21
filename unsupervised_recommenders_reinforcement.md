# 3) Unsupervised Learning, Recommender and Reinforcement Learning
## 3.1) Unsupervised Learning: Clustering

What is clustering? Finds similar datapoints.
When clustering, we do not have "y".
Some utilities for clustering is to analyze DNA data, make groups of people.

#### K-Means Intuition

<img width="636" alt="image" src="https://user-images.githubusercontent.com/99626376/200371040-bc461872-98f0-4031-b773-6d37cd6ad68d.png">

The k-means will initially take random guesses on what might be the centers of the clusters.

It will randomly pick two points.

<img width="672" alt="image" src="https://user-images.githubusercontent.com/99626376/200372134-638a7604-91ea-42bb-aea6-d93d568346ea.png">

After picking a cluster centroid, it will assign each of the points to a cluster centroid.

After that, the k-means will move the cluster centroid to the average place of the dots that belong to that group.

<img width="688" alt="image" src="https://user-images.githubusercontent.com/99626376/200372164-60b4317d-0c0e-49b5-80c7-d393bf268763.png">

The k-means will continue to do the same as before again and again...

<img width="661" alt="image" src="https://user-images.githubusercontent.com/99626376/200372203-dd2c4213-0ec9-4a65-b9a5-e3132708b5c5.png">

With that, points will change colors (clusters).

<img width="683" alt="image" src="https://user-images.githubusercontent.com/99626376/200372248-a7f90abf-6221-433e-b3f0-a5c6e068f57c.png">

With time, no more points will change. When that happens, that means that the algorithm has converged.

<img width="685" alt="image" src="https://user-images.githubusercontent.com/99626376/200373741-761f0c58-fd91-40d6-85ca-92a692728d84.png">

#### K-Means Algorithm
1) Randomly initialize K cluster centroids
- Each centroid will have the same amount of dimensions as all the data in consideration.
2) Assign a cluster to each of the dots based on the dot proximity to the clusters centroids. The distance is based on |x - m|.
3) Move the cluster centroids to the middle point of all x's.
4) Repeat steps 2 and 3 until there are no more changes in points.

<img width="673" alt="image" src="https://user-images.githubusercontent.com/99626376/200373812-46b323ba-2f2c-4114-8c99-b8b9130db4ec.png">

Sometimes, a cluster may end up with no dots and the mean of 0 values is non defined... In that case, we can:
- change the cluster division to k-1 clusters.
- restart the clustering based on different randomly initialized points.

#### K-Means Optimization Objective

<img width="675" alt="image" src="https://user-images.githubusercontent.com/99626376/200373869-06156841-4ebf-41ec-a2a0-ad3d0abad467.png">

Attention: generally, we use the squared difference between every point.

<img width="662" alt="image" src="https://user-images.githubusercontent.com/99626376/200373917-555846c5-3bcf-494a-972d-61819d8e8de8.png">

The cost function is all the squared differences.
The cost function is called the "Distortion function".
The function will tend to move the center towards a center penalized more by outliers due to squared differences.
The cost function should never go up.

#### Initializing K-Means
How do we take the random guess for the cluster centroids in the first attempt?
The most common way to "randomly" select data in the some amount as the number of clusters. This will lead to a first grouping in clusters with cluster centroids being the samples selected.

<img width="329" alt="image" src="https://user-images.githubusercontent.com/99626376/200373978-97e8cb3f-7069-4d08-b04e-832c7f34d153.png">

Neverthless, the initial randomly selected cluster will start at bad points... With that, the local optimum might not be global optimum.

<img width="495" alt="image" src="https://user-images.githubusercontent.com/99626376/200374031-cc5d6828-8c9b-4366-b35c-344673ec9a44.png">

The global optimum would be:

<img width="246" alt="image" src="https://user-images.githubusercontent.com/99626376/200374052-23d2399a-3e45-462f-a6bd-e7e5742f932f.png">

The best way to solve this problem would be to run the k-means multiple times and find the best cost function.
Pick the set of cluster that gave the lowest cost "j".

<img width="676" alt="image" src="https://user-images.githubusercontent.com/99626376/200374079-6484e603-45aa-4861-bb9e-f507cf65f04c.png">

Trying 50 or even 100 initialization will give the best results!

#### Choosing the Numbers of Cluster

<img width="625" alt="image" src="https://user-images.githubusercontent.com/99626376/200374162-887e1719-69a6-4f22-a7f0-d8c1a33f576d.png">

#### Elbow Method
How many clusters should we choose?
"Elbow Method": K-Means runs k-means with a variaty of clusters. After that, we plot the cost function for each number of k-means.

<img width="376" alt="image" src="https://user-images.githubusercontent.com/99626376/200374206-cebb60bb-f835-4d6a-8132-32d3fd330f59.png">

The idea is to find the "elbow": the point where the cost function starts to decrease less rapidly. Neverthless, not every plot will show an elbow.

#### Choose based on metric for how well it performs for the later purpose
Example: when choosing the number of sizes for t-shirts, should we choose 3 sizes or 5 sizes?

<img width="685" alt="image" src="https://user-images.githubusercontent.com/99626376/200374245-62433d87-7cfb-4425-95a7-dbeacc829154.png">

The first would lead to a bigger cost function, but we would have a smaller cost in production.

## 3.1) Unsupervised Learning: Anomaly Detection
#### Find Unusual Events
Detect if an aircraft engine is functioning well.

<img width="418" alt="image" src="https://user-images.githubusercontent.com/99626376/200374279-bfbdcf27-0857-48ad-8ea6-f5e41081b7a7.png">

After the algorithm has seen the examples, the anomaly detection will try to understand if the engine looks different or similar to the previous example.
If the new data point is different from the previous ones we can define it as an anomaly.

#### Density Estimation
We build a model to see which values are more likely to happen and what are the values which are unlikely to happen.

<img width="357" alt="image" src="https://user-images.githubusercontent.com/99626376/200374458-c0e0f00d-5360-423c-aa97-035aeebbfad9.png">

After that, we define a point "e" that is the separation between normal or anomaly.

<img width="435" alt="image" src="https://user-images.githubusercontent.com/99626376/200374486-ec219277-f720-400b-83a2-0dad941eccb0.png">

#### Example
- fraud detection (websites, banks, etc...)
If we find an anomaly, we generally make a new step for security.

#### Gaussian (Normal) Distribution
Say x is a number. If x is a distributed Gaussian with mean and variance.

<img width="659" alt="image" src="https://user-images.githubusercontent.com/99626376/200374566-a828129d-85aa-4a64-830b-f764451d643a.png">

In the training set, we will estimate the parameters of mean and sigma square:

<img width="624" alt="image" src="https://user-images.githubusercontent.com/99626376/200374602-7ff73111-d82e-424f-92fb-64e687f32344.png">
<img width="646" alt="image" src="https://user-images.githubusercontent.com/99626376/200374634-d4fdc461-19d1-41fe-94a9-c71c6b711615.png">

What we are doing is finding the maximum likelyhood for mu and sigma.
For pratical detection algorithms we generally have many features.

#### Anomaly Detection Algorithm
Training set with n features

<img width="467" alt="image" src="https://user-images.githubusercontent.com/99626376/200374672-e8e1804b-6751-4814-a102-2b7601c40d47.png">

We will then build a density estimation.
We will than understand the probability of a point for each variable:

<img width="653" alt="image" src="https://user-images.githubusercontent.com/99626376/200374742-cc804acd-d933-4fb7-8cbf-24f1cc1e8e7d.png">

The probabilities do not have to be independent for this algorithm to work (really?)

<img width="656" alt="image" src="https://user-images.githubusercontent.com/99626376/200374777-d812da4b-d0bf-4164-9290-f5a0b59c9405.png">

It will flag as anomaly if one or more of the features is very big or very small.

<img width="644" alt="image" src="https://user-images.githubusercontent.com/99626376/200374822-0cc8ccb0-5f23-478e-8039-59c898089fdf.png">

#### Developing and Evaluating an Anomaly Detection System
Real-number evaluation: making decision with a way of evaluating our learning algorithm.
The real-number evaluation is important to define the feature engineering, choosing features, etc...

<img width="658" alt="image" src="https://user-images.githubusercontent.com/99626376/200374874-9ff03eac-87bd-49a9-a1d8-a0dc627fa0fe.png">

To make a real-number evaluation we define some data as anomalous data and some data as non anomalous data.

Cross validation sets:
will have some anomalous and not anomalous observations in each validation set.
After the cross validation we have also a testing set:

<img width="675" alt="image" src="https://user-images.githubusercontent.com/99626376/200374921-3884b9f4-aef6-43e3-874d-75520c775787.png">

We can then train the algorithm in the training set and use the test set to understand how well the algorithm is doing on the out-of-sample.

This is a good way to tune the parameter "e": the best parameter can be done by getting the same result as the training.

<img width="673" alt="image" src="https://user-images.githubusercontent.com/99626376/200374971-49d166a7-b3c3-42e7-bf5e-807dd65f8fe3.png">

What is the "e" that minimizes the amount of error comparing to the regression.
Evaluation metrics for algorithms is different for datas that are skewed.

<img width="686" alt="image" src="https://user-images.githubusercontent.com/99626376/200374999-e20beb34-5cd0-4d67-8028-5a6e16102359.png">

Therefore, a small number of labeled examples can help very much in the process of detecting anomalies. ...But why should we use anomaly detection when we have the possibility of using supervised learning?
Answer next:

#### Anomaly Detection vs. Supervised Learning
The anomaly detection is better when we have a very small number of positive examples (less than 20) and a large number of negative examples.

#### Choosing what features to use
Choosing features is really important is really important for unlabeled data.
Carefully choosing the features is crucial.

For anomaly detection non-gaussian features are problematic. We should transform the features to a gaussian one:

<img width="559" alt="image" src="https://user-images.githubusercontent.com/99626376/203080316-86bd81d4-b039-41f0-97a8-d5b38e1a14d0.png">

The transformation can be log(x + c) beig "c" the constant in order to be able to make the feature gaussian.

The transformation can be log(x + c) beig "c" the constant in order to be able to make the feature gaussian.

The transformation can be log(x + c) beig "c" the constant in order to be able to make the feature gaussian.

Others transformations are viable as well:

<img width="291" alt="image" src="https://user-images.githubusercontent.com/99626376/203080836-866e0969-9107-4386-b423-6417245362ec.png">

<img width="310" alt="image" src="https://user-images.githubusercontent.com/99626376/203081343-26a53ff9-318a-4bac-b7aa-9d3876145249.png">

Remember to apply the same transformation to cross-validation and testing set!

Error analysis for anomaly detection is crucial if the CV set does not make it work well.

We want p(x) to be small for anomalous examples "x".

<img width="496" alt="image" src="https://user-images.githubusercontent.com/99626376/203082319-a404a4a7-f4e1-4e08-8cee-9413ee81e09d.png">

Than, try to find "x2" to find better detections.

Connections between variables are also importants: for example, it can be normal for a computer to have high CPU usage, but only if also has high traffic. Therefore, the CPU divided by traffic can be a good way to detect an anomaly!



















