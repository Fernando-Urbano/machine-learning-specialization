# 2) Advanced Learning Algorithms
## 2.1) Neural Networks Intuition
Inference: make predictions with neural networks.

#### Neurons and the Brain
Neural network were invented many decades ago.
Are also called artifficial neural network.

#### How the brain works?
The brain is the most powerfull machine in the world. The algorith gained more fame in the 80s and 90s and than fell out of favor again in the late 90s. Then, it resurgence from around 2005 with the idea of Deep Learning.
The speech recognition and than computer vision were some of the first applications of neural networks.

speech -> images -> text (NLP) -> ...

Neural networks today does not have much to do with how the brain works.

<img width="319" alt="image" src="https://user-images.githubusercontent.com/99626376/199850903-6914dd92-9424-4a18-95b4-bd008cff7936.png">

A neuron receives inputs in form of eletrical signals and than gives an information (output) to other neurons.
A neuron does not only receive inputs from one neuron, but from various neurons.

Simplified Mathematical Model of a Neuron:
Inputs a number and makes a computation to give another number:

<img width="353" alt="image" src="https://user-images.githubusercontent.com/99626376/199850937-3fddb45e-b92e-40ac-b8d7-db3c8fdcf2e9.png">

Collective, the neurons can input a number, carry out some computation and give an output.

<img width="274" alt="image" src="https://user-images.githubusercontent.com/99626376/199850958-e9ff1041-8e32-494f-93ef-00abca32ecf7.png">

Today, biological neurons have very little importance in determinating the current machine learnings.
Attention: neural network is only useful when we have a lot of data!

<img width="862" alt="image" src="https://user-images.githubusercontent.com/99626376/199850992-5d79aef7-021d-46ec-ab7f-b5540bf55139.png">

The neural networks have a marginal benefit comparing to traditional AI much bigger when we have bigger datasets.

It is necessary to have faster computer processors (GPU) to make it work well! Otherwise, the model will take too long to train.

#### Demand Prediction: example of neural networks
Classification task to see if the t-shirt will or not be a top seller.

<img width="380" alt="image" src="https://user-images.githubusercontent.com/99626376/199851048-674b0306-240b-4720-96d0-0cad0ecec8a5.png">

The classification task would fit a sigmoid function.
In this case, we will change the "function of w and b" name to "a" of "activation". The computation of the formula is what a single neuron does.

<img width="820" alt="image" src="https://user-images.githubusercontent.com/99626376/199851066-8b575857-f7e7-44dc-a146-cfd49836a072.png">

Another way to think about a neuron is to think that the neuron is a tiny little computer.

#### More complex example
Now we have more features to define if it will or not be a top seller.
Now we have price, shipping cost, marketing, and material. We also believe that the fact that the total "price" is important and we call that variable affordbility.

We train neurons in the following way:
- afforbality: we use price and shipping costs;
- awareness: we use marketing;
- perceived quality: we use price and material.

Than, we wire this three neurons to another neuron to the right.
We consider those three neurons which will give inputs to the other neuron a layer.

<img width="826" alt="image" src="https://user-images.githubusercontent.com/99626376/199851102-b62dbad2-cb96-4a95-87d4-2d9bf1c4d60a.png">

Layers will can have one neuron or more than one neuron.
The last layer is also considered the output layer.

Just remembering, the first layer will give the activations.
We begin with the 4 numbers (4 features that we have) and those 4 numbers will become 3 numbers and the 3 numbers will compute the final number.
The list of four numbers is called an "input layer".

In a large neural network, it would be difficult to select which features would be used for each neuron. Therefore, we use that every neuron will have access to every feature, but will not necessary use all.

we than say that the numbers of features are "x vector" which, after going to middle layer (also called "hidden layer") become "a vector", which finally becomes "a".

<img width="804" alt="image" src="https://user-images.githubusercontent.com/99626376/199851135-1496507e-0200-4f6a-9345-611a4f503be9.png">

At the end, the neural networks are logistic regressions that creat new features. Neural network makes automatically steady of "manually enginerring" new features.
The neural network does not need you to decide which features you want to use. It will try compositions by itself.

#### Multiple Hidden Layers
Also called a "multilayer perceptron".

<img width="411" alt="image" src="https://user-images.githubusercontent.com/99626376/199851177-6f4a53a4-d20d-40f1-a15d-464bf89ee746.png">

<img width="477" alt="image" src="https://user-images.githubusercontent.com/99626376/199851220-ebd8c046-1035-490f-88ac-473ef2f6353d.png">

Neural Network Architecture: When building a neural network, we must specify how many hidden layers do we want and how many neurons we will have in each hidden layer.

#### Recognizing Images: example of neural networks
The following image is an image of 1000 x 1000 pixels.

<img width="557" alt="image" src="https://user-images.githubusercontent.com/99626376/199851259-19579d79-7ed3-4d5e-91ee-4e003ab99781.png">

Each pixel can be translated into a number by its intensity. In this example, the intensity of the pixel goes from 0 to 255, being the darkest and 255 the brightest.
If we would enroll the pixels into a vector, it would lead to a 1 million observations in my x vector.

<img width="862" alt="image" src="https://user-images.githubusercontent.com/99626376/199851273-21eb72e2-73bd-4862-a51f-96758f2556f1.png">

Can you train a model with this x vector?

<img width="889" alt="image" src="https://user-images.githubusercontent.com/99626376/200061355-ea1fffde-cded-4abd-9024-e9e84b0b019e.png">

After training the model for a lot of faces, we can check what are the characteristics (features) that the neuron is looking for.
In the first layer, we often see that the neurons are looking for very short lines.


In the second layer, the neurons try to find bigger parts. The first neuron is trying to detect the presence of an eye, the second tehe superior part of an eye, etc...

<img width="992" alt="image" src="https://user-images.githubusercontent.com/99626376/200061505-3f5ac3d6-d92f-495b-8101-9bf459b452d8.png">

After, in the last layer, the neuron tries to find the shapes of faces which correspond the best.

The cars would be the same idea: smaller features, middle features, bigger features...

<img width="873" alt="image" src="https://user-images.githubusercontent.com/99626376/200061590-f5caaee6-fa17-49ca-845a-214586c1f71d.png">

## 2.2) Neural Network Model
#### Neural Network Layer

In this video, we learn how to construct a layer of neural network.
The first hidden layer will take the inputs. The first parameter will be a logistic regression taking the parameters "vector w" and "b" to identify z and than g(z).
In the example, it uses one feature for each neuron.

<img width="899" alt="image" src="https://user-images.githubusercontent.com/99626376/200061673-56a41038-da30-49c9-9272-eb96de766d28.png">

The vectors of three numbers generated by the three parameters will become the "vector a" which will be useful to get the output layer or the following hidden layer.

It is useful to enumerate the layers:
- input layer: layer 0;
- hidden layer: layer 1 to layer (n - 1)
- output layer: layer n

The [1] would be the output of layer 1:

<img width="470" alt="image" src="https://user-images.githubusercontent.com/99626376/200061727-876cad34-f8a4-4f1a-bfaf-78090ca95f89.png">

This is useful to make the reading easier.

Layer 2 will use the activation vector steady of the inputs to generate a new logistic regression.

<img width="494" alt="image" src="https://user-images.githubusercontent.com/99626376/200061774-225d5c49-e7bc-46f0-9978-fe07c101e6d0.png">

The next step will define if the predicted value is 0 or 1.

<img width="905" alt="image" src="https://user-images.githubusercontent.com/99626376/200061819-ed3e0535-e17c-4baf-b823-eeb8436aad19.png">

The `a[2]` is different from the many `a[1]`

#### More Complex Neural Networks
More complex:

<img width="461" alt="image" src="https://user-images.githubusercontent.com/99626376/200062465-d4d45bd7-9557-48a3-9bc1-a475a73935af.png">

If we say that the neural network has "n layers" it includes the input layer, but not the output layer.
The third layer will have the activation function based on layer 2:

<img width="774" alt="image" src="https://user-images.githubusercontent.com/99626376/200062501-3fc36482-b6e6-4417-80c3-f3779ba78bde.png">

<img width="306" alt="image" src="https://user-images.githubusercontent.com/99626376/200062583-89ee7bee-4566-4d2f-bca6-a3ee5d9de557.png">

<img width="321" alt="image" src="https://user-images.githubusercontent.com/99626376/200062638-709138f3-b87c-4119-8a48-28dc951f6521.png">

The following considers:
- the parameters computed in layer 3.
- the activation vector fro layer 2 ("a" vector)
to generate other "a" scalars. The "a" scalars of layer 3 are put together to compute parameters of layer four (w[4] and b[4]).

When l = 1, we actually have "x vector" (inputs).

#### Forward Propagation: Making Predictions with Neural Networks
A handwritten digit recognition to define if we have written a 0 or a 1.

We make a 64 pixel intensity.

<img width="362" alt="image" src="https://user-images.githubusercontent.com/99626376/200062735-2ac166bf-aadb-4b58-8e6e-b983ab88c6f0.png">

To classify, we make a neural network with 3 layers:
- layer 0: input
- layer 1: 25 units/neurons
- layer 2: 15 units
- output layer

<img width="637" alt="image" src="https://user-images.githubusercontent.com/99626376/200062893-5f043e3b-9a11-42be-9fff-39d87690215e.png">

The first layer will generate a vector of 25 scalars. For that, we need 25 functions. The 25 functions will have a matrix with "m" observations and 64 (n) features.
The second layer will generate a vector of 15 scalars. For that, we need 15 functions. The 15 functions will a matrix with "m" observations and 25 (n, which is the same as a[1]) features.
The third layer (output layer) will generate a scalar. For that, we need 1 function. The 1 function will have a matrix with "m" observations and 15 (n, which is the same as a[2]) features.

This algorithm is called forward propagation.
On the other hand, the backwards propagation is used to learn.

## 2.3) Neural Network TensorFlow Implementation

#### Training Details
1) specify how to compute the output given an input.
2) specify loss and cost function.
3) train on data to minimize J(w, b)

<img width="673" alt="image" src="https://user-images.githubusercontent.com/99626376/200063130-918475d3-281d-4a8c-91ac-a7bcbea4ef71.png">

<img width="626" alt="image" src="https://user-images.githubusercontent.com/99626376/200063173-21545aee-6898-42b2-baf0-41959f123ef7.png">

<img width="670" alt="image" src="https://user-images.githubusercontent.com/99626376/200063474-334033b3-7505-4087-8732-2c594e8b0d02.png">

BinaryCrossentropy is used for classification problems.
MeanSquaredError is used for regression problems.

<img width="611" alt="image" src="https://user-images.githubusercontent.com/99626376/200063499-5340ac6d-a772-41b3-a323-368114c2857c.png">

Gradient descent is the way known to minimize the cost function J. There are other ways, steady of gradient descent, to minimize J.
"epochs" means the amount of iterations done.

#### Alternative to the Sigmoid Function
In some cases the sigmoid function might not be the best fit for the neuron.
We can choose the activation function of each layer and each neuron.

#### ReLU (Rectified Linear Unit)

<img width="574" alt="image" src="https://user-images.githubusercontent.com/99626376/200063556-222ea89a-4b9b-4049-8f84-8de9fdf431ab.png">

It goes in a linear way when it is bigger than 0. This kind of activation function is really useful when the data should not consider negative numbers. Example: awareness of the clothes in the previous example. Awareness should never be negative.

#### Linear Activation Function

<img width="466" alt="image" src="https://user-images.githubusercontent.com/99626376/200063636-ed5770f5-b73f-4111-9ce7-624c8ac3f5c5.png">

It is also considered to be no activation function.

#### Softmax Activation Function
Also important. Will be used later.

#### Choosing Activation Function
#### Output Layer
First point: for the output layer it depends of the type of output we are expecting. For example, for a classification problem, the sigmoid activation function is a better fit.

Linear activation function is better whe dealing with continuous possibilities. For example, the expected return of a certain stock.

If "y" can never be negative, the ReLU it a good choice. For example, when predicting the price of a house.

#### Hidden Layers
For hidden layers, it is very commmon to train models using ReLU independently of the kind of problem. Why that?
- it is a bit faster to compute.
- ReLU goes flat on only one part of the function, while the sigmoid function goes flat on both sides. Because it goes flat, gradient descent will be really slow. 

<img width="593" alt="image" src="https://user-images.githubusercontent.com/99626376/200063751-78d21d0e-a23b-4991-a9a7-073d91dd8e46.png">

Therefore, ReLU is almost always recommended:

<img width="544" alt="image" src="https://user-images.githubusercontent.com/99626376/200063796-299678f0-a601-40b5-b0f8-dd34b7a5eb1f.png">

### Why do we use activation functions?
Why not just use the linear activation function
(g(z) = z)?

The neural network would not be able to fit anything more complex than a linear model.

For one feature and two layers with two neurons:

<img width="617" alt="image" src="https://user-images.githubusercontent.com/99626376/200063849-94ff4866-89c2-474e-9e6c-ec7427af0a71.png">

We could conclude that the model would become a regression again.

The same would also happen if we use linear activation function for every hidden layer and a sigmoid function for the output layer in a classification problem: we wold end up having a logistic regression.

## 2.4) Multiclass Classification
Classification with many possible outputs.
In the previous example, we were trying to decide between 0 or 1.
In a multiclass example:

<img width="594" alt="image" src="https://user-images.githubusercontent.com/99626376/200063942-878943eb-d12b-40f0-8f10-0f3a2a020b5b.png">

We want now to estimate, for example, the change that y is any of the following discrete values:

<img width="617" alt="image" src="https://user-images.githubusercontent.com/99626376/200064035-ec519b5a-e39b-4cd8-9c75-96bee5040077.png">

We can use a decision boundary that can classify more than one group:

<img width="578" alt="image" src="https://user-images.githubusercontent.com/99626376/200064082-ab6558a3-e09c-42b2-af32-ea6019ece11c.png">

#### Softmax
Logistic regression applies when y can be 0 or 1.
We use z = w x + b and than g(z) = sigmoid(z).
In this case, the probabilitiy of being equal to 1 + the probability of being equal to 0 must be 1!

<img width="596" alt="image" src="https://user-images.githubusercontent.com/99626376/200064141-f381fc93-f849-4799-b585-52aeb98020c8.png">

Softmax will compute a "z" for each regression against not being. The actual probability of that particular group will be obtained by dividing the exponential of that z by the sum of the exponential of all z's.

<img width="619" alt="image" src="https://user-images.githubusercontent.com/99626376/200064191-7fbe13a5-0cc6-43be-9d36-80a6d60a6534.png">

The formula for more would lead to more possibilities.

<img width="600" alt="image" src="https://user-images.githubusercontent.com/99626376/200064441-2f065312-fee8-4b8b-8bb5-f9a4c5ecfdc6.png">

If N=2 gives a very similar result that a logistic regression.

#### Cost function for Softmax
The cost function of a logistic regression:

<img width="542" alt="image" src="https://user-images.githubusercontent.com/99626376/200064497-7df3dfbe-0057-494b-a349-456239e23111.png">

For the softmax regression, the result will be derived from the crossentropy loss with for y = j, the -log(aj).
Therefore, the higher the probability given by a of being correct, the smaller the loss of that particular sample.

![image](https://user-images.githubusercontent.com/99626376/200064549-289003d7-77d0-4a7b-ab3b-2c0b6c372a29.png)

Therefore, if y = 2, the model does not consider the probability given by the model for y = 1 or y = 3 or y = 4. It only looks at the probability of y = 2 vs. y != 2.

#### Neural Network with Softmax Regression
For classification with N kind of outputs, we use a softmax regression with the output layer with N neurons. 
Example: if we use 10 different classes, we would give 10 output layers:

<img width="409" alt="image" src="https://user-images.githubusercontent.com/99626376/200064594-919927e3-da8a-4069-a063-470429b69433.png">

In this case, z1 = w1 a + b1, z2 = w2 a + b2, ...
In the end, we have "a" for each class which is derived by dividing that specific "z" by all the "z's".

<img width="599" alt="image" src="https://user-images.githubusercontent.com/99626376/200064627-05470a69-431b-42ce-92d4-eee0b700820c.png">

For a logistic regression, the result "a1" would be a function only of "z1". On the otherhand, the softmax function will derive a1 using z1, z2, z3, ..., zN.
This property is a bit unique to the softmax regression.

#### How to implement with tensorflow?
The last layer will need to be a softmax with the amount of units the same as the "units".

<img width="608" alt="image" src="https://user-images.githubusercontent.com/99626376/200064714-2eae4edd-ae88-4d08-8fb1-784dee98bd26.png">

The tensorflow will than use the "SparseCategoricalCrossentropy". Each digit will be one of those values.

<img width="618" alt="image" src="https://user-images.githubusercontent.com/99626376/200064743-291fa034-d353-4325-b2fc-4354e5443b8b.png">

We can train the model in the same way:

<img width="631" alt="image" src="https://user-images.githubusercontent.com/99626376/200064775-ee246cbb-9085-481f-9a98-73c35842b84f.png">

But it is not recommended that we use this code in this way because another way will work better.

#### Improved Softmax Implementation
There is a even better way than using. Because the computer only has a finit amount of space to store each digit the result from 2 / 10000 is different from the result of 1 + (1 / 10000) - (1 - 1 / 10000)

```
x1 = 2.0 / 1e4
x2 = 1 + (1 / 1e4) - (1 - 1 / 1e4)
print(f'{x2:.18f} = {x1:.18f}?')
```
Answer:
```
0.000199999999999978 = 0.000200000000000000?
```
Some more accurate computations can avoid errors like that for softmax.

For example, in the logistic regression, using 1 / (1 + e^z) steady of "a" will lead to a better result.

<img width="595" alt="image" src="https://user-images.githubusercontent.com/99626376/200351896-797c250c-3853-4240-b9e9-84b7d1d152e3.png">

To compute it better in tensorflow, we can use the last layer with "linear" activation function and than use "from_logits=True" to compute the answer separetally afterwards.

<img width="739" alt="image" src="https://user-images.githubusercontent.com/99626376/200351945-8c688d65-277c-4d0c-ae9a-49bc3eb010c7.png">

"from_logits" g(z) after having computed "z".

For logistic regression the problem is not that big, but for softmax regression the problem gets a bit bigger.

The conventional code:

<img width="1049" alt="image" src="https://user-images.githubusercontent.com/99626376/200352002-34b18f33-a088-426d-a365-b402e03f56b1.png">

The more numerically accurate way:

<img width="676" alt="image" src="https://user-images.githubusercontent.com/99626376/200352063-ac1ca42f-f967-46b5-bb90-867c7f6cba3d.png">

Makes it more accurately!

When predicting: we must transform "z" in "a".

<img width="1045" alt="image" src="https://user-images.githubusercontent.com/99626376/200352264-e7753756-86c2-45eb-a7f4-d13caa1f798f.png">

The same would be necessary for logistic regression:

<img width="873" alt="image" src="https://user-images.githubusercontent.com/99626376/200352398-68665f18-692e-4d67-a2a7-986602042931.png">

#### Multi-label classication
In some problems, we can have more than one label. For example, in a image (input), we can check if we have a person, a car, a bus.

<img width="670" alt="image" src="https://user-images.githubusercontent.com/99626376/200352501-bffd9a00-8361-4025-bf0f-73fa472457de.png">

How to build a neural network to know the answer to a problem like that?

#### Build different neural networks

<img width="690" alt="image" src="https://user-images.githubusercontent.com/99626376/200352589-bf5585dc-1be4-40fb-9bf5-1416fb6b8e51.png">

#### Build a single neural networks to detect all the answers at once

<img width="551" alt="image" src="https://user-images.githubusercontent.com/99626376/200355416-d4ce1953-fdb0-4ce8-a750-43a0a87025c4.png">

At the end, we use three sigmoid activations function for each of the neurons.

<img width="466" alt="image" src="https://user-images.githubusercontent.com/99626376/200355465-50fe2c8c-bce4-46f4-84b5-2a22766ea9a2.png">

#### Multi-class vs. Multi-label
Multi-label can have more than 1 output true. The probability of one are not perfect correlated with another option in this case.

In the multi-class, there can only be one answer.

## 2.5) Additional Neural Network Concepts
#### Advanced Optimization: Adam Algorithm
There are other algoithms today better than gradient descent to make parameter optimatization.

Gradient Descent will work like that:

<img width="635" alt="image" src="https://user-images.githubusercontent.com/99626376/200367281-d4651ec3-5cfb-4793-986a-7a68e002bec9.png">

#### Adam Algorithm
The adam algorithm will automatically adjust the learning rate if the model realizes it is taking too little steps.
The adam algorithm will also decrease alpha (learning rate) if it sees that the learning rate is too big and the algorithm is making steps back and forward for a certain parameter.

<img width="570" alt="image" src="https://user-images.githubusercontent.com/99626376/200355839-d1d4a512-2bf5-45ae-b69e-0364f1543f8d.png">

#### Adam Algorithm Intuition
Adam means **Ada**aptive **M**oment Estimation
The adam algorithm will also consider that the learning rate will be different for each parameter.

<img width="679" alt="image" src="https://user-images.githubusercontent.com/99626376/200355884-1f740dd9-4fc0-4732-a334-6d0f9f3327f8.png">

The details of how adam does that is beyond the scope of this course.
The difference for that is that in compile we must specify the optimizer:

`model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategorialCrossentropy(from_logits=True)
)`

<img width="668" alt="image" src="https://user-images.githubusercontent.com/99626376/200367493-d81d55b1-0d99-4310-841b-c07627066d76.png">

It needs a initial alpha rate. It is worth to try smaller or bigger learning rate as well.
It typically works much faster than gradient descent.

#### Denser Layer Type
There are also different types of layers.
In the typical format of a neural network:

<img width="669" alt="image" src="https://user-images.githubusercontent.com/99626376/200367561-b8a90e51-6af1-4852-9a8c-ef75ff939e64.png">

In the typical format of a neural network, all the neurons use all the features.

#### Convolutional Layer
A convolutional layer will make some neurons only look at some features.

<img width="671" alt="image" src="https://user-images.githubusercontent.com/99626376/200367600-fa3576d1-a3dd-497c-8e06-72bde7163cb9.png">

In the example, the blue neuron will only look at that first quadrant, the pink neuron only at another limited region...

Why do that?
- faster computation;
- need less training data (less prone to overfitting);

If we have lots of neurons will have convolutional specification, than we call it a convolutional neural network.

An example of a convolutional neural network is a cardiac cronogram with the heart bits...
We can use each point in trawime as a x...

<img width="220" alt="image" src="https://user-images.githubusercontent.com/99626376/200367662-7a2fb96c-d4c1-43a7-81c0-ac3a3cda1ed2.png">

Than we make a convolutional layer to only look at the first 20 x's, than another to look at x11 to x30, than another to look at x21 to x40, etc...

<img width="661" alt="image" src="https://user-images.githubusercontent.com/99626376/200367697-c05e3ccb-44f8-4558-962b-78d72996078b.png">

We can also make the second layer convolutional: the second layer can have neurons which look at only the first 3 activation functions, another one to look at the first and 3 activation fuctions, etc...

<img width="682" alt="image" src="https://user-images.githubusercontent.com/99626376/200367738-2d03fa19-3a48-4ba9-a96a-1418ddea94e1.png">

With convolutional layers, we can build better versions of neural networks.

#### Deciding what to try next
What to do when the model does lots of errors?
- get more observations
- get smaller of bigger number of features
- make polinomial features
- choose lambda well when dealing with RIDGE, LASSO, etc

#### Carry out diagnostics
Diagnostic are done to gain insights into what is working and what needs to be improved.
They take time to be implemented, but that time is generally worth it!

#### Evaluating performance of model
Having a systematic way on how to evaluate a model will help to get better results.
By computing the test error we can understand how well the model is generalizing. Computing the error for the training sample is also important: if the testing and training errors are too different from each other, there is probably a problem.
Also, when dealing with classification problems, understand what is the fraction of the training and testing set the algorithm has missclassified.

#### Model selection and training/cross validation/test sets
Remember: training error will not be a good predictor of how the model will predict.
The problem with the idea of using testing error is that we are optimizing the algorithm based on the test set: therefore, we can expect to have a better error in the test set than in other generalizations. Therefore, we should split the data in:
- training: 60% of data.
- crossvalidation (also called dev set): 20% of data.
- testing: 20% of data.

<img width="622" alt="image" src="https://user-images.githubusercontent.com/99626376/200929477-fac7ed5e-50e7-4e03-ac13-893747bac9bf.png">

With that, we can do the model selection. We choose the model by looking at the model with the lowest CV error. Furthermore, to understand how the model would do in generalization, we use the testing error. This also works to feature neural network models, tree based, etc...

#### Diagnosing bias and variance
Bias and variance happen on different occasions:

![image](https://user-images.githubusercontent.com/99626376/200953341-e8f14999-00d8-45c5-b4b6-e7bb25c41180.png)

Overfitting: variance.
Underfitting: bias.

When the error is high in training and high error in CV the algorithm has high bias.
When the error is low in training and high error in CV the algorithm has high variance.

A good model has similar and small amount of error in training and CV.

![image](https://user-images.githubusercontent.com/99626376/200953652-b6d4ded4-7cc6-4271-bdb8-3542eff35463.png)

Higher and higher order polinomial will lead to smaller training error, but the same does not happen for CV:

![image](https://user-images.githubusercontent.com/99626376/200953911-b0c7abb8-bb52-4e2c-af58-a4af44a7a8b8.png)

![image](https://user-images.githubusercontent.com/99626376/200954276-b55543f1-10de-4519-98ae-82af666b14a6.png)

![image](https://user-images.githubusercontent.com/99626376/200954316-541a064e-afcb-45e2-8439-5a33511d6b05.png)

It is really rare to have to have high bias and high variance:
- some features are overfitting while others are underfitting.

#### Regularization and bias/variance
Regularization affects the bias and variance.
When lambda is a very large variable. When that happens, the model is highly motivated to have small parameters. This model has a high bias. The high variance, in the other hand, happens when the lambda is too small.

When we work with regularization, the value of the cost function goes up with bigger lambda for training samples.

![image](https://user-images.githubusercontent.com/99626376/200955419-4354e30f-1773-4a0e-9b99-1bfdd9d12730.png)

The algorithm should try to understand which is minimizes the cost function of the CV.

#### Establishing a baseline level of performance
It is necessary to set a base line. If the training set has similar error to the base line and the CV set has a higher error, the model more probably has a variance problem.

"What is the level of error you can reasonably hope to get to?"
- Human level performance
- Competing algorithms performance
- Guess based on experience

![image](https://user-images.githubusercontent.com/99626376/200957373-02ee9eab-a9d0-49b7-a185-4b24f1095103.png)

High bias or high variance is defined by the distant between baseline - training and training - CV.

#### Learning Curves
Is it a function o the number of examples it has. It also showns the CV function cost. The functions are very different for training and CV. That is because in training, when the number of observations is smaller the parameters have less data to fit.

![image](https://user-images.githubusercontent.com/99626376/200966620-eb84813a-a4dd-47c2-bd5f-aa3c01c7b108.png)

With the increase of observations:

![image](https://user-images.githubusercontent.com/99626376/200966758-58b3725f-2529-45a1-abc8-15a31d41a7b3.png)

Algorithms with high bias will have the shape in the learning curve:

![image](https://user-images.githubusercontent.com/99626376/200966890-5d4c5036-7b60-483a-b9a3-bb90638d86df.png)

The curves will flatten after a while. If the model has high bias, it will not increase performance when we have an increase in training sample.

Algorithms with high variance will generally have a bigger difference between training sample and CV but will not get flatten so fast.

![image](https://user-images.githubusercontent.com/99626376/200967206-cab9a45c-797a-429c-8f42-e5892152a96d.png)

Therefore, when we have high variance, a good idea is to increase the sample size.

#### Deciding what to try next revised
It is always important to check if the algorithm is having high variance or high variance.
When the algorithm has high variance:
- get more training examples.
- try smaller sets of features.
- try increasing lambda
When the algorithm has high bias:
- try adding polynomial features.
- try decreasing lambda.
- try getting additional features.

#### Bias and variance in neural networks
Neural networks will produce better predictions and help us choose in the tradeoff between bias and variance.
Simple recept to get an accurate model:
1) train your model and check if it doing well compared to the base-line.
2) if the model is not doing well, add more layers or more neurons per layer.
3) where it starts to do well on the training set, check if it does well in the CV set.
4) if it does not do well in CV set, get more data!
5) after it starts doing well in CV set, the algorithm is done!

A large training network will usually do as well or better than a smaller one so long as regularization is chosen appropriately. 

![image](https://user-images.githubusercontent.com/99626376/200969201-cdacfd45-6f62-4e8f-aee6-b1249b9212d9.png)

To regularize the neural network use lambda in the cost function:

![image](https://user-images.githubusercontent.com/99626376/200969447-a0dac2b1-21a3-4b82-b272-270e7a31771a.png)

It hardly ever hurts to regularize the algorithm performance. It just takes longer.


#### Iterative loop of ML development
Development a machine learning system is made by:
1) choose architecture: choose model, data, etc...
2) train model
3) diagnostics: understand bias/variance and restart with new information at (1)

![image](https://user-images.githubusercontent.com/99626376/200972679-c46101ff-f424-408c-9bea-edc0f39cd89a.png)

#### Error analysis
After bias and variance, error analysis is the most important.
Error analysis means manually looking at the errors to understand why the errors is happening.
Also looking by the errors we can find which kind of errors are more important: sometimes there are few errors that happen due to cause A but the cause A would generate a bigger problem.

![image](https://user-images.githubusercontent.com/99626376/200973619-2bce2bad-8b96-4c90-b422-26b8796b7ec0.png)

#### Adding data
Adding data can be done more efficiently if done more correctly. Some tecniques are more useful than others for each application.
How to add more data?
- add more data of the types where error analysis has indicated it might help. Go to unlabeled data and find more examples of the current problem.
- data augmentation: modifying an existing training example to create a new example. that can be done in images (changing color, font, size, etc...)

![image](https://user-images.githubusercontent.com/99626376/200974273-fc5eec6f-548e-4e18-9aad-84f4f1d4c788.png)

More advanced methods of data augmentation are done by distorting the image:

![image](https://user-images.githubusercontent.com/99626376/200974362-adde36ba-99e1-482d-b3c2-b60cac44b7b6.png)

The same can be done with speech recognition by adding background noisy:

![image](https://user-images.githubusercontent.com/99626376/200974498-bd5536c3-db85-4292-8f04-3681db7f5de6.png)

![image](https://user-images.githubusercontent.com/99626376/200974767-909f104a-04e9-4e6d-8fbe-7da130737402.png)

- data synthesis: using artificial data inputs to create a new training example. example: photo OCR: where an algorithm is asked to recognize text in image.

![image](https://user-images.githubusercontent.com/99626376/200974961-b255bbaa-72ad-4bbf-ba3b-996047ba7697.png)

One way to create new data for the task: the notebook has a lot of fonts which can be used mixed with different highlights, contrasts, etc...

In the pass:
people focused on inproving the code.

Now:
people focus on inproving data (data-centric approach)

![image](https://user-images.githubusercontent.com/99626376/200975333-9dcc03a9-531d-4fb1-8313-0fbe25012f8e.png)

Collecting more data tends to generate better outputs.

#### Transfer learning
Transfer learning: using data from a different task.
Example:
- we have a goal to construct an algorithm that can recognize which number between 0 and 9 is written.
- we do not have much data about it.
- we have much data about an algorithm used to recognize cars, cats, houses and dogs.
- we can use the parameters of the layers of the last algorithm in our model to train the goal model (considering that the features are the same because we are still talking about images).
- in this case, only the last layer of the goal model with be trained.
- in more detail, there are two ways to complete the goal model:
    - option 1: only train the output layer parameters and leave the rest (better when there is very small data set for the goal model).
    - option 2: train all the parameters from all the layers but having it inilizalized in the values of the previous parameters (better when there is a little more data for the goal model).
- by learning about one kind of thing, the model can learn about another.

![image](https://user-images.githubusercontent.com/99626376/200977676-d3c3b56d-f87c-4e9a-8288-99bdeab6ad48.png)

The first step is called (1) supervised pretraining and the second is (2) fine tuning.

#### Why does transfer learning works?
It works because the neural networks can learn to detect "edges", "corners", "curves". The new output can use those previous detections to understand the new goal. Important: the pre-training has to have the same dimensions.

1. Download NN parameters pretrained on a large dataset with same input type (image, audio, text) as your application.
2. Further train (fine tune) the network on your own data.

#### Full cycle of a machine learning project
Training a model is just part of a cycle.
When building a machine learning project:
1) scope the project: define the project goal.
2) collect data: decide what data is important (target and features).
3) train model: training, error analysis, iterative improvement.
4) get more data from other sources or from deployment.

<img width="647" alt="image" src="https://user-images.githubusercontent.com/99626376/201180980-14acf5d5-6645-499b-b81e-e0118ac57c7c.png">

Deployment: take the machine learning model and use it inside a inference server.
The model APP will make an API call to ask the inference server to run the model.
The model will work with the API to give the result back.

<img width="521" alt="image" src="https://user-images.githubusercontent.com/99626376/201180268-3b7539f4-e370-4b9c-ab5e-63c2bef40709.png">

Software engineering may be needed for:
   - ensure reliable and efficient predictions.
   - scaling.
   - logging.
   - system monitoring
   
MLOps: machine learning operations (how to maintain and construct good machine learning teams).

#### Fairness, bias, and ethics
Bias:
- hiring tool that discriminates against women.
- facial recognition system matching dark skinned individuals to criminal mugshots.
- etc...
Adverse:
- generate fake videos (Deepfake).
- fake content for commercial or political purposes.
- fraud.
Get a more diverse team to brainstorm things that might go wrong, with emphasis on possible har to vulnerable groups.
Carry out literature search on standards/guidelines for your industry.
Audit systems against possible harm prior to deployment.

#### Error metrics for skewed datasets
With very skewed datasets, accuracy does not work well.
Example: rare disease classification where only 0.5% of patients have the disease. In this case, just saying that we always have 0, would lead to an accuracy 99.5% accuracy.

#### Precision/Recall
y = 1 in presence of rare class.

We construct a matrix:

<img width="276" alt="image" src="https://user-images.githubusercontent.com/99626376/201184483-f263511f-7ea0-4b24-95b3-636ae5fd6adf.png">

True positive: predicted as positive and is positive.
True negative: predicted as negative and is negative.
False positive: predicted as positive and is negative.
False negative: predicted as negative and is positive.

<img width="278" alt="image" src="https://user-images.githubusercontent.com/99626376/201184812-3b284f9f-2985-4c11-b3f3-6210728d0f74.png">

Precision: true positives / (true positive + false positive)
Precision: true positives / predicted positives

<img width="420" alt="image" src="https://user-images.githubusercontent.com/99626376/201185079-d090f03c-c1dd-493a-9839-23c734e74d5e.png">

Recall: true positive / (true positive + false negative)
Recall: true positive / actual positives

<img width="413" alt="image" src="https://user-images.githubusercontent.com/99626376/201185361-4e850c08-d9d9-46e5-bc1b-338ceeb5f158.png">

Recall and precision are useful to understand how well an inbalanced dataset is doing.

#### Trading off precision and recall
Trade off between and precision and recall happens.
Raising the threashold will increase the precision and reduce the recall.
Lowering the threashold will reduce the precision and increase the recall.

The tradeoff can be viewed as this:

<img width="197" alt="image" src="https://user-images.githubusercontent.com/99626376/201201234-96b422dd-e248-4d36-9909-be0c72e711f8.png">

We can choose the optimal threashold by choosing in this tradeoff line.

#### F1 Score
Automatically comparing threashold. For example: in the following case, which algorithm would be best?

<img width="300" alt="image" src="https://user-images.githubusercontent.com/99626376/201201789-b902e95c-c3a8-49f4-ac9a-9f1a1d75f9a3.png">

Good way to understand which one is best is by using the F1 score:

<img width="294" alt="image" src="https://user-images.githubusercontent.com/99626376/201202385-ef16042e-aabf-4a5b-a591-9809739f91df.png">

Therefore, the F1 will generally choose the values which are not to extreme.

#### Decision Tree Model
Example: 

<img width="501" alt="image" src="https://user-images.githubusercontent.com/99626376/201203001-5b5e0423-1d97-4b2c-ae01-e2cfa182294b.png">

A decision tree algorithm:

<img width="436" alt="image" src="https://user-images.githubusercontent.com/99626376/201203065-6a26b496-c057-4691-b4de-94f3fe6b2a53.png">

A node is a separation of a feature to define the separation:

<img width="429" alt="image" src="https://user-images.githubusercontent.com/99626376/201203248-6c69d447-5d56-4bef-b9f7-c9341bb94ead.png">

Root node: the first node.
Decision nodes: new separations.
Leaf nodes: the nodes to make the decisions.

<img width="473" alt="image" src="https://user-images.githubusercontent.com/99626376/201203412-426eca6d-8897-4209-93f5-3e61113482ce.png">

Trees do not have to have the same amount of nodes in each classification:

<img width="630" alt="image" src="https://user-images.githubusercontent.com/99626376/201203590-178085b5-83e3-4939-833a-ba724351beb3.png">

#### Decision tree learning process
The process of building a decision tree is based on:
1 - decide which feature is to be used in the root node.
2 - decide the next nodes...

How to choose what feature to split?
Maximize purity (or minimize inpurity for each separation).

Purity meaning the best separation of labels by the that specific node. This relates to purity.

When do you stop splitting?
- when a node is 100% one class.
- when splitting a node will result in the tree exceeding a maximum depth.
- when improvements in purity score are below a threshold.
- when number of examples in a node is below a threshold.

We start by saying that the first node is depth 0:

<img width="196" alt="image" src="https://user-images.githubusercontent.com/99626376/201204801-36880d51-5241-478f-be02-3c2f4d6be8d8.png">

By making the tree small, we have a smaller chance to overfit.

There are many ways to stop splitting.

#### Entropy: measuring purity
Entropy is higher when the percentage of classes is more similar:

<img width="245" alt="image" src="https://user-images.githubusercontent.com/99626376/201214602-ed828083-61c4-452b-8925-027683ad494f.png">

If the group can be separated more, the entropy decreases as the entropy increases.

<img width="437" alt="image" src="https://user-images.githubusercontent.com/99626376/201214982-8ab75d73-329c-4378-af68-2d4d7b15f8cb.png">

The entropy function is found by:

<img width="380" alt="image" src="https://user-images.githubusercontent.com/99626376/201216533-99508fe0-487d-4d63-89a4-cfbaa63b886a.png">

Considering:

<img width="217" alt="image" src="https://user-images.githubusercontent.com/99626376/201216605-7c59d160-69c0-494d-ab16-9500b0f25b93.png">

Entropy is a fraction of the level impurity of a tree.

#### Choosing a split: information gain
By dividing a tree we are trying to reduce entropy by the maximum amount.
After, we continue to apply the entropy level for each decision level and each node.

<img width="658" alt="image" src="https://user-images.githubusercontent.com/99626376/201217318-4ae3b749-46ba-42d6-8469-90638cc04827.png">

The entropy says that the first separation is better: we should make a weighted average of entropy (weighted by the number of observations by branch).

Alternativily, we can compute the reduction in entropy: considering that the original group had the same amount for boths groups, the entropy of the group before was 1.

<img width="649" alt="image" src="https://user-images.githubusercontent.com/99626376/201217890-706a2116-49c1-4366-800c-b2ba42d1c440.png">

#### Information gain:

<img width="623" alt="image" src="https://user-images.githubusercontent.com/99626376/201218184-6b5b4817-b29c-4489-b75e-816853cb351f.png">

Information gain calculating by comparing entropy of before the separation to the entropy after the separation. The information gain will define the feature to split.

#### Decision tree putting it together
- start with all examples
- calculate information gain for all possible features, and pick the one with the highest information gain.
- split the dataset according to selected feature, and create left and right branches of the tree.
- keep repeating splitting process until stopping criteria is met:
    - when a node is 100% one class.
    - when splitting a node will result in the tree exceeding a maximum depth.
    - when information gain from additional splits is less than threshold.
    - when number of examples in a node is below a threshold.

Recursive algorithm: the first part of the code is the same as the next, and the next, and the next...
Each part of the decision tree is a decision tree itself.

We can pick the maximum depth by cross validation or by picking a minimun information gain or also by defining what should be the smallest number of observations in one group.

#### Using one-hot encoding of categorial features
In the last example, we only had features that had two options. Now, when we have features that take on many values, we need one-hot-encoding. That means trasforming the variables in dummies.
If a categorial feature can take on "k" values, create a "k" binary features (0 or 1 valued). Different from the regression, where we would have to put "k-1" in order to avoid multicollinearity.

#### Continuous features
How to use features that are continuous. 

![image](https://user-images.githubusercontent.com/99626376/201242969-656bcf47-ddfa-4b6e-9e0b-e6d3785836c0.png)

That would be done by applying a threshold value in the continuous feature.
The idea is to find the threshold that gives the best information gain.

![image](https://user-images.githubusercontent.com/99626376/201243145-ac83aa05-fc40-4a86-8251-0ef252107864.png)

![image](https://user-images.githubusercontent.com/99626376/201243371-a94734b6-2eda-46ba-80dd-7ac93a6d382c.png)

The continuous variable will be chosen if the information gain is bigger than any other feature.

#### Regression trees

![image](https://user-images.githubusercontent.com/99626376/201243619-514b7de6-6e4e-48db-9d9e-c17eb66c527f.png)

In this case, we would predict a number and not the classification.

The regression tree will make a decision by the average of observations.

How to choose a split?

The split should be done by separating the groups in a way that the variance of weights inside a group is small.

![image](https://user-images.githubusercontent.com/99626376/201243970-1e455fae-28f9-4c83-a3cd-606cf23fc34e.png)

The variance of each group is weighted by the amount of observations in each group:

![image](https://user-images.githubusercontent.com/99626376/201244074-2e7bee37-660c-41fa-b02b-5363a2e79c06.png)

#### Using multiple decision trees
Using multiple trees is useful when changing the sample used. For that, we train multiple decisions tree: each decision tree will have a subsample. That is called a tree ensemple!

![image](https://user-images.githubusercontent.com/99626376/201381431-af4180ae-0e4c-4fd1-a685-248474590258.png)

In the end, the predictions of each of the threes are considered. The result that gets more "votes" will be the results used.

![image](https://user-images.githubusercontent.com/99626376/201381702-57893baf-4166-4e61-959e-eceaeffbc5e9.png)

In the previous case: cat!

How do we define how to make the subsamples?

#### Sampling with replacement
Used to make a tree with sampling we need this!

In that we:
- take all the observations in a "box".
- take a sample from the observations in the "box.
- make a decision tree with that sample.
- give back the observations in the sample to the observations "box".
- retake a new sample from the box (can have the same observations as before)...

By using this tecnique multiple times, we get different samples...

We construct multiple training sets that will have the same training size then the original, but might have repeated the same observation.

![image](https://user-images.githubusercontent.com/99626376/201382844-66eac932-81ff-4929-8726-8aec57fa70c3.png)

There will be repeat observations (and will not contain all the observations).

#### Random forest algorithm (bagged decision tree)
Works much better than using a single tree.

![image](https://user-images.githubusercontent.com/99626376/201387683-916bd91e-0e7e-446b-be70-d9c2db293d39.png)

Using more than 100 trees does not help, therefore, there is no need to make thousands.

It is common that the first node is the same for most of the trees.

To make the features used different for many different trees, we should use a subset of features to decide, being a that number of features "k" (k < n). Generally, the number of subset features chosen is $k=n^{1/2}$. With larger number of features, this tends to work better.

The sampling makes the algorithm explore more possibilities. Any further change, has a smaller chance of being different from the samples used.

#### XGBoost: tree emsemble
XGBoost is advanced quickly and really good!
There is a small modification from the normal random forest.

Steady of picking the sample observations with the same probability, we assign a higher probability to pick examples that were misclassified before.

![image](https://user-images.githubusercontent.com/99626376/201392747-1251c512-b337-4f21-aaca-ad0da96da927.png)

The next decision tree will focus more attention in the samples that were poorly trained before.

This is the idea behind boosting!

![image](https://user-images.githubusercontent.com/99626376/201393178-76ae7741-8ac0-4136-8d29-83ceeae6c9d5.png)

The boosting procedure will consider not only the misclassified results from the last interation, but from all the previous ones.

There is a math behind the chance of picking a bad vs a good classification observation.

![image](https://user-images.githubusercontent.com/99626376/201394085-404fe0cc-9730-444d-8779-77fc60f6e1b8.png)

XGBoost and NN are the ones which gain more competitions!

![image](https://user-images.githubusercontent.com/99626376/201394316-2410ce28-8643-4668-88a3-79a9d9cd0ef9.png)

#### When to use decision tree?
Decision trees and tree ensemble:
- works well on tabular data.
- 










   

