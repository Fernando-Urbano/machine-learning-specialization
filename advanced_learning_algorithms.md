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
![image-13.png](attachment:image-13.png)
After training the model for a lot of faces, we can check what are the characteristics (features) that the neuron is looking for.
In the first layer, we often see that the neurons are looking for very short lines.
![image-14.png](attachment:image-14.png)
In the second layer, the neurons try to find bigger parts. The first neuron is trying to detect the presence of an eye, the second tehe superior part of an eye, etc...
![image-15.png](attachment:image-15.png)
After, in the last layer, the neuron tries to find the shapes of faces which correspond the best.

The cars would be the same idea: smaller features, middle features, bigger features...

![image-16.png](attachment:image-16.png)

# 2) Advanced Learning Algorithms
## 2.2) Neural Network Model
### Neural Network Layer
In this video, we learn how to construct a layer of neural network.
The first hidden layer will take the inputs. The first parameter will be a logistic regression taking the parameters "vector w" and "b" to identify z and than g(z).
In the example, it uses one feature for each neuron.
![image-18.png](attachment:image-18.png)
The vectors of three numbers generated by the three parameters will become the "vector a" which will be useful to get the output layer or the following hidden layer.

It is useful to enumerate the layers:
- input layer: layer 0;
- hidden layer: layer 1 to layer (n - 1)
- output layer: layer n

The [1] would be the output of layer 1:
![image-19.png](attachment:image-19.png)
This is useful to make the reading easier.

Layer 2 will use the activation vector steady of the inputs to generate a new logistic regression.
![image-20.png](attachment:image-20.png)

The next step will define if the predicted value is 0 or 1.
![image-21.png](attachment:image-21.png)
The a[2] is different from the many a[1]
### More Complex Neural Networks
More complex:
![image-22.png](attachment:image-22.png)
If we say that the neural network has "n layers" it includes the input layer, but not the output layer.
The third layer will have the activation function based on layer 2:
![image-23.png](attachment:image-23.png)
![image-24.png](attachment:image-24.png)
The following considers:
- the parameters computed in layer 3.
- the activation vector fro layer 2 ("a" vector)
to generate other "a" scalars. The "a" scalars of layer 3 are put together to compute parameters of layer four (w[4] and b[4]).
The notation will always be like this:
![image-25.png](attachment:image-25.png)
When l = 1, we actually have "x vector" (inputs).
### Forward Propagation: Making Predictions with Neural Networks
A handwritten digit recognition to define if we have written a 0 or a 1.

We make a 64 pixel intensity.
![image-26.png](attachment:image-26.png)
To classify, we make a neural network with 3 layers:
- layer 0: input
- layer 1: 25 units/neurons
- layer 2: 15 units
- output layer
![image-27.png](attachment:image-27.png)
The first layer will generate a vector of 25 scalars. For that, we need 25 functions. The 25 functions will have a matrix with "m" observations and 64 (n) features.
The second layer will generate a vector of 15 scalars. For that, we need 15 functions. The 15 functions will a matrix with "m" observations and 25 (n, which is the same as a[1]) features.
The third layer (output layer) will generate a scalar. For that, we need 1 function. The 1 function will have a matrix with "m" observations and 15 (n, which is the same as a[2]) features.

This algorithm is called forward propagation.
On the other hand, the backwards propagation is used to learn.

# 2) Advanced Learning Algorithms
## 2.3) Neural Network TensorFlow Implementation
### Inference in Code
We want to predict if this is a good roasted coffee or not based on the temperature and duration of the roast.
