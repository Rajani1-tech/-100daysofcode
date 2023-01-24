# -100daysofcode
# Day 1 of #100daysofcode

I've gained knowledge of the core concepts of deep learning, including concepts like neuron networks, activation functions, and hidden layers.
In order to assess how the model's performance changes when a hidden layer is introduced,
I have also classified handwritten digits using a rudimentary neural network with just input and output layers.

![Screenshot (39)](https://user-images.githubusercontent.com/83020452/212559783-60fb1d8b-d541-4132-a43e-f73682ba897d.png)

# Day 2 0f #100daysofcode

Activation function refers to the output of that node given an input or set of inputs. It maps the resulting values in between 0 to 1 or -1 to 1 etc. (depending upon the function).

1. Sigmoid Activation Function 
It takes a real number as input and outputs a probability that is always between 0 and 1. The sigmoid is a monotonic, continuously differentiable, nonlinear function with a set output range. The function's main drawback is that it causes the issue of "vanishing gradients" since its output isn't zero-centered. In the neural network's buried layer, computation also requires a lot of time.
 
2. Tanh Activation Function  
It is similar to a logistic sigmoid but superior. The tanh function's range goes from (-1 to 1). Additionally sigmoidal is tanh (s-shaped). The positive aspect of this is that the zero inputs will be mapped close to zero in the tanh graph while the negative inputs will be significantly negative.

Both sigmoid and Tanh activation functions are used in feed-forward nets. 
 
3. ReLU Activation Function  
The ReLU is the most used activation function in the world right now. Since it is used in almost all convolutional neural networks or deep learning. The formula is deceptively simple: max (0, z). Its key benefit is that, compared to tanh and sigmoid, it avoids and corrects the vanishing gradient problem and requires less processing effort. 
 
4. Leaky ReLU  
It is an upgraded variant of ReLU with a tiny slope for negative values rather than a flat slope. Before training, the slope coefficient is calculated. It is often used in situations where sparse gradients may be an issue. 

![https___mlfromscratch_com_content_images_2019_12_activation-functions](https://user-images.githubusercontent.com/83020452/213482102-253b09b0-9d67-4ba1-a300-f39ccc0259f7.gif)


# Day 3 of #100daysofcode

Today, as part of my learning adventure, I discovered loss functions and how they are implemented using Tensorflow. Only MSE has been deployed so far; the rest will be in the next days.


Loss Functions
It is the most crucial component of neural networks since it, together with the optimization functions, is directly in charge of adjusting the model to the supplied training data.

Types of Loss Functions
1. Regression Loss Functions are employed in regression neural networks; given an input value, the model forecasts a matching output value (instead of using pre-selected labels); Example: Mean Squared Error, Mean Absolute Error

2. Classification Loss Functions are used in classification neural networks; given an input, the network generates a vector of probabilities for the input to belong to several pre-set categories; the neural network may then choose the category with the highest likelihood of belonging; For instance, categorical cross-entropy and binary cross-entropy
![image](https://user-images.githubusercontent.com/83020452/213480190-ae15fc57-f604-45f3-9c93-ad145611a818.png)
![image](https://user-images.githubusercontent.com/83020452/213480246-40769e33-e9c3-4dfd-be9e-bfdb203e60eb.png)

# Day 4 of #100daysofcode
Today, I developed Binary Cross-Entropy, a sort of classification loss function, continuing my learning process from yesterday when I implemented MSE, a regression loss function.

![image](https://user-images.githubusercontent.com/83020452/213482000-f69bcf83-a59e-482d-b233-1133b8b7313b.png)



# Day 5 of #100daysofcode

On my deep learning adventure, today I learned about optimization, its many forms, and how to pick the optimal optimizer to shorten training time. I just know a handful of its sorts, and I'll learn the rest in the following days.
Optimizers are algorithms or techniques that alter the weights and learning rates of neural networks in order to minimize losses and produce the most precise results.
Below is an explanation of several optimizers:

1. Batch Gradient Descent (BGD)
BGD is a variation of the gradient descent algorithm that calculates the error for each of the training datasets, but only updates the model after all training examples have been evaluated.

2. Mini Batch Gradient Descent(MGD)
MGD is a variation of the gradient descent algorithm that splits the training datasets into small batches that are used to calculate model error and update model coefficients.

3. Stochastic Gradient Descent(SGD)
SGD is a variation of the gradient descent that calculates the error and updates the model for each record in the training datasets.

4. Gradient Descent with Momentum
It is invented for reducing high variance in SGD and softening the convergence. It accelerates the convergence towards the relevant direction and reduces the fluctuation to the irrelevant direction.

![image](https://user-images.githubusercontent.com/83020452/213480603-c9759056-a64c-43f8-91e5-85b8577bf6bd.png)

# Day 6 of #100daysofcode

I have concluded learning the several optimizer kinds that were left over from yesterday.

1. Adagrad Optimizer
Adagrad is an optimizer with parameter-specific learning rates, which are adapted relative to how frequently a parameter gets updated during training. The more updates a parameter receives, the smaller the updates. Due to its quicker learning rate reduction for frequent parameters and slower learning rate reduction for uncommon parameters, this method works well with sparse data.
The denominator grows aggressively monotonically as squared gradients are added, which is AdaGrad's drawback. The learning rate eventually reaches an infinitesimally tiny value, at which time the algorithm effectively stops moving in the direction of the minimum.

2. Root Mean Squared Propagation
When AdaGrad was getting near to convergence, it became immobile because the degraded learning rate prevented it from moving in the vertical direction. This issue is solved by RMSProp by being less aggressive toward decay.
RMSProp, is a variation on gradient descent, and the AdaGrad version of gradient descent that adapts the step size for each parameter using a declining average of partial gradients. The adaptive learning rate's focus on more recent gradients is managed by the hyperparameter,
sometimes referred to as the decay rate. RMSProp will often perform better than AdaGrad.

3. Adam Optimization
Adam, which stands for Adaptive Moment Estimation, combines the concepts of Momentum Optimization and RMSProp. Momentum Optimization maintains a record of an exponentially decaying average of previous gradients, while RMSProp maintains a record of an exponentially decaying average of previously squared gradients. Adam uses the average of the second moments of the gradients in addition to the average of the first moments, which is how RMSProp adjusts the parameter learning rates (the uncentered variance).

![1674230353703](https://user-images.githubusercontent.com/83020452/214364787-a42df3f1-636a-4ef4-8dd6-2c12b8396e16.gif)

# Day 7 of #100daysofcode

Today, as I continue on my deep learning adventure, I learned once more about the numerous neural network types, their applications, and both their pros and cons.

![1674319388048](https://user-images.githubusercontent.com/83020452/214365138-cd21d338-2d33-4d06-a2b9-06de04752865.jpg)

# Day 8 of #100daysofcode
On my learning journey today, I discovered the Convolutional Neural Network, its fundamental parts, and how it is implemented.

CNN is a deep learning model that can handle data having a grid pattern, like photographs, from simple to complex patterns.
The following five fundamental elements make up CNN:
1. Convolution: identifying features in a picture; 
 2. ReLU: to smear the picture and highlight boundaries
3. Pooling: for repairing damaged images
4. Flattening: making the image into an acceptable representation 
5. Full connection: to use a neural network to analyze data

![1674497375323](https://user-images.githubusercontent.com/83020452/214365327-5a766b24-33ad-4918-952f-4713d21a8d1e.jpg)

![1674497375218](https://user-images.githubusercontent.com/83020452/214365350-c442292c-a0f6-4ab4-96a1-9a8aaf945466.jpg)

  # Day 9 of #100daysofcode

Today, as part of my deep learning quest, I learned about LeNET and its implementation.


![1674580237980](https://user-images.githubusercontent.com/83020452/214365592-e1312a64-c3de-4493-9148-1cbb9531cd9a.jpg)




