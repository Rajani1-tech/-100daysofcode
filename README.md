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
 # Day 10 of #100daysofcode
 
 Today, I discovered AlexNet and how it operates.
AlexNet is the deep learning architecture that popularized CNN. In terms of design, the AlexNet network was quite similar to the LeNet network, but it was deeper, larger, and contained Convolutional Layers piled on top of one another. At the time of its release, the AlexNet architecture produced findings that were cutting-edge for application with large-scale picture collections. Max-pooling layers, fully connected layers, dropout layers, and three more types of layers make up AlexNet's five convolutional layers. All layers employ the Relu activation function. The output layer has Softmax as its activation function. This architecture has almost 60 million parameters overall.

![1674669919777](https://user-images.githubusercontent.com/83020452/216763132-a3234f2a-b944-4953-9bd4-998a2bddeedd.jpg)

# Day 11 of #100daysofcode

I gained knowledge of many CNN architectures, like GoogLeNet and VGGNet, and I also discovered how to build them using Keras.

# Day 12 of #100daysofcode

Today I learned about Xception while continuing my deep learning quest.
Xception is a convolutional neural network (CNN) architecture developed by Google for image classification tasks. It is a variant of the Inception architecture and is designed to improve upon the Inception model by reducing the number of computations required while maintaining or improving accuracy. Xception is known for its exceptional performance on image classification tasks and is often used as a feature extractor in computer vision applications.

It combines the concepts of Google Neural Network and ResNet Architecture. The data initially passes via the entering flow, following which it moves through the middle flow, where it repeats itself eight times, and ultimately passes through the exit flow.



![1674929064764](https://user-images.githubusercontent.com/83020452/216763201-36a5fd23-1c2c-4cf7-9e3f-016cdc8f5393.jpg)

# Day 12 of #100daysofcode

Today, while exploring deep learning, I learned about RNN and gained a basic understanding of its purpose, workings, and applications.

RNN stands for Recurrent Neural Network, a type of neural network used for processing sequential data such as time-series data, speech, text, and video. RNNs have a feedback loop that allows information to persist from one step of the sequence to the next, allowing the network to capture temporal dependencies in the data.
RNNs are used because they are effective in processing sequential data by retaining information from previous time steps, allowing the network to capture patterns and dependencies in the data. This makes RNNs well-suited for tasks such as:
1. Natural language processing (NLP) and text generation
2. Speech recognition and synthesis
3. Time-series prediction and analysis
4. Video classification and captioning
5. Machine translation.


![1675100505171](https://user-images.githubusercontent.com/83020452/216763302-f4255f7e-fdda-4e00-89a8-d514c5336ca5.gif)



# Day 13 of #100daysofcode

Today, I gained knowledge on RNN operation.
In Recurrent Neural networks, the information cycles through a loop to the middle hidden layer. The input to the neural network is received by layer "x," which processes it before sending it to layer "m."

There may be several hidden layers in the middle layer "h," each with its own activation functions, weights, and biases. We can utilize a recurrent neural network if our neural network's hidden layers' various parameters are independent of each other and the prior layer, or if our neural network lacks memory.


![1675187296216](https://user-images.githubusercontent.com/83020452/216763347-99e8979e-0a01-4887-800f-80de53d6b35a.gif)

# Day 14 of #100daysofcode

Today I explored the following sorts of RNN:
1. One to One RNN
This type of neural network is known as the Vanilla Neural Network.It is applied to issues in generic machine learning that have a single input and a single output.

2. One to Many RNN
This type of neural network has a single input and multiple outputs. An example of this is the image caption.

3. Many to One RNN
A series of inputs are used by this RNN to produce a single output. Sentiment analysis is a good example of this type of network, which allows for the classification of a given sentence as reflecting either positive or negative thoughts.

4. Many to Many RNN
This RNN takes a sequence of inputs and generates a sequence of outputs. Machine translation is one example.



![1675444002702](https://user-images.githubusercontent.com/83020452/216763397-547d9e9d-c74d-4756-919d-8b949271d4b3.jpg)


# Day 15 0f #100daysofcode

I learned about the implementation of RNN today, and the following are the steps I took to do it.

1. Read the dataset from the specified URL.

2. Separate the data into training and test sets.

3. Prepare the input according to the necessary Keras format.

4. Create an RNN model, then train it.

5. Make the predictions on the training and test sets, then display the root mean square error for each set.

6. See the outcome

![image](https://user-images.githubusercontent.com/83020452/221071044-bfcec7c9-aac1-4701-a990-646c9a97603c.png)
![image](https://user-images.githubusercontent.com/83020452/221071019-472c63eb-59b3-4db7-88d0-651b563cde22.png)

 # Day 16 of #100daysofcode

I am now aware of the two main problems with RNN as a consequence of learning and implementing it. I also discovered today how to address these problems.

1. Vanishing Gradient Problem
The vanishing gradient problem in RNNs refers to the difficulty in training long-term dependencies in sequential data. The problem occurs when the gradients used to update the model's parameters become extremely small, making it difficult for the optimizer to update the weights and improve the model. This can lead to poor performance, especially when working with longer sequences.
Solutions to this problem include using more advanced RNN architectures, such as LSTMs and GRUs, which are better suited to handle long-term dependencies.

2. Exploding Gradient Problem
The exploding gradient problem in RNNs refers to the situation where the gradients used to update the model's parameters become extremely large, causing the model to diverge and not converge during training. This can lead to numerical instability, causing the model's parameters to have very large values, resulting in poor performance. The exploding gradient problem is often a result of using a large learning rate or using a deep RNN with many layers, which can cause the gradients to become very large and unstable.
Solutions to this problem include using gradient clipping, where the gradients are clipped to a maximum value to prevent them from becoming too large.

![image](https://user-images.githubusercontent.com/83020452/221072391-28b5b54a-2b2e-4534-b0e7-3e80788c8347.png)

# Day 17 of #1oodaysofcode

Today I gained some more in-depth knowledge about LSTM, how it functions, and some of its uses.

LSTM stands for Long Short-Term Memory, which is a type of Recurrent Neural Network (RNN) architecture used for processing sequential data. LSTMs are designed to overcome the vanishing gradient problem faced by traditional RNNs by using a gating mechanism to control the flow of information through the network and preserve long-term dependencies. LSTMs are widely used in a variety of applications, including natural language processing, speech recognition, and time-series prediction.


LSTMs work by introducing a series of gates within the network to control the flow of information and preserve long-term dependencies. The gates consist of sigmoid neural networks that learn to open and close based on the input data. The LSTM has three main components: the forget gate, input gate, and output gate.

1. The forget gate determines which information from the previous state should be forgotten.

2. The input gate decides which new information from the current input should be added to the cell state.

3. The output gate decides what information from the cell state should be passed on to the output.

The cell state acts as a memory store within the LSTM, and its content is updated at each time step based on the input, forget, and input gates. The output of the LSTM is produced based on the output gate and the cell state. By controlling the flow of information through the network in this way, LSTMs can effectively preserve and use long-term dependencies in sequential data.

![image](https://user-images.githubusercontent.com/83020452/221071361-712f228a-82d3-4b5f-8235-e55dd26fbb40.png)

# Day 18 of #100daysofcode

I forgot to post about my learning yesterday, so I figured I'd do it today. As I become more familiar with Gated Recurrent Unit, I learn how it differs from LSM.

A gated recurrent unit (GRU) is a gating mechanism in recurrent neural networks (RNN) similar to a long short-term memory (LSTM) unit but without an output gate. When working with smaller datasets, they perform better than LSTM.

GRUs are able to solve the vanishing gradient problem by using an update gate and a reset gate. The reset gate regulates information that exits memory, and the update gate regulates information that enters memory. The information that will be conveyed to the output is chosen by both of them, and each operates as a vector.


It can be applied to the modeling of polyphonic music, speech signals, and handwriting recognization.

![image](https://user-images.githubusercontent.com/83020452/221072481-f754d8ae-4d32-4018-b61b-02350af2d581.png)

# Day 19 of #100daysofcode

I completed the project Fake News Classifier using LSTM while learning deep learning.

![image](https://user-images.githubusercontent.com/83020452/221072619-23225ecd-2699-4e9c-a84a-10f98ce4841a.png)
![image](https://user-images.githubusercontent.com/83020452/221072643-2ec69653-e1f6-482f-a553-fe18de5b5ce7.png)

![image](https://user-images.githubusercontent.com/83020452/221072666-b43095a6-ef2b-4975-a706-6230e7fcb2b7.png)
![image](https://user-images.githubusercontent.com/83020452/221072685-a6305219-cb62-4f69-90cb-be9d0d366b2d.png)


# Day 20 of #100daysofcode

Today, I discovered bidirectional LSTM, how it operates, and how it differs from LSTM.

An RNN that can analyze both forward and backward data sequences is known as a bidirectional LSTM. This makes it particularly helpful for applications like speech recognition, natural language processing, and image captioning that require understanding the context of the entire input sequence.

The main characteristic of a bidirectional LSTM is the employment of two independent LSTM layers, one for processing the input sequence forward and another for processing the sequence backward. The two layers' outputs are then concatenated, giving the model a more thorough description of the input sequence. The output that has been concatenated can then be utilized to construct a prediction or a sequence.

One advantage of utilizing a bidirectional LSTM is that it can detect long-term dependencies in the data that conventional LSTMs could overlook. The model is able to incorporate both past and future context by processing the input sequence in both directions, which can be very helpful in situations where the outcome is dependent on the complete input sequence. Moreover, when the input sequence is noisy or lacking, bidirectional LSTMs may perform better than unidirectional LSTMs.

![image](https://user-images.githubusercontent.com/83020452/221072806-be94ac6e-7bc7-4280-979f-a9c698d9cc76.png)


# Day 21 of #100daysofcode

Today, I learn more about word embedding and word2Vec.

Word embedding is a technique used in natural language processing and machine learning that maps words or phrases to vectors of real numbers in a high-dimensional space.

The process of creating word embeddings involves analyzing large amounts of text data to learn a mapping from words to numerical vectors. This can be done using various techniques such as neural networks, principal component analysis (PCA), or matrix factorization.

Similarly, Word2Vec is a popular unsupervised learning algorithm used for generating word embeddings from text data. The algorithm is based on the idea that the meaning of a word can be inferred from its context, and that similar words will have similar contexts.

The Word2Vec algorithm trains a neural network on a large corpus of text data to learn vector representations of words. There are two main approaches to training a Word2Vec model: Continuous Bag of Words (CBOW) and Skip-gram.


# Day 22 of #100daysofcode
With the help of bidirectional LSTM, I classified sequences today.

![image](https://user-images.githubusercontent.com/83020452/221072934-c3660662-9a4a-45f7-8170-ccf5f1b3f047.png)


 # Day 23 of #100daysofcode
 
I learned about the attention model today.

Attention models are neural network input processing strategies that enable the network to concentrate on particular elements of a complicated input, one at a time until the entire dataset is categorized. The idea is to divide difficult tasks into manageable attentional chunks that are processed sequentially. Similar to how the human mind breaks down a new challenge into smaller jobs and tackles them one at a time.
Continuous reinforcement or backpropagation training is required for attention models to function well.

Attention is used for a variety of activities, including memory in neural Turing machines, reasoning in differentiable neural computers, language processing in transformers and LSTMs, and multisensory data processing (sound, pictures, video, and text) in perceivers.

![image](https://user-images.githubusercontent.com/83020452/221073012-ac08579f-9221-4761-9c74-e3fffe7336c3.png)


Day 24 of #100daysofcode
I discovered today what bert is and how it works.
Bert (Bidirectional Encoder Representations from Transformers) is a natural language processing (NLP) model developed by Google in 2018.

Bert works by pre-training a deep neural network using a large corpus of text data. During pre-training, the model is trained to predict missing words in a sentence (masked language modeling) and to identify the relationship between two different sentences (next sentence prediction).

The pre-training process allows Bert to learn a set of contextualized representations for words, which can be fine-tuned on a wide range of downstream NLP tasks. This means that Bert can be applied to different NLP tasks without needing to be re-trained from scratch each time.
When Bert is used for a specific NLP task, such as sentiment analysis, the model is first fine-tuned on a labeled dataset specific to that task.

During fine-tuning, the weights of the pre-trained Bert model are adjusted to optimize the model for the specific task. Once fine-tuning is complete, the Bert model can be used to make predictions on new text data.


![image](https://user-images.githubusercontent.com/83020452/226668400-e706fd70-1af2-42aa-a339-79f33b285ac3.png)


Day 25 of #100daysofcode
Today, I discovered several sorts of hyperparameter tuning.

Hyperparameter tuning is the process of selecting the best set of hyperparameters for a machine-learning algorithm. Hyperparameters are variables that are set before the model training process and cannot be learned from data. Examples of hyperparameters include learning rate, batch size, number of hidden layers, and regularization strength.

The process of hyperparameter tuning involves searching through a space of possible hyperparameter values and evaluating the performance of the model with each set of hyperparameters. This can be done using techniques like grid search, random search, or Bayesian optimization. The goal is to find the set of hyperparameters that results in the best model performance on a validation set, without overfitting the training set.

Grid Search
Grid search is a hyperparameter tuning technique that involves searching through a manually defined set of hyperparameter values for a machine learning algorithm. It works by creating a grid of all possible combinations of hyperparameters and evaluating the performance of the model for each combination.

Random Search
Randomized search is a hyperparameter tuning technique that involves randomly sampling hyperparameters from a distribution instead of exhaustively searching through a predefined set of hyperparameters like in grid search.


![image](https://user-images.githubusercontent.com/83020452/226668698-98bf0cf8-f7fd-4d57-b99d-9043e9612e83.png)



