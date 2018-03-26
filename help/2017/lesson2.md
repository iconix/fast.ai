# Lesson 2: Understanding the finetuning and fitting processes; technical intro to CNNs

[Lesson page](http://course17.fast.ai/lessons/lesson2.html) | [Wiki notes](http://wiki.fast.ai/index.php/Lesson_2_Notes) | [Video](https://www.youtube.com/watch?v=e3aM6XTekJc)

## Lesson 1 Homework Review

**Standard 5 steps for debugging computer vision problems.** Look at examples (in validation set) of each of:
1. A few correct labels at random
2. A few incorrect labels at random
3. The most correct labels of each class (highest probability of being correct)
4. The most incorrect labels of each class (highest probability of being incorrect)
5. The most uncertain labels (probabilities closest to 0.5)

Probability outputs of a deep learning network are **mathematical, not statistical**.
- Mathematical probability - all possible output probabilities add up to 1
- Not statistical probability - a measurement of likelihood for the output

## New Content

### What is a deep neural network?

- Jeremy: "Basically is a bunch of matrix products followed by activation functions"
- Remember (from Lesson 0): _Deep Learning = Linear Algebra + Optimization_, and **Convolutional Neural Networks (CNNs) = Convolutions + Optimization**
    - **Convolutions: subset of Linear Algebra that finds the fingerprint of an image.** It takes a "filter" (e.g., 3x3 matrix), "correlates" it (i.e., multiplies the filter by each area) to an image, and then rotates the filter by 90 degrees and repeats the correlation until the filter has been rotated and correlated in all possible directions on the image
    - **Convolutions are a method of feature engineering in deep learning**
        - Manual feature engineering is pre-deep learning
- Deep learning takes the _activations_ (outputs of the matrix products) and puts them through a **non-linearity** of some sort
    - Tanh, sigmoid...
    - Most commonly nowadays: `max(0,x)` - rectified linear activation (relu)
        - Basically just turns negative numbers into 0
- relu (and minor variations) for intermediate layers and softmax for output layers will be by far the most common activation function config

### Why finetune?

It's smart to try to take advantage of the **prior knowledge** encoded in _low-level filters_ (e.g., features core to any computer vision problem) learned by a pre-trained model, especially if the pre-trained model learned from a larger dataset than available to you. **Finetuning** is the process of incorporating these low-level filters while re-mapping the higher-level filters to whatever classification task you're currently attempting to do.

### Fitting

Neural network (weight) initialization is key to the success of efficiently learning in a neural network. Even with random weights, it's still important that the corresponding random output is on a similar scale to the expected output so that learning doesn't take forever.
- Modern deep learning libraries (e.g., Keras) handle weight initialization for you

Once initialized with random weights, **fitting** begins - the process of going from random weights producing random output to weights that will get us as close to our target outputs as possible. To do this, we need an **optimization algorithm**. The most common optimization algorithm, and one that is ubiquitous throughout deep learning, is **Gradient Descent**.
- Rachel: "It's powerful to think that you can start with something random and by iterating eventually get to something that works."

# Reading Notes

Lesson 2 Readings:
- Stanford's [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/) - The following from module 1:
    - Optimization: Stochastic Gradient Descent
    - Backpropagation, Intuitions
    - Neural Networks Part 1: Setting up the Architecture
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap1.html) book - chapters 1, 2, & 3
- [A unique path to deep learning expertise](http://www.fast.ai/2016/10/08/overview/) - (optional) some brief thoughts about the teaching methods used in the class
- [A Mathematician's Lament](https://www.maa.org/external_archive/devlin/LockhartsLament.pdf) - (optional) more background on the teaching methods for this course

[Question running through my mind as I read the Stanford CNN pages] Why read this? This class has been professed as not being about the math...
- Maybe because I knew this premise of the class, I was able to lighten my expectations of understanding every equation and instead, I focused on the intuition. Then the reading wasn't so bad and became more enjoyable.
- Maybe they plan on wedging the underlying complexity into our heads, slowly and with pace over time
- [[Rachel](http://forums.fast.ai/t/is-it-fine-if-i-do-not-understand-the-maths-completely-in-lesson-2-readings/3323/4)] "**Definitely more about getting intuition!** Also, don't let trying to understand all the details bog you down, because **the course returns to concepts several times, with progressively more detail as time goes on**."

[[Lesson 2 Discussion](http://forums.fast.ai/t/lesson-2-discussion/161/2?u=iconix)] "There are some fantastic readings that cover similar material to this week's course. I've added them to the lesson wiki page. In particular, the Stanford CNN course is some of the best technical writing I've seen, and is totally up to date. **If you have 8 hours to spend this week, spending half that time reading the Stanford notes would be an excellent use of your time this week.**" - Jeremy Howard

## CS231n

### Optimization: Stochastic Gradient Descent

http://cs231n.github.io/optimization-1/ : _optimization landscapes, local search, learning rate, analytic/numerical gradient_

Three key components to image classification:

1. [Infinitely Flexible Function] (Parameterized) **score function** mapping raw image pixels to class scores
    - Linear function
    - Neural Networks
    - Convolutional Neural Networks
2. [Parameter Fitting I] **Loss function** measuring the quality of a particular set of parameters based on how well the induced scores agreed with the ground truth labels in the training data.
    - Softmax
    - SVM
3. [Parameter Fitting II] **Optimization** to find the set of parameters `W` that minimize the loss function
    - Backpropagation: analytical (calculus-based) Gradient Descent via the chain rule

"The **gradient** is just a **vector of slopes** (more commonly referred to as **derivatives**) for each dimension in the input space."
- When the functions of interest take a vector of numbers instead of a single number, we call the derivatives partial derivatives, and the gradient is simply the **vector of partial derivatives** in each dimension.

#### Core idea: iterative refinement

Of course, it turns out that we can do much better. The core idea is that finding the best set of weights W is a very difficult or even impossible problem (especially once W contains weights for entire complex neural networks), but the problem of refining a specific set of weights W to be slightly better is significantly less difficult. In other words, our approach will be to start with a random W and then iteratively refine it, making it slightly better each time.

**Our strategy will be to start with random weights and iteratively refine them over time to get lower loss.**

#### Blindfolded hiker analogy

One analogy that you may find helpful going forward is to think of yourself as **hiking on a hilly terrain with a blindfold on, and trying to reach the bottom**. In the example of CIFAR-10, the hills are 30,730-dimensional, since the dimensions of W are 10 x 3073. **At every point on the hill we achieve a particular loss (the height of the terrain).**
- Following the Gradient: In our hiking analogy, this approach roughly corresponds to feeling the slope of the hill below our feet and stepping down the direction that feels steepest.

#### SGD

In large-scale applications (such as the ILSVRC challenge), the training data can have on order of millions of examples. Hence, it seems wasteful to compute the full loss function over the entire training set in order to perform only a single parameter update. A very common approach to addressing this challenge is to compute the gradient over **batches** of the training data.

...

The extreme case of this is a setting where the mini-batch contains only a single example. This process is called **Stochastic Gradient Descent (SGD)** (or also sometimes **on-line** gradient descent). This is relatively less common to see because in practice due to vectorized code optimizations it can be computationally much more efficient to evaluate the gradient for 100 examples, than the gradient for one example 100 times. Even though SGD technically refers to using a single example at a time to evaluate the gradient, **you will hear people use the term SGD even when referring to mini-batch gradient descent** (i.e. mentions of MGD for "Minibatch Gradient Descent", or BGD for "Batch gradient descent" are rare to see), where it is usually assumed that mini-batches are used. The size of the mini-batch is a hyperparameter but it is not very common to cross-validate it. It is usually based on memory constraints (if any), or set to some value, e.g. 32, 64 or 128. We use powers of 2 in practice because many vectorized operation implementations work faster when their inputs are sized in powers of 2.

#### Core Takeaway

**The ability to compute the gradient of a loss function with respect to its weights (and have some intuitive understanding of it) is the most important skill needed to design, train and understand neural networks.**

### Backpropagation, Intuitions

http://cs231n.github.io/optimization-2/ : _chain rule interpretation, real-valued circuits, patterns in gradient flow_

Backpropagation, which is a way of computing gradients of expressions through recursive application of chain rule.

`f(x,y,z)=(x+y)z, where x = -2; y = 5; z = -4`

![The real-valued "circuit" on left shows the visual representation of the computation](https://github.com/iconix/fast.ai/raw/master/help/assets/real_valued_circuit.png)

The real-valued "circuit" on left shows the visual representation of the computation. The forward pass computes values from inputs to output (shown in green). The backward pass then performs backpropagation which starts at the end and recursively applies the chain rule to compute the gradients (shown in red) all the way to the inputs of the circuit. The gradients can be thought of as flowing backwards through the circuit.

Notice that **backpropagation is a beautifully local process**. Every gate in a circuit diagram gets some inputs and can **right away compute two things: 1. its output value and 2. the local gradient of its inputs with respect to its output value**. Notice that the gates can do this completely independently without being aware of any of the details of the full circuit that they are embedded in.

**Patterns in backward flow**: add gate, max gate, multiply gate, unintuitive effects and their consequences

**Staged computation** is important for practical implementations of backpropagation. You always want to break up your function into modules for which you can **easily derive local gradients**, and then chain them with chain rule... Hence, decompose your expressions into stages such that you can differentiate every stage independently (the stages will be matrix vector multiplies, or max operations, or sum operations, etc.) and then backprop through the variables one step at a time.

#### Common stages

- Add
    - ![Add gate](https://github.com/iconix/fast.ai/raw/master/help/assets/add_gate.png)
- Max
    - ![Max gate](https://github.com/iconix/fast.ai/raw/master/help/assets/max_gate.png)
- Mult
    - ![Mult gate](https://github.com/iconix/fast.ai/raw/master/help/assets/mult_gate.png)
- 1/x
- Translation (c + x)
- e^x
- Scale (ax)

    ![Other gates](https://github.com/iconix/fast.ai/raw/master/help/assets/gates_etc.png)

### Neural Networks Part 1: Setting up the Architecture

http://cs231n.github.io/neural-networks-1/ : _model of a biological neuron, activation functions, neural net architecture, representational power

The area of Neural Networks has originally been primarily inspired by the goal of modeling biological neural systems, but has since diverged and become a matter of engineering and achieving good results in Machine Learning tasks.

#### Biological motivation and connections

|
![Biological neuron](https://github.com/iconix/fast.ai/raw/master/help/assets/neuron_bio.png)
|
![Deep learning neuron](https://github.com/iconix/fast.ai/raw/master/help/assets/neuron_dl.jpg)
|

**A single neuron can be used to implement a binary classifier** (e.g. binary Softmax or binary SVM classifiers)

#### Common activation functions

- Sigmoid: takes a real-valued input (the signal strength after the sum) and squashes it to range between 0 and 1
    - **Never use**
- Tanh: squashes a real-valued number to the range [-1, 1]
    - **Try this**, but expect it be worse than ReLU/maxout
- Rectified Linear Unit (ReLU): the activation is thresholded at zero, using f(x) = max(0, x)
    - **Use this**, but be careful with your learning rates and possibly monitor the fraction of "dead" units in a network
- Leaky ReLU
    - **If concerned about ReLU caveats, try this**
- Maxout
    - Notice that both ReLU and Leaky ReLU are a special case of this form
    - **If concerned about ReLU caveats, try this**

#### Neural Net Naming Conventions

- Single-layer Neural Network: describes a network with no hidden layers (input directly mapped to output)
    - You can sometimes hear people say that logistic regression or SVMs are simply a special case of single-layer Neural Networks

The forward pass of a **fully-connected layer** corresponds to one matrix multiplication followed by a bias offset and an activation function.

As we increase the size and number of layers in a Neural Network, the **capacity** of the network increases.
- Larger networks will always work better than smaller networks, but their higher model capacity must be appropriately addressed with **stronger regularization** (such as higher weight decay), or they might overfit

This layered architecture enables very efficient evaluation of Neural Networks based on **matrix multiplications interwoven with the application of the activation function**.
