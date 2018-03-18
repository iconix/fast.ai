# Glossary

While compiling my own glossary based on this course, I stumbled upon some other nice glossaries:
- [Machine Learning Glossary](https://developers.google.com/machine-learning/glossary/) by Google
- [Deep Learning Glossary](http://www.wildml.com/deep-learning-glossary/) by WildML

## Deep Learning Terminology

- **Activation function**: a non-linearity that takes a single number (e.g., signal strength after the sum of a neuron's inputs) and performs a certain fixed mathematical operation on it (to "squash" the signal between a range of values, for example); in a neuron, we used this to model the _firing rate_, or frequency of a neuron's output spikes.
    - Common functions include (with [Stanford CNN pages'](http://cs231n.github.io/) recommendations following in parentheses): _sigmoid_ ("never use"); _rectified linear unit/ReLu_ ("use this, but be careful with your learning rates and possibly monitor the fraction of 'dead' units in a network"); _leaky ReLu_ ("if concerned about ReLU caveats, try this"); _maxout_ ("if concerned about ReLU caveats, try this"); _tanh_ ("try this, but expect it be worse than ReLU/maxout"); _softmax_ (fast.ai: "good for final output layer").
- **Backpropagation** (backward propagation of errors): the method of analytical (calculus-based) Gradient Descent that uses recursive application of the chain rule
- **Batch normalization**: **TODO**
- **Convolution**: the mathematical operation of 1) correlating a _filter_ to an area of an image, 2) multiplying each overlapping value between the filter and image area (an element-wise operation), and 3) summing up these products. In CNNs, this sum then replaces the center pixel of the image area correlated. This operation is performed repeatedly for each area of the entire image, resulting in the construction of a new image, each pixel having been replaced. The new image represents what was "seen" by a particular filter, and this process can be used to extract features of the original image; see also _filter (kernel)_.
- **Convolutional layer**: a layer that uses sets of _filters_ as weights, and transforms its input using _convolutions_ (instead of matrix multiplication). The output layer is simply k representations of the original image, where k is the number of filters; see also _filter, convolution_.
- **Convolutional Neural Networks (CNNs)**: allow computers to "see".
- **CUDA**: an Nvidia GPU programming environment; it contains _cuDNN_, the CUDA Deep Neural Network library.
- **Data augmentation**: **TODO**
- **Deep learning**: **TODO**
- **Deep neural network (DNN)**: a neural network with more than 1 hidden layer
- **Dropout**: **TODO**
- **Ensembling**: building multiple versions of your model and adding them together.
- [_The Master Algorithm_](https://iconix.github.io/notes/2017/09/23/master-algorithm.html) (pp. 237-8) called this _metalearning_ and then briefly discussed the types _stacking_ and _boosting_
- **Epoch**: one full pass through the training data.
- **Exploding gradients**: when the power of imperfect scaling of a weight matrix is multiplied again and again and again; LSTMs and GRUs help mitigate this; see also _Long-Term Short-Term Memory (LSTM), Gated Recurrent Unit (GRU)_
- **Filter (Kernel)**: a (conventionally) 3x3 matrix that is applied to an image via the process of _convolution_. They can be used to identify particular visual "elements" of an image (see <http://setosa.io/ev/image-kernels/> for more). In deep learning, these filters are randomly generated and then optimized with gradient descent.
- **Finetuning**: the process of building a new model from a pre-trained one that can already solve a similar problem. This is done by incorporating prior knowledge encoded in the lower-level filters of the pre-trained model, while re-mapping the higher-level filters to whatever classification problem you're currently attempting to solve.
- **Fitting**: the process of _optimizing_ the weights of a neural network such that the input vectors that pass through the network produce output vectors that are as close as possible to the true output vectors, across multiple labeled input vectors; see also _optimization_.
- **Fully-connected (Dense) layer**: the most common layer type (for regular neural networks), in which neurons between two adjacent layers are fully pairwise connected, but neurons within a single layer share no connections
- **Gated Recurrent Unit (GRU)**: **TODO**; see also _exploding gradient_
- **Global average pooling**: **TODO**
- **Gradient**: a vector of partial derivatives for each dimension in the input space.
    - Note: Gradient is a generalization of slope (derivatives) for functions that don't take a single number but a vector of numbers; and when the functions take a vector of numbers, we call the derivatives _partial derivatives_.
- **Gradient Descent**: the procedure of repeatedly evaluating the gradient and then performing a parameter update in order to minimize the loss function; this is currently by far, the most common and established way of optimizing Neural Network loss functions.
    - **_Stochastic (SGD)_**: when you compute the gradient over batches from the training data, instead of the entire training set.
        - Note: Traditionally, this definition actually referred to "mini-batch gradient descent" or "batch gradient descent", while SGD referred to computing the gradient over a single example. This is no longer the common practice.
- **Hidden layer (Activation layer?)**: **TODO**
- **Keras**: a flexible, easy to use deep learning library that translates deep learning architecture (neural nets) into Theano or Tensorflow code. Keras reads groups of images and labels in batches, using a fixed directory structure, where images from each category for training must be placed in a separate folder.
- **L1/L2 Regularization**: **TODO**
- **Learning rate**: the rate at which we update parameters through gradient descent. This can be thought of as the size of the "step" we take down the gradient in the hypersurface defined by our loss function.
    - **Learning rate annealing**: gradually decreasing the learning rate
    - **Dynamic learning rate**: an approach to dynamically adjust the learning rate to improve optimization performance, using information gained from the optimization process itself. The alternative is a _constant learning rate_, but this can result in longer-than-necessary training times, _unstable optimization_, or getting stuck in unsatisfactory local minima or _saddle points_; see also _unstable optimization, saddle point_.
        - Common methods: Adagrad, RMSprop, Adam (RMSprop + momentum)
- **Logarithmic (Log) Loss**: **TODO**
- **Long-Term Short-Term Memory (LSTM)**: an recurrent architecture that replaces the standard RNN loop with a loop that contains a neural network. The neural network decides how much of the state matrix to keep and lose, helping to learn how to avoid exploding gradients; see also _exploding gradient_
- **Loss (cost) function**: a function that measures the quality of a particular set of parameters based on how well the induced scores agreed with the ground truth labels in the training data. Examples: Softmax, SVM
- **Max pooling layer**: a layer that reduces the dimensionality (resolution) of images by reducing the number of pixels in the image. It does so by replacing an entire NxN area of the input image with the maximum pixel value in that area, forcing our neural network to look at larger areas of the image at a time.
    - In order to make up for the information lost upon pooling in a CNN, we'll typically increase the number of filters in subsequent convolutional layers.
- **Mini-batch (batch)**: a subset of the training data used when training or predicting, in order to speed up training and to avoid running out of memory.
- **Momentum**: a technique intended to push optimization through challenging areas like _saddle points_. It works by works by adding a new term (that can be thought of as the "average of the previous gradients") to the update function, which helps force any kind of zig-zagging through a saddle point to move along the average direction we're zig-zagging towards; see also _saddle point, hyperparameter_.
- **Neural network architecture**: the way in which you stack distinct layers (fully-connected, convolutional, pooling, padding) and apply activation functions to each layer (conventionally with a difference in activations between hidden vs output layers) to construct a model for deep learning.
- **One-Hot Encoding**: an approach to encoding categorical variables, where an array contains just a single 1 in the position corresponding to the category; very common in deep learning.
- **Optimization**: a process of finding the set of parameters `W` that minimize the loss function. example: gradient descent
    - **Unstable optimization**: if the _learning rate_ is too big (e.g., step size is too large), we might step over potential optimums or even bounce out of an optimum.
- **Overfitting**: occurs when a model with high capacity fits the noise or specificities in the training data, instead of the (assumed) underlying relationship. More specifically, this occurs when a model is using too many parameters/dimensions and has been trained for so long that it can no longer generalize to other datasets.
- **Pseudo-labeling**: a _semi-supervised learning_ approach where we take a model trained on labelled data that is outputting reasonably good validation results. We then use this model to make predictions on all of our unlabelled data, and then use those predictions as labels themselves.
- **Rectified linear unit (ReLu)**: an activation function often applied to the hidden layers of a neural network. It is simply the function `max(0, x)`.
- **ResNet(50)**: **TODO**
- **Saddle point**: an area (a "shallow valley") where the gradient of the loss function often becomes very small in one or more axes, but there is no minima present; often present in the hypersurface defined by the loss function of a neural network.
- **Score function**: a parameterized function that maps input (e.g., raw image pixels) to scores (e.g., class). Examples: Linear function, Neural Networks, Convolutional Neural Networks.
- **Softmax**: an activation function often applied to the final output layer of a neural network because we'd like to 1) transform each output as a portion of a total sum, so that it can be interpreted as a probability, and 2) mimic one-hot encoding by pushing the highest output as close to 1 as possible and our lower outputs as close to zero (which is achieved by a non-linear exponential function). It is the function `e^x/sum(e^x)`.
- **Tensor**: refers to a matrix with more than 2 dimensions; e.g., 28x28x12. This '3d tensor' could be produced (for example) by passing a 28x28 pixel image through a convolutional layer with 12 filters, since each filter will produce a new image.
    - In a 3d tensor, the third dimension is often referred to as the _channel_ or _depth_
- **Theano/Tensorflow**: takes Python code and turns it into compiled GPU code. _Theano_ is more mature, easier to use, and good for single GPU use cases, while _Tensorflow_ was created by Google and thus thrives in multi-GPU scenarios.
- **Transfer learning**: using a pre-trained network and fine-tuning it to accomplish a different yet similar task
- **Underfitting**: occurs when a model lacks the complexity to accurately capture the complexity inherent in the problem you're trying to solve.
- **Vgg16**: 2014 ImageNet competition winner; a very simple model to create, understand, and finetune.
- **Zero-padding layer**: a layer that adds extra borders of zero pixels around the image prior to passing through a filter so that the output shape from the filter is the same as the input shape. This is needed because _convolving_ each pixel of an image necessarily operates on the premise that each pixel has 8 surrounding pixels (if the filter is conventionally 3x3); see also _convolution_.

## Non-Deep Terminology
_Non-deep terms that come up in deep learning._

- **Bias**: **TODO**
- **Boosting**: **TODO**
- **Collaborative filtering**: a recommendation system which gives results that match the preferences of users that are similar to you; contrast with an approach based on metadata.
- **Data leakage**: occurs when something about the target you're trying to predict is encoded in the things you're predicting with but that information will not be available/helpful in practice
- **Dimensionality reduction**: **TODO**
    - **Principal Component Analysis (PCA)**: **TODO**
    - **t-Distributed Stochastic Neighbor Embedding (t-SNE)**: **TODO**
- **Embedding**: a numeric vector whose floats describe the latent factors of its source; see also _latent factor_.
- **Hyperparameter**: a parameter used in a model that is not modified through training; rather it is chosen a priori and modified by the user to optimize the desired outcome.
- **Latent factor**: a factor influencing an outcome, even though we don't know what they are
- **Literate Programming**: documenting what you are doing in a deep way as you code. Like a lab notebook, as you do experiments, data scientists keep track of what worked and what didn't.
- **Residual**: in statistics, difference between the thing you're trying to predict and your actuals; key innovation of the ResNet architecture
- **Semi-supervised learning**: using unlabelled data (e.g., test data) to help us gain more information and understanding of the population structure in general.
- **Sentiment analysis**: given text, predict the sentiment expressed within it (i.e. negative, neutral, positive)
- **Universal approximation machines**: **TODO**
