# Lesson 3: new Keras layers; final activation functions; underfitting and overfitting; CNN rules of thumb

[Lesson page](http://course17.fast.ai/lessons/lesson3.html) | [Wiki notes](http://wiki.fast.ai/index.php/Lesson_3_Notes) | [Video](https://www.youtube.com/watch?v=6kwQEBMandw)

## Deep learning automatically learns the most useful features

A key advantage of neural networks as opposed to traditional machine learning techniques is that we can **avoid manual feature engineering** of our raw input, i.e. constructing pre-determined features from our raw data that we think is important.
- Traditional machine learning might have a human pick a set of N filters as features of a machine learning algorithm. **In deep learning, we allow our networks to "learn" what N filters are best from the raw data, given labelled output.**
- This is incredibly powerful because we **avoid discarding raw data that could be useful**, and it **removes the onerous task of manually designing features**.

## New Keras layer definitions

- **Conv(2D)**: converts the input of 3 colors/channels in RBG to N (e.g., 64) channels - one channel per filter
- **MaxPooling(2D)**: gradually simplifies pixel blocks in an image to a single pixel that contains the max pixel value - larger and larger areas with smaller and smaller images (# of pixels)
    - When we reduce (half) resolution, we need to maintain the amount of information content. We do this by hiking up (doubling) the number of filters in the step following MaxPooling
        - This is why the neural architecture increases the number of filters as we add convolutional layers
    - A convolution is position invariant: it finds a pattern regardless of whereabouts in the image it is. But we need to identify position to some extent, so **max pooling helps encourage the network to think about larger and larger swaths of the image and therefore, positional relationships**
- **ZeroPadding(2D)**: sticks zeros around the outside of an image - this is also known as a _same convolution_ (versus a _valid convolution_)
- **Flatten**: a Keras thing where they say, "don't think of the layers anymore as being X by Y channel matrices, think of them as being a single vector" so it can concatenate all the layers together

## Softmax/sigmoid as the final activation function

While _ReLu_ is by far the most common activation function for hidden layers of a neural network, **_Softmax_ is often used for the final layer** of a neural network.
- `softmax = exp(x)/sum(exp(x))`
- Final layer is trying to map to a one-hot encoded output
- We need two things that softmax provides
	1. We want to interpret the final layer as probabilities that sum to 1
	2. One probability should be higher than all the rest to represent the classification label. Intuitively, the exponential does this.
- [This isn't clarified until [Lesson 4](https://github.com/iconix/fast.ai/blob/master/help/2017/lesson4.md), but...] while _softmax_ is used for **categorical classification** (categorical_crossentropy, one-hot encoded output), **_sigmoid_ is the final layer for binary classification (binary_crossentropy)**.

## Good vs Bad Architecture

Even though all neural networks can theoretically learn anything ("infinitely flexible function"), **choice comes down to optimizing the speed of learning**.

## Avoiding underfitting and overfitting

**Underfitting**: using a model that isn't complex enough, not enough parameters
- Training error <<< Validation error

**Overfitting**: using a model that is too complex to generalize
- Training set accuracy >>> Validation/Test set accuracy

### Approaches to reducing overfitting

1. Add more data
    - And make sure it is normalized - both the inputs and the activation layers (via Batch Normalization)
2. Use data augmentation
3. Use architectures that generalize well
4. Add regularization (dropout, L2/L1 regularization)
5. Reduce architecture complexity
    - Hard to do if you're finetuning (how do you know what to remove?). Steps 1-4 are the main approaches.

### Rules of thumb from this lesson

1.  CNNs: 3x3 filters seem to work best.
2.  No standard for how many layers of 3x3 filters to use.
3.  Each time you apply MaxPooling, double the number of filters.
4.  The more different your problem is from the pre-trained model, the more finetuning you'll need (more later layers to retrain from scratch).
    - How far back? Relies on intuition and experimentation
    - One point of intuition: Convolutional layers are about spatial relationships and recognizing how things in space relate to each other
5. Training convolutional layers takes up computational time; training dense layers takes up memory
6. Standard to applying dropout to convolutional layers: you don't want to dropout too much in the early layers. Dropout(0.1) for 1st conv layer, Dropout(0.2) for 2nd conv layer, Dropout(0.3) for 3rd conv layer, etc. until you reach your first dense layer, where you can start dropping out a constant rate.
7. Always use: data augmentation, batch normalization, and input normalization - not just when you see overfitting.
8. **Start with overfitting - all good architectures should be capable of overfitting.**
