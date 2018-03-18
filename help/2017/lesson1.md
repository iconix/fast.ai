# Lesson 1: How to approach this course; Deep learning tips

[Lesson page](http://course17.fast.ai/lessons/lesson1.html) | [Wiki notes](http://wiki.fast.ai/index.php/Lesson_1_Notes) | [Video](https://www.youtube.com/watch?v=Th_ckFbc6bI)

**How to spend the recommended 10 hours/week on this course**:
1. Watch through first time to get high-level idea (2-2.5 hours per video)
2. Read the wiki, try the notebooks (5 hours)
    - Wiki includes a timeline with video links per lesson
    - "Try the notebooks" means: read through the class notebook and then recreate the notebook from scratch (see: [How to use the Provided
      Notebooks](http://wiki.fast.ai/index.php/How_to_use_the_Provided_Notebooks))
    - Participate in the forums if
      you get stuck
        - Google Brain rule: "If you have a problem, you first try to fix it yourself for half an hour. After a half an hour, you have to go ask somebody."
    - I'm using the [OneNote Web Clipper](https://www.onenote.com/clipper) to pull wiki notes into my OneNote notebook - then I can highlight and do some light augmentation of the notes.
3. Watch the lesson again to get
     more detail (2-2.5 hours)
    - I take the majority of my notes during this watch through

**GPUs** are good at linear algebra and matrix
operations, because graphics and video games need these things. So does deep learning. Fortunately, the gaming industry has made GPUs much cheaper. 

**How to split data for ML**:
- Training set (64%): fit parameters
- Validation set (16% || 20% of training set): finetune parameters of a pre-trained model
- Test set (20%): blind test the model
- Sample: data to develop against for faster iterating

    _Percentages based on [this StackOverflow](https://stackoverflow.com/a/13623707)_

**Always explore the data a pre-trained model was
trained on**, so you can understand its limits and bias.

You can **replicate** an ImageNet winner's result (e.g., Vgg16, 2014) with just:

1. The source code (provides neural network architecture)
2. Learned parameters/weights

**Deep learning tech stack**

    Vgg16

    ^

    |

    Keras (deep learning architecture -> Theano/Tensorflow/CNTK Code)

    ^

    |

    Theano (Python code -> compiled GPU code)

    ^

    |

    CUDA programming environment; cuDNN (CUDA Deep Neural Network library)

**Tensorflow vs Theano (vs CNTK)**

- Tensorflow is from Google and is particularly good on running things on multiple GPUs
- Theano is more mature, easier to use, and good for single GPU
    - [No longer being actively developed as of September 2017](https://groups.google.com/forum/#!topic/theano-users/7Poq8BZutbY)
- Easy to switch between the
     two, via config (keras.json)
- CNTK support for Keras is [in beta as of August 2017](https://jamesmccaffrey.wordpress.com/2017/08/29/using-keras-with-cntk-on-mnist/)
