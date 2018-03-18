# Lesson 0: Why Deep Learning; Intro to Convolutions

[Lesson page](http://course17.fast.ai/lessons/lesson0.html) | [Video](https://www.youtube.com/watch?v=ACU-T9L4_lI)

Deep learning = **Linear algebra + Optimization**

Deep learning is:
- a way of doing machine learning.
- an algorithm with three characteristics:
    - "infinitely flexible function" - inspired by the way the human brain works (neural networks as _universal approximation machines_)
    - "all-purpose parameter fitting" - through _gradient descent_ and _backward propagation of errors_ (via the _chain rule_)
    - "fast and scalable" - the most recent development; cloud computing + **GPUs**, which are optimized for _matrix operations_ that are computationally expensive

**The three characteristics of a deep learning algorithm**
![The three characteristics of a deep learning algorithm](https://github.com/iconix/fast.ai/raw/master/help/assets/dl_characteristics.png)

Arthur Samuel: father of ML; coined the term "machine learning" in 1959
- Programmed a computer to beat him in checkers

Convolutional Neural Networks (CNNs) = **Convolutions + Optimization**
- Convolutions? a subset of linear algebra
- Take a "filter", "correlate" it to an image, rotate it by 90 degrees; repeat until you've rotated the filter in all possible directions
    - Filter: 3x3 matrices
    - Correlate: multiply the
     filter by each area
    - This is _feature engineering_
     in deep learning - we're trying to find a fingerprint
        - But instead of carefully selecting N filters, we start with N random filters and then make them better and better (i.e., find the optimal set of filters)

Deep learning allows us to learn things with **unstructured** data.

Current deep learning **drawbacks**: code required, manual parameter tuning, lack of standardized error bars.

**Manual feature engineering is pre-deep learning**
> "Adding metadata or doing hand engineering of the features is not at all useful… **just take the raw data** (images, audio signals)… the important thing is what you correlate it against (the dependent variable, the truth data)."

Useful prerequisites for deep learning:
- Familiarity with [matrix multiplication](https://www.khanacademy.org/math/linear-algebra/matrix-transformations), basic differentiation, and the [chain rule](https://www.khanacademy.org/math/calculus-home/taking-derivatives-calc/chain-rule-calc/v/chain-rule-introduction)
- Python [list comprehensions](http://treyhunner.com/2015/12/python-list-comprehensions-now-in-color/)
- Python [data science libraries](http://wiki.fast.ai/index.php/Python_libraries) (numpy, scipy, scikit-learn, pandas, Jupyter notebook, matplotlib)

    ...and you **do _not_** actually need [a PhD + all of this](https://news.ycombinator.com/item?id=12901536) to get started!

Teaching philosophy: **Good education is not overly complicated**.
