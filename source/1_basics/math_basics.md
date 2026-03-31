## Math Basics


Math is the foundation of machine learning, deep learning, and computer vision.


Example
- Definition of neural entwork layers
- Loss functions to adjust neural network parameters
- Evaluation metrics to assess model performance
- Transformation of images
- 3D and camera geometry


### Calculus

What is calculus?

Calculus is the study of how things change.

- It is used to study the rate of change of functions.
- It is used to study the accumulation of quantities.
- It is used to study the optimization of functions.
- It is used to study the differential equations.


#### Functions

A function is a rule that assigns a unique output to each input.



Example:
- $f(x) = x^2$
- $f(x) = sin(x)$
- $f(x) = exp(x)$

Visualization of a function:


```{figure} ../_static/imgs/math_basics/function.png
:width: 80%
:alt: Function

Visualization of a 2D function and a 3D function.
```

What is *not* a function?


```{figure} ../_static/imgs/math_basics/not_function.png
:width: 80%
:alt: Not a Function

If there are multiple outputs for the same input, it is not a function.
```

#### Continuous functions

Definition: A function is continuous if it is continuous at every point in its domain.


```{figure} ../_static/imgs/math_basics/cont_func.png
:width: 40%
:alt: Not a continous function 

Not a continous function because it has a discontinuity. 
```



#### Derivatives and differentiable functions


Definition: The derivative of a function $f(x)$ is the rate of change of the function at a point $x$.


```{figure} ../_static/imgs/math_basics/derivative.png
:width: 80%
:alt: Derivative measures the rate of change of a function.

Derivative measures the rate of change of a function at a point.
```



Example:

- $f(x) = x^2$
- $f'(x) = ?$


```{figure} ../_static/imgs/math_basics/derivative2.png
:width: 80%
:alt: Derivative example.

Plot of the function and its derivative.
```




#### Partial derivatives

Definition: The partial derivative of a function $f(x, y)$ is the rate of change of the function at a point $(x, y)$ with respect to $x$.


Example:

- $f(x, y) = x^2 + y^2$
- $\frac{\partial f}{\partial x} = ?$
- $\frac{\partial f}{\partial y} = ?$


```{figure} ../_static/imgs/math_basics/partial_d.png
:width: 80%
:alt: Partial derivative.

Computing the partial derivative of a function with two variables.
```





#### Gradients


Definition: The gradient of a function $f(x, y)$ is the vector of its partial derivatives.

```{figure} ../_static/imgs/math_basics/grad1.png
:width: 80%
:alt: gradient.

Gradient.
```

**Gradient is important because we will use it to train neural networks.**

Example:

- $f(x, y) = x^2 + y^2$
- $\nabla f = ?$


Visually, what is the gradient of a function? 

```{figure} ../_static/imgs/math_basics/grad2.png
:width: 80%
:alt: gradient.

Gradient intuition.
```



#### Min and max of functions


Definition: The minimum of a function $f(x)$ is the point where the function is at its lowest value, and the maximum of a function $f(x)$ is the point where the function is at its highest value.


```{figure} ../_static/imgs/math_basics/minmax.png
:width: 80%
:alt: finding the min of a function.

Finding the min of a function is important for machine learning.
```



Why we care about the minimum and maximum of a function?

Because the minimum of a function means a solution to an optimization problem -> a trained neural network model.


Gradient and minimum/maximum:
We will find the minimum of a function by following the gradient and at the minimum, the gradient is 0.



### Linear algebra

Linear algebra is the study of vectors and matrices.


- Vectors are used to represent data points in high-dimensional space.
- Matrices are used to represent linear transformations.
- Operations on vectors and matrices are used to solve linear systems of equations.

```{figure} ../_static/imgs/math_basics/la.png
:width: 80%
:alt: Linear algebra.

Linear algebra is the study of vectors and matrices.
```


#### Vector


A vector is a list of numbers.


- Low dimension
- High dimension


Example:

- $v = [1, 2, 3]$
- $v = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]$



Visually, what is a vector?

```{figure} ../_static/imgs/math_basics/vector.png
:width: 80%
:alt: vector.

Visualization of the vector $[1,-1]$.
```




#### Matrix

Definition: A matrix is a 2D array of numbers.


Example:

- $A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}$


Visually, what is a matrix?

```{figure} ../_static/imgs/math_basics/mat.png
:width: 80%
:alt: matrix.

Matrix can transform a point to a different position.
```


```{figure} ../_static/imgs/math_basics/mat2.png
:width: 80%
:alt: matrix transformation.

Matrix transformation.
```



#### Tensor


Definition: A tensor is a multi-dimensional array of numbers.

```{figure} ../_static/imgs/math_basics/tensor.png
:width: 80%
:alt: tensor.

Tensor is the generalization of vectors and matrices to higher dimensions.
```



Example of a 2D tensor:

$T = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}$


Example of a 3D tensor:

$T = \begin{bmatrix} \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix} & \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix} & \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix} \end{bmatrix}$


Example of a 4D tensor:

$T = \begin{bmatrix} \begin{bmatrix} \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix} & \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix} & \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix} \end{bmatrix} & \begin{bmatrix} \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix} & \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix} & \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix} \end{bmatrix} & \begin{bmatrix} \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix} & \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix} & \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix} \end{bmatrix} \end{bmatrix}$



Tensor is everywhere in deep learning. Tensorflow and PyTorch are both built on top of tensors.

#### Vector operations


Example:
- Vector addition
- Vector subtraction
- Vector norm
- Vector dot product
- Vector cross product

```{figure} ../_static/imgs/math_basics/vector_op.png
:width: 80%
:alt: vector operations.

Vector operations.
```



#### Matrix operations

Example:
- Addition
- Multiplication
- Norm
- Transpose
- Inverse

```{figure} ../_static/imgs/math_basics/mat_op.png
:width: 80%
:alt: matrix operations.

Matrix operations.
```



#### Linear transforms

Definition: A linear transform is a function that maps a vector to a vector.

Example:

- $T(x) = Ax$


Eigenvector is the direction of the linear transform that is stretched by the matrix but not rotated.


```{figure} ../_static/imgs/math_basics/eigen.png
:width: 80%
:alt: eigenvector.

Eigenvector direction.
```



#### Linear basis


Definition: A linear basis is a set of vectors that span a vector space.


Example:

- $B = \{1, x, x^2, x^3\}$


We can represent any vector in a vector space as a linear combination of the basis vectors.

- $v = a_1v_1 + a_2v_2 + \ldots + a_nv_n$


Example:

- $v = 1 \cdot 1 + 2 \cdot x + 3 \cdot x^2 + 4 \cdot x^3$


```{figure} ../_static/imgs/math_basics/basis.png
:width: 80%
:alt: basis.

Any point in the vector space can be represented as a linear combination of the basis vectors.
```



### Probability

Probability is the study of uncertainty and randomness.


- It is used to model uncertainty in data and predictions.
- It is used to quantify the likelihood of events.
- It is used to make decisions under uncertainty.
- It is fundamental to machine learning and deep learning.



#### Random variables


Definition: A random variable is a variable whose value is determined by the outcome of a random event.


Example:
- $X$ = the outcome of rolling a die (can be 1, 2, 3, 4, 5, or 6)
- $Y$ = the height of a randomly selected person
- $Z$ = the pixel intensity value at a random location in an image


Types of random variables:
- Discrete: takes on a countable set of values (e.g., die roll)
- Continuous: takes on any value in an interval (e.g., height, pixel intensity)

---

### Distribution

Definition: A probability distribution describes how the probabilities are distributed over the possible values of a random variable.

```{figure} ../_static/imgs/math_basics/dist.png
:width: 80%
:alt: distribution.

Distribution of a random variable.
```



For a discrete random variable $X$, the probability mass function (PMF) $P(X = x)$ gives the probability that $X$ takes the value $x$.


For a continuous random variable $X$, the probability density function (PDF) $p(x)$ describes the relative likelihood of $X$ taking a value near $x$.


Example: For a fair die, $P(X = 1) = P(X = 2) = \ldots = P(X = 6) = \frac{1}{6}$.


---

### Gaussian

Definition: A Gaussian (or normal) distribution is a continuous probability distribution characterized by its mean $\mu$ and variance $\sigma^2$.

```{figure} ../_static/imgs/math_basics/gaussian.png
:width: 80%
:alt: 1D gaussian.

1D Gaussian distribution. The curve is symmetric and centered at the mean. It is also known as the bell curve.
```



The probability density function of a Gaussian distribution is:

$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$


Example:
- Height of people in a population
- Measurement errors
- Pixel values in natural images (often approximately Gaussian)


Visually, what does a Gaussian distribution look like?


---

### Mix of Gaussian

Definition: A mixture of Gaussians is a probability distribution that is a weighted sum of multiple Gaussian distributions.

```{figure} ../_static/imgs/math_basics/mog.png
:width: 80%
:alt: mixture of gaussians.

Mixture of Gaussians.
```



The probability density function of a mixture of $K$ Gaussians is:

$$p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \sigma_k^2)$$

where $\pi_k$ are the mixing weights (probabilities) that sum to 1.


Example:
- Modeling pixel colors in an image (different objects have different color distributions)
- Clustering data points that form multiple groups
- Modeling complex data distributions that cannot be captured by a single Gaussian


Visually, what does a mixture of Gaussians look like?


---

### Sampling

Definition: Sampling is the process of generating random values from a probability distribution.


Example:
- Sampling from a uniform distribution: generate random numbers between 0 and 1
- Sampling from a Gaussian: generate random values that follow a Gaussian distribution
- Sampling from a dataset: randomly select data points from a training set


Why is sampling important?
- Generate synthetic data
- Monte Carlo methods for estimation
- Stochastic gradient descent uses random sampling of batches
- Data augmentation in computer vision


Also, ChatGPT uses sampling to generate text.

---

### Bayes

Bayes' theorem relates the conditional probability of an event given evidence to the prior probability and likelihood.

```{figure} ../_static/imgs/math_basics/bayes.png
:width: 80%
:alt: bayes theorem.

Bayes theorem: if you see a photo looks like a dragon, it is more likely to be something else because it is rare to see a dragon.
```



Bayes' theorem:

$$P(A | B) = \frac{P(B | A) P(A)}{P(B)}$$

where:
- $P(A | B)$ is the posterior probability
- $P(B | A)$ is the likelihood
- $P(A)$ is the prior probability
- $P(B)$ is the evidence


Example:
- Medical diagnosis: Given symptoms (evidence), what is the probability of a disease (hypothesis)?
- Image classification: Given an image (evidence), what is the probability it belongs to a class (hypothesis)?
- Spam detection: Given email content (evidence), what is the probability it is spam (hypothesis)?


Bayesian inference is fundamental to probabilistic machine learning models.
