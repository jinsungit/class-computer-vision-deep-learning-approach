# Self-check: Keyword concepts

---

## Math basics

### 1. What is *not* a function

Which situation breaks the usual definition of a function from a set $X$ to a set $Y$?

**A.** Each element of $X$ is assigned exactly one element of $Y$.  
**B.** Two different inputs in $X$ map to the same output in $Y$.  
**C.** One input in $X$ is assigned two different outputs in $Y$.  
**D.** Possible values of $X$ and $Y$ are discrete.

<details>
<summary>Show answer</summary>

**C.** A function assigns **exactly one** output to each input; two outputs for the same input is not a function.

</details>

### 2. What is gradient

The gradient $\nabla f$ of a function $f$ at a point (when it exists) points in the direction of:

**A.** Steepest **decrease** of $f$.  
**B.** Steepest **increase** of $f$.  
**C.** Zero change no matter how you move.  
**D.** The global minimum.

<details>
<summary>Show answer</summary>

**B.** The gradient points in the direction of **steepest ascent** (steepest increase).

</details>

### 3. Why we care about gradient

Why do we use gradients when training models?

**A.** They describe how small parameter changes can improve the model.  
**B.** They describe how good your model is.  
**C.** They remove the need for any training data.  
**D.** They only apply to linear models.

<details>
<summary>Show answer</summary>

**A.** Gradients link **parameter changes** to **loss changes**, enabling iterative optimization (e.g., gradient descent).

</details>

### 4. Min / max of functions and gradient

For a smooth function $f$, what is typically true at a local minimum where the gradient exists?

**A.** $\nabla f = 0$.  
**B.** $\nabla f = 1$.  
**C.** $\nabla f$ is undefined.  
**D.** $\nabla f$ is any non-zero constant.

<details>
<summary>Show answer</summary>

**A.** At an interior local extremum (min or max) of a smooth function, the gradient is typically **zero**.

</details>

### 5. Vector

In this course, a **vector** is best thought of as:

**A.** A list (or array) of numbers.  
**B.** A curve.  
**C.** A single random number.  
**D.** A transformation.

<details>
<summary>Show answer</summary>

**A.** Vectors are usually **ordered collections of numbers** (coordinates in $\mathbb{R}^n$).

</details>

### 6. Matrix

A **matrix** is naturally:

**A.** A 2D array of numbers, with rows and columns.  
**B.** The same dimension as a vector, just written differently.  
**C.** A single number.  
**D.** Only used for images, never for text.

<details>
<summary>Show answer</summary>

**A.** A matrix is a **rectangular table** of numbers.

</details>

### 7. Tensor

In deep learning usage, a **tensor** most often means:

**A.** A multi-dimensional array generalizing scalars, vectors, and matrices.  
**B.** The input data.  
**C.** The loss value.  
**D.** A Python list of strings.

<details>
<summary>Show answer</summary>

**A.** Tensors generalize to **many axes** (e.g., batch, channels, height, width).

</details>

### 8. Vector ops: addition, norm, and dot product

The **dot product** of two real vectors $\mathbf{u}$ and $\mathbf{v}$ is:

**A.** A **scalar** measuring the similarity between two vectors.  
**B.** A vector.  
**C.** Always positive.  
**D.** Always zero if $\mathbf{u} \neq \mathbf{v}$.

<details>
<summary>Show answer</summary>

**A.** $\mathbf{u}\cdot\mathbf{v} = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$ — a **scalar**.

</details>

### 9. What is a linear transform

Which choice best matches a **linear** map from $\mathbb{R}^n$ to $\mathbb{R}^m$ (no extra bias term)?

When we want to represent a linear transformation between two vectors, we use:

**A.** A data structure.  
**B.** A scalar.  
**C.** A matrix.  
**D.** A vector.

<details>
<summary>Show answer</summary>

**C.** Linear transformations correspond to **matrix multiplication** (plus a bias in code is an **affine** map).

</details>

### 10. What is linear basis

If we have a linear basis in a vector space, we can represent any vector as a linear combination of those basis vectors.

**A.** True.  
**B.** False.  

<details>
<summary>Show answer</summary>

**A.** Bases provide a coordinate system via **unique** linear combinations.

</details>

### 11. Types of random variables: discrete and continuous

A **discrete** random variable takes values that are:

**A.** Only integers (always) — no exceptions.  
**B.** Not possible to list all values.  
**C.** From a **countable** set (you can list outcomes, e.g. coin flip, class label).  
**D.** Always Gaussian.

<details>
<summary>Show answer</summary>

**C.** Discrete = outcomes from a **countable** set; continuous often means an interval of $\mathbb{R}$.

</details>

### 12. Gaussian distribution: mean and variance

A univariate Gaussian $N(\mu, \sigma^2)$ is mainly specified by:

**A.** Only $\mu$.  
**B.** Only $\sigma^2$.  
**C.** Mean $\mu$ and variance $\sigma^2$.  
**D.** The maximum of the density only.

<details>
<summary>Show answer</summary>

**C.** **Mean** (center) and **variance** (spread) define the standard 1D Gaussian.

</details>

### 13. Bayes theorem and image classification example

Consider the following scenario: You are given a *photograph* and you need to classify it into either a dragon or a horse. We observe that the image looks both like a dragon and a horse, what should you do with the Bayes theorem-type reasoning? 


**A.** I will guess randomly.  
**B.** Must be a dragon because it is more rare (low prior).  
**C.** Cannot be a dragon because it is more rare (low prior).  
**D.** Neither of the above.

<details>
<summary>Show answer</summary>

**C.** Posterior is proportional to **prior** $\times$ **likelihood**; that is the Bayes pattern for classification reasoning.

</details>

---

## Machine learning basics

### 14. What is machine learning: predicting the future based on the past

In the usual ML story, models:

**A.** Learn patterns from past examples to predict on **new** data.  
**B.** Never use data, only closed-form physics formulas.  
**C.** Memorize one training point and ignore the rest.  
**D.** Must be hand-coded `if` statements only.

<details>
<summary>Show answer</summary>

**A.** ML uses **past data** to generalize to **future / unseen** instances.

</details>

### 15. Practice formulation of learning

In practice, we don't have full access to the entire data distribution, we have a finite dataset. Which of the following is correct?

**A.** The model we learned from the finite dataset is not useful for the real data distribution.  
**B.** Since we don't have full access to the entire data distribution, we can't predict the future using the finite dataset.  
**C.** We can estimate the performance of the model by evaluating it on the dataset at hand.  
**D.** Neither of the above.

<details>
<summary>Show answer</summary>

**C.** We separate the data into training, validation, and test sets to estimate the performance of the model learned from the finite dataset on the real data distribution.

</details>

### 16. What is supervised learning

**Supervised learning** means training with:

**A.** Exact input–output pairs $(x, y)$ where $y$ is the supervision signal.  
**B.** Only $x$ with no labels.  
**C.** Random rewards from an environment (for example, a thumb-up from the user by clicking a button).  
**D.** No data.

<details>
<summary>Show answer</summary>

**A.** Supervision = provided **labels** or **targets** paired with inputs.

</details>

### 17. Classification

**Classification** predicts:

**A.** A real number like price or temperature.  
**B.** A **category** (class) or class probabilities.  
**C.** A set.  
**D.** A cluster of data points with no class names.

<details>
<summary>Show answer</summary>

**B.** Outputs are **discrete classes** (possibly as a distribution over classes).

</details>

### 18. Regression

**Regression** typically predicts:

**A.** Is this image a cat or a dog?  
**B.** A **numerical** value.  
**C.** Only “spam vs not spam.”  
**D.** A permutation of labels.

<details>
<summary>Show answer</summary>

**B.** Regression targets are usually **continuous** quantities.

</details>

### 19. Classification vs regression: which one to choose

You usually pick **classification** over regression NOT because:

**A.** Classification is more well-defined and well-studied.  
**B.** The target values are from a set of possible values.  
**C.** You care more about something being present or not.  
**D.** You have to use a big model.

<details>
<summary>Show answer</summary>

**D.** Whether to use classification or regression is not related to the size of the model.

</details>

### 20. What is unsupervised learning

**Unsupervised learning** works with data:

**A.** Without provided labels, to find structure (i.e., clusters).  
**B.** Only as $(x,y)$ pairs with labels for every point.  
**C.** With numerical labels.  
**D.** With discrete labels.

<details>
<summary>Show answer</summary>

**A.** **No labels** — discover structure from $x$ alone (or weak signals).

</details>

### 21. Idea of K-means clustering

**K-means** will find:

**A.** K-1 clusters.  
**B.** K clusters.  
**C.** K+1 clusters.  
**D.** Unknown number of clusters.

<details>
<summary>Show answer</summary>

**B.** K-means will find K clusters, but it might not be the optimal number of clusters.

</details>

### 22. Idea of dimensionality reduction

**Dimensionality reduction** seeks to:

**A.** Represent data with **fewer** dimensions while keeping useful information.  
**B.** Duplicate features.  
**C.** Expand data.  
**D.** Increase noise in data.

<details>
<summary>Show answer</summary>

**A.** Lower-dimensional representations for visualization, speed, or denoising.

</details>

### 23. Model is a function mapping between $x$ and $y$

In supervised learning, a **model** is:

**A.** A map from inputs $x$ to predictions $\hat{y}$ (approximating the true $y$).  
**B.** Some CSV file on the hard drive.  
**C.** The optimizer without parameters.  
**D.** The evaluation metric alone.

<details>
<summary>Show answer</summary>

**A.** A learned **function** $x \mapsto \hat{y}$.

</details>

### 24. Linear model is a model with linear function $Ax$

Here “linear model” means predictions are:

**A.** A **linear** function of the features (e.g., $\hat{y} = A x$ in vector form, maybe plus bias).  
**B.** A deep ReLU network with 100 layers.  
**C.** A quadratic function $f(x) = ax^2 + bx + c$.  
**D.** Random guesses.

<details>
<summary>Show answer</summary>

**A.** **Linear** in the feature vector (not necessarily linear in underlying signals if features are fixed).

</details>

### 25. Why split data into train / val / test?

We split data mainly to:

**A.** Meet a legal requirement (e.g., for data privacy).  
**B.** Train models, tune choices on validation data, and **honestly** estimate generalization on held-out test data.  
**C.** Delete half the samples on purpose.  
**D.** Speed up forward pass.

<details>
<summary>Show answer</summary>

**B.** Separation prevents **cheating** your own evaluation and reduces overtuning.

</details>

### 26. Idea of cross-validation

**Cross-validation** helps:

**A.** Estimate performance more reliably by repeating train/eval on different partitions.  
**B.** Remove bias from the model.  
**C.** Guarantee global optimality.  
**D.** Measure the quality of the dataset.

<details>
<summary>Show answer</summary>

**A.** Averages over splits to reduce **variance** in the performance estimate.

</details>

### 27. What is a metric

An evaluation **metric** is:

**A.** A quantity summarizing how good predictions are (accuracy, mAP, RMSE, …).  
**B.** Not very practical to use.  
**C.** The batch size.  
**D.** The name of the optimizer.

<details>
<summary>Show answer</summary>

**A.** Metrics **measure** quality on a dataset or benchmark.

</details>

### 28. What is overfitting

**Overfitting** usually means:

**A.** The model fits training data **too** closely and may perform poorly on new data.  
**B.** The model is too simple.  
**C.** We trained the model for too few steps.  
**D.** The dataset is too difficult.

<details>
<summary>Show answer</summary>

**A.** **High train, low test** — memorization / high variance.

</details>

---

## Deep learning basics

### 29. AI = deep learning?

Which statement is most accurate?

**A.** Deep learning is the only topic in all of AI.  
**B.** Deep learning is **one** family of methods inside the broader field of AI.  
**C.** AI and deep learning mean unrelated things.  
**D.** Deep learning never uses data.

<details>
<summary>Show answer</summary>

**B.** **AI** is broad; **deep learning** is a prominent **subset** (neural networks with depth and representation learning).

</details>

### 30. Neuron vs neural networks

A single **neuron** in a neural network is:

**A.** One small computational unit; a **neural network** combines many such units in layers.  
**B.** The entire trained model by definition.  
**C.** The same as the full dataset.  
**D.** Only a loss value.

<details>
<summary>Show answer</summary>

**A.** **Compose** many neurons → network.

</details>

### 31. What is an activation function? ReLU

The **ReLU** activation is:

**A.** $\max(0, z)$.  
**B.** $1/(1+e^{-z})$.  
**C.** $\tanh(z)$.  
**D.** $|z|^2$.

<details>
<summary>Show answer</summary>

**A.** ReLU “clips” negatives to zero.

</details>

### 32. A single (linear) layer

One **linear layer** typically computes:

**A.** $\mathbf{y} = W\mathbf{x} + \mathbf{b}$ (matrix multiply plus bias).  
**B.** Spatial max pooling.  
**C.** Softmax.  
**D.** A decision tree split.

<details>
<summary>Show answer</summary>

**A.** Linear map = **weight matrix** and optional **bias**.

</details>

### 33. Multi-layer → learn complex mapping

**Stacking** nonlinear layers allows a network to:

**A.** Represent only linear functions overall.  
**B.** Build **complex** input–output relationships than one linear layer alone.  
**C.** Remove all parameters.  
**D.** Ignore the input.

<details>
<summary>Show answer</summary>

**B.** Composition of linear + nonlinear layers yields **nonlinear** functions, allowing the network to approximate complex relationships.

</details>

### 34. Main idea of training: data, forward, backward, update

Training usually repeats:

**A.** Forward pass → loss → backward pass → parameter update.  
**B.** Forward pass only, forever.  
**C.** Random weight changes with no loss.  
**D.** Sorting the dataset.

<details>
<summary>Show answer</summary>

**A.** Classic **supervised** neural net training loop.

</details>

### 35. Backpropagation in the backward pass

**Backpropagation** computes:

**A.** Gradients of the loss with respect to model parameters.  
**B.** Gradients of the data with respect to the loss.  
**C.** Metrics for the trained model.  
**D.** Another round of forward pass.

<details>
<summary>Show answer</summary>

**A.** Backpropagation is an efficient **gradient** computation method for multiple layers of networks.

</details>

### 36. The concept of batch in deep learning

A **mini-batch** is:

**A.** The whole training set every step.  
**B.** A small **subset** of examples used for one gradient step.  
**C.** The validation set only.  
**D.** One convolution kernel.

<details>
<summary>Show answer</summary>

**B.** Batching balances **noise** vs **memory** vs **speed**.

</details>

### 37. What is a loss function

A **loss** for training is:

**A.** A scalar objective we try to **minimize** so the model fits data.  
**B.** Only used after deployment, never in training.  
**C.** Always identical to accuracy.  
**D.** A list of layer names.

<details>
<summary>Show answer</summary>

**A.** Loss drives **optimization** (when differentiable).

</details>

### 38. Loss vs metric

Which is **true**?

**A.** Loss is what training optimizes; metrics summarize performance for analysis and reporting.  
**B.** Loss and metric must always be the same number.  
**C.** Metrics are never computed on validation data.  
**D.** Loss cannot depend on model outputs.

<details>
<summary>Show answer</summary>

**A.** **Loss** for **learning**; **metrics** for **evaluation** and comparison.

</details>

### 39. Neural nets

In this class, “**neural network**” usually means:

**A.** A parameterized composition of layers trained with data and gradients.  
**B.** Only a biological brain simulation with no math.  
**C.** Some spreadsheet data.  
**D.** Computer Vision.

<details>
<summary>Show answer</summary>

**A.** **Learned** layered function, typically **differentiable**.

</details>

### 40. Feedforward networks have no feedback

A **feedforward** network:

**A.** Passes information from input toward output **without** recurrent cycles and lateral connections in the graph.  
**B.** Is also called a recurrent network.  
**C.** Has feedback loops when we run the forward pass without running the backward pass.  
**D.** Cannot have more than one layer.

<details>
<summary>Show answer</summary>

**A.** No **recurrent connections** along the main forward path (contrast with RNNs). Does not have feedback loops when we just run the forward pass.

</details>

### 41. Multi-layer Perceptron (MLP)

An **MLP** is typically:

**A.** Alternating **linear** layers and **nonlinear activations** (sometimes dropout etc.).  
**B.** Only one convolution.  
**C.** Only k-nearest neighbors.  
**D.** A purely linear layer with no activations.

<details>
<summary>Show answer</summary>

**A.** **Stacked** linear + nonlinear blocks (fully connected).

</details>

### 42. Idea of dropout layer

**Dropout** during training:

**A.** Randomly zeros some units to reduce heavy dependence on a few units.  
**B.** Deletes the dataset from disk.  
**C.** Forces all activations to 1.  
**D.** Removes backpropagation.

<details>
<summary>Show answer</summary>

**A.** **Random** neuron masking as **regularization**.

</details>

### 43. Idea of convolution in 2D

2D convolution slides a **small filter** over an image so **each output location** combines information from:

**A.** All pixels in the image.  
**B.** A **local neighborhood**.  
**C.** Only the top row of pixels.  
**D.** Random pixels ignoring position.

<details>
<summary>Show answer</summary>

**B.** **Local receptive field** + **weight sharing** across positions.

</details>

### 44. 2D convolution example

If we have a 3x3 input image and a 3x3 filter, and we run 2D convolution once by aligning the filter at the center of the input image, the output size will be:

**A.** 2x2.  
**B.** 1x1.  
**C.** 3x3.  
**D.** 4x4.

<details>
<summary>Show answer</summary>

**B.** 1x1.

</details>

### 45. Main idea of CNNs

**CNNs** exploit:

**A.** Multiple small filters to extract useful information.  
**B.** Fully connecting every pixel to every other at all layers.  
**C.** Pixel sorting.  
**D.** Unsupervised clustering.

<details>
<summary>Show answer</summary>

**A.** **Spatial structure** + **parameter sharing** → efficient vision models.

</details>

### 46. Convolutional layers: filters = output channels

If a conv layer learns **K** distinct filters for the given input depth, you typically get:

**A.** **K** output feature maps (channels).  
**B.** Always one single-channel map regardless of K.  
**C.** K-1 output classes.  
**D.** K different datasets.

<details>
<summary>Show answer</summary>

**A.** Number of filters ↔ **output channels** (per spatial location over depth).

</details>

### 47. Pooling layer: max and average pooling

**Max pooling** over a small spatial window:

**A.** Takes the **maximum** value in each window (often reducing spatial size).  
**B.** Increases the size of the image.  
**C.** Computes a score for each class.  
**D.** Replaces all values with zero.

<details>
<summary>Show answer</summary>

**A.** **Downsample** + **local summary**; average pooling uses the **mean** instead.

</details>

### 48. Fully connected layer = linear layer

A **fully connected** layer means:

**A.** Every input unit connects to every output unit with its own weight.  
**B.** A non-linear mapping.  
**C.** No learnable parameters.  
**D.** Hard to learn.

<details>
<summary>Show answer</summary>

**A.** Dense **linear** map $W\mathbf{x}+\mathbf{b}$.

</details>

### 49. Why residual connections in ResNet?

Residual shortcuts in **ResNet** mainly help:

**A.** Train **very deep** networks by easing gradient flow through identity skips.  
**B.** Remove all nonlinearities.  
**C.** Increase the size of the image.  
**D.** Remove convolutions entirely.

<details>
<summary>Show answer</summary>

**A.** **Identity + residual** blocks mitigate vanishing/optimization issues in depth.

</details>

### 50. Strong training signal at every layer (ResNet)

Skip connections can give **earlier layers** more direct paths for gradients because:

**A.** Gradients can propagate along shortcuts.  
**B.** All layers share one single weight.  
**C.** The loss becomes better.  
**D.** Training skips the forward pass.

<details>
<summary>Show answer</summary>

**A.** **Shortcut paths** improve **gradient** transport in deep stacks.

</details>

### 51. What is U-Net

**U-Net** is best described as:

**A.** A **U-shaped** encoder–decoder with skip connections, often for **dense** per-pixel outputs the same size as the input image.  
**B.** A classifier that outputs only one label for the whole image only.  
**C.** A pure language model.  
**D.** Only logistic regression.

<details>
<summary>Show answer</summary>

**A.** Common for **segmentation**-style **pixel-wise** prediction; skips preserve spatial detail.

</details>
