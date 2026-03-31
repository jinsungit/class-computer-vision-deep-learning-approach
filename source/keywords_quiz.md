# Self-check: Keyword concepts

---

## Math basics

### 1. What is *not* a function

Which situation breaks the usual definition of a function from a set $X$ to a set $Y$?

**A.** Two different inputs in $X$ map to the same output in $Y$.  
**B.** One input in $X$ is assigned two different outputs in $Y$.  
**C.** Each element of $X$ is assigned exactly one element of $Y$.  
**D.** Possible values of $X$ and $Y$ are discrete.  

<details>
<summary>Show answer</summary>

**B.** A function assigns **exactly one** output to each input; two outputs for the same input is not a function.  

</details>

### 2. What is gradient

The gradient $\nabla f$ of a function $f$ at a point (when it exists) points in the direction of:

**A.** Steepest **decrease** of $f$.  
**B.** Zero change no matter how you move.  
**C.** The global minimum.  
**D.** Steepest **increase** of $f$.  

<details>
<summary>Show answer</summary>

**D.** The gradient points in the direction of **steepest ascent** (steepest increase).  

</details>

### 3. Why we care about gradient

Why do we use gradients when training models?

**A.** They describe how small parameter changes can improve the model.  
**B.** They describe how good your model is.  
**C.** They only apply to linear models.  
**D.** They remove the need for any training data.  

<details>
<summary>Show answer</summary>

**A.** Gradients link **parameter changes** to **loss changes**, enabling iterative optimization (e.g., gradient descent).  

</details>

### 4. Min / max of functions and gradient

For a smooth function $f$, what is typically true at a local minimum where the gradient exists?

**A.** $\nabla f = 0$.  
**B.** $\nabla f$ is undefined.  
**C.** $\nabla f$ is any non-zero constant.  
**D.** $\nabla f = 1$.  

<details>
<summary>Show answer</summary>

**A.** At an interior local extremum (min or max) of a smooth function, the gradient is typically **zero**.  

</details>

### 5. Vector

In this course, a **vector** is best thought of as:

**A.** A single random number.  
**B.** A list (or array) of numbers.  
**C.** A curve.  
**D.** A transformation.  

<details>
<summary>Show answer</summary>

**B.** Vectors are usually **ordered collections of numbers** (coordinates in $\mathbb{R}^n$).  

</details>

### 6. Matrix

A **matrix** is naturally:

**A.** A 2D array of numbers, with rows and columns.  
**B.** Only used for images, never for text.  
**C.** The same dimension as a vector, just written differently.  
**D.** A single number.  

<details>
<summary>Show answer</summary>

**A.** A matrix is a **rectangular table** of numbers.  

</details>

### 7. Tensor

In deep learning usage, a **tensor** most often means:

**A.** A multi-dimensional array generalizing scalars, vectors, and matrices.  
**B.** A Python list of strings.  
**C.** The loss value.  
**D.** The input data.  

<details>
<summary>Show answer</summary>

**A.** Tensors generalize to **many axes** (e.g., batch, channels, height, width).  

</details>

### 8. Vector ops: addition, norm, and dot product

The **dot product** of two real vectors $\mathbf{u}$ and $\mathbf{v}$ is:

**A.** A vector.  
**B.** Always zero if $\mathbf{u} \neq \mathbf{v}$.  
**C.** Always positive.  
**D.** A **scalar** measuring the similarity between two vectors.  

<details>
<summary>Show answer</summary>

**D.** $\mathbf{u}\cdot\mathbf{v} = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$ — a **scalar**.  

</details>

### 9. What is a linear transform

Which choice best matches a **linear** map from $\mathbb{R}^n$ to $\mathbb{R}^m$ (no extra bias term)?

When we want to represent a linear transformation between two vectors, we use:

**A.** A data structure.  
**B.** A vector.  
**C.** A matrix.  
**D.** A scalar.  

<details>
<summary>Show answer</summary>

**C.** Linear transformations correspond to **matrix multiplication** (plus a bias in code is an **affine** map).  

</details>

### 10. What is linear basis

If we have a linear basis in a vector space, we can represent any vector as a linear combination of those basis vectors.

**A.** False.  
**B.** True.  

<details>
<summary>Show answer</summary>

**B.** Bases provide a coordinate system via **unique** linear combinations.  

</details>

### 11. Types of random variables: discrete and continuous

A **discrete** random variable takes values that are:

**A.** From a **countable** set (you can list outcomes, e.g. coin flip, class label).  
**B.** Not possible to list all values.  
**C.** Always Gaussian.  
**D.** Only integers (always) — no exceptions.  

<details>
<summary>Show answer</summary>

**A.** Discrete = outcomes from a **countable** set; continuous often means an interval of $\mathbb{R}$.  

</details>

### 12. Gaussian distribution: mean and variance

A univariate Gaussian $N(\mu, \sigma^2)$ is mainly specified by:

**A.** Only $\sigma^2$.  
**B.** Mean $\mu$ and variance $\sigma^2$.  
**C.** Only $\mu$.  
**D.** The maximum of the density only.  

<details>
<summary>Show answer</summary>

**B.** **Mean** (center) and **variance** (spread) define the standard 1D Gaussian.  

</details>

### 13. Bayes theorem and image classification example

Consider the following scenario: You are given a *photograph* and you need to classify it into either a dragon or a horse. We observe that the image looks both like a dragon and a horse, what should you do with the Bayes theorem-type reasoning? 


**A.** Neither of the above.  
**B.** Must be a dragon because it is more rare (low prior).  
**C.** I will guess randomly.  
**D.** Cannot be a dragon because it is more rare (low prior).  

<details>
<summary>Show answer</summary>

**D.** Posterior is proportional to **prior** $\times$ **likelihood**; that is the Bayes pattern for classification reasoning.  

</details>

---

## Machine learning basics

### 14. What is machine learning: predicting the future based on the past

In the usual ML story, models:

**A.** Must be hand-coded `if` statements only.  
**B.** Never use data, only closed-form physics formulas.  
**C.** Learn patterns from past examples to predict on **new** data.  
**D.** Memorize one training point and ignore the rest.  

<details>
<summary>Show answer</summary>

**C.** ML uses **past data** to generalize to **future / unseen** instances.  

</details>

### 15. Practice formulation of learning

In practice, we don't have full access to the entire data distribution, we have a finite dataset. Which of the following is correct?

**A.** Neither of the above.  
**B.** The model we learned from the finite dataset is not useful for the real data distribution.  
**C.** Since we don't have full access to the entire data distribution, we can't predict the future using the finite dataset.  
**D.** We can estimate the performance of the model by evaluating it on the dataset at hand.  

<details>
<summary>Show answer</summary>

**D.** We separate the data into training, validation, and test sets to estimate the performance of the model learned from the finite dataset on the real data distribution.  

</details>

### 16. What is supervised learning

**Supervised learning** means training with:

**A.** Exact input–output pairs $(x, y)$ where $y$ is the supervision signal.  
**B.** No data.  
**C.** Only $x$ with no labels.  
**D.** Random rewards from an environment (for example, a thumb-up from the user by clicking a button).  

<details>
<summary>Show answer</summary>

**A.** Supervision = provided **labels** or **targets** paired with inputs.  

</details>

### 17. Classification

**Classification** predicts:

**A.** A set.  
**B.** A **category** (class) or class probabilities.  
**C.** A cluster of data points with no class names.  
**D.** A real number like price or temperature.  

<details>
<summary>Show answer</summary>

**B.** Outputs are **discrete classes** (possibly as a distribution over classes).  

</details>

### 18. Regression

**Regression** typically predicts:

**A.** A permutation of labels.  
**B.** A **numerical** value.  
**C.** Only “spam vs not spam.”  
**D.** Is this image a cat or a dog?  

<details>
<summary>Show answer</summary>

**B.** Regression targets are usually **continuous** quantities.  

</details>

### 19. Classification vs regression: which one to choose

You usually pick **classification** over regression NOT because:

**A.** Classification is more well-defined and well-studied.  
**B.** You have to use a big model.  
**C.** The target values are from a set of possible values.  
**D.** You care more about something being present or not.  

<details>
<summary>Show answer</summary>

**B.** Whether to use classification or regression is not related to the size of the model.  

</details>

### 20. What is unsupervised learning

**Unsupervised learning** works with data:

**A.** With numerical labels.  
**B.** Without provided labels, to find structure (i.e., clusters).  
**C.** With discrete labels.  
**D.** Only as $(x,y)$ pairs with labels for every point.  

<details>
<summary>Show answer</summary>

**B.** **No labels** — discover structure from $x$ alone (or weak signals).  

</details>

### 21. Idea of K-means clustering

**K-means** will find:

**A.** K clusters.  
**B.** K-1 clusters.  
**C.** Unknown number of clusters.  
**D.** K+1 clusters.  

<details>
<summary>Show answer</summary>

**A.** K-means will find K clusters, but it might not be the optimal number of clusters.  

</details>

### 22. Idea of dimensionality reduction

**Dimensionality reduction** seeks to:

**A.** Expand data.  
**B.** Represent data with **fewer** dimensions while keeping useful information.  
**C.** Duplicate features.  
**D.** Increase noise in data.  

<details>
<summary>Show answer</summary>

**B.** Lower-dimensional representations for visualization, speed, or denoising.  

</details>

### 23. Model is a function mapping between $x$ and $y$

In supervised learning, a **model** is:

**A.** Some CSV file on the hard drive.  
**B.** The evaluation metric alone.  
**C.** A map from inputs $x$ to predictions $\hat{y}$ (approximating the true $y$).  
**D.** The optimizer without parameters.  

<details>
<summary>Show answer</summary>

**C.** A learned **function** $x \mapsto \hat{y}$.  

</details>

### 24. Linear model is a model with linear function $Ax$

Here “linear model” means predictions are:

**A.** Random guesses.  
**B.** A deep ReLU network with 100 layers.  
**C.** A **linear** function of the features (e.g., $\hat{y} = A x$ in vector form, maybe plus bias).  
**D.** A quadratic function $f(x) = ax^2 + bx + c$.  

<details>
<summary>Show answer</summary>

**C.** **Linear** in the feature vector (not necessarily linear in underlying signals if features are fixed).  

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

**A.** Remove bias from the model.  
**B.** Estimate performance more reliably by repeating train/eval on different partitions.  
**C.** Guarantee global optimality.  
**D.** Measure the quality of the dataset.  

<details>
<summary>Show answer</summary>

**B.** Averages over splits to reduce **variance** in the performance estimate.  

</details>

### 27. What is a metric

An evaluation **metric** is:

**A.** A quantity summarizing how good predictions are (accuracy, mAP, RMSE, …).  
**B.** The batch size.  
**C.** Not very practical to use.  
**D.** The name of the optimizer.  

<details>
<summary>Show answer</summary>

**A.** Metrics **measure** quality on a dataset or benchmark.  

</details>

### 28. What is overfitting

**Overfitting** usually means:

**A.** We trained the model for too few steps.  
**B.** The dataset is too difficult.  
**C.** The model fits training data **too** closely and may perform poorly on new data.  
**D.** The model is too simple.  

<details>
<summary>Show answer</summary>

**C.** **High train, low test** — memorization / high variance.  

</details>

---

## Deep learning basics

### 29. AI = deep learning?

Which statement is most accurate?

**A.** AI and deep learning mean unrelated things.  
**B.** Deep learning is **one** family of methods inside the broader field of AI.  
**C.** Deep learning never uses data.  
**D.** Deep learning is the only topic in all of AI.  

<details>
<summary>Show answer</summary>

**B.** **AI** is broad; **deep learning** is a prominent **subset** (neural networks with depth and representation learning).  

</details>

### 30. Neuron vs neural networks

A single **neuron** in a neural network is:

**A.** Only a loss value.  
**B.** The entire trained model by definition.  
**C.** The same as the full dataset.  
**D.** One small computational unit; a **neural network** combines many such units in layers.  

<details>
<summary>Show answer</summary>

**D.** **Compose** many neurons → network.  

</details>

### 31. What is an activation function? ReLU

The **ReLU** activation is:

**A.** $\tanh(z)$.  
**B.** $|z|^2$.  
**C.** $1/(1+e^{-z})$.  
**D.** $\max(0, z)$.  

<details>
<summary>Show answer</summary>

**D.** ReLU “clips” negatives to zero.  

</details>

### 32. A single (linear) layer

One **linear layer** typically computes:

**A.** A decision tree split.  
**B.** Softmax.  
**C.** $\mathbf{y} = W\mathbf{x} + \mathbf{b}$ (matrix multiply plus bias).  
**D.** Spatial max pooling.  

<details>
<summary>Show answer</summary>

**C.** Linear map = **weight matrix** and optional **bias**.  

</details>

### 33. Multi-layer → learn complex mapping

**Stacking** nonlinear layers allows a network to:

**A.** Ignore the input.  
**B.** Remove all parameters.  
**C.** Represent only linear functions overall.  
**D.** Build **complex** input–output relationships than one linear layer alone.  

<details>
<summary>Show answer</summary>

**D.** Composition of linear + nonlinear layers yields **nonlinear** functions, allowing the network to approximate complex relationships.  

</details>

### 34. Main idea of training: data, forward, backward, update

Training usually repeats:

**A.** Random weight changes with no loss.  
**B.** Forward pass → loss → backward pass → parameter update.  
**C.** Sorting the dataset.  
**D.** Forward pass only, forever.  

<details>
<summary>Show answer</summary>

**B.** Classic **supervised** neural net training loop.  

</details>

### 35. Backpropagation in the backward pass

**Backpropagation** computes:

**A.** Gradients of the data with respect to the loss.  
**B.** Gradients of the loss with respect to model parameters.  
**C.** Another round of forward pass.  
**D.** Metrics for the trained model.  

<details>
<summary>Show answer</summary>

**B.** Backpropagation is an efficient **gradient** computation method for multiple layers of networks.  

</details>

### 36. The concept of batch in deep learning

A **mini-batch** is:

**A.** The validation set only.  
**B.** One convolution kernel.  
**C.** A small **subset** of examples used for one gradient step.  
**D.** The whole training set every step.  

<details>
<summary>Show answer</summary>

**C.** Batching balances **noise** vs **memory** vs **speed**.  

</details>

### 37. What is a loss function

A **loss** for training is:

**A.** A list of layer names.  
**B.** A scalar objective we try to **minimize** so the model fits data.  
**C.** Only used after deployment, never in training.  
**D.** Always identical to accuracy.  

<details>
<summary>Show answer</summary>

**B.** Loss drives **optimization** (when differentiable).  

</details>

### 38. Loss vs metric

Which is **true**?

**A.** Metrics are never computed on validation data.  
**B.** Loss is what training optimizes; metrics summarize performance for analysis and reporting.  
**C.** Loss cannot depend on model outputs.  
**D.** Loss and metric must always be the same number.  

<details>
<summary>Show answer</summary>

**B.** **Loss** for **learning**; **metrics** for **evaluation** and comparison.  

</details>

### 39. Neural nets

In this class, “**neural network**” usually means:

**A.** Computer Vision.  
**B.** A parameterized composition of layers trained with data and gradients.  
**C.** Some spreadsheet data.  
**D.** Only a biological brain simulation with no math.  

<details>
<summary>Show answer</summary>

**B.** **Learned** layered function, typically **differentiable**.  

</details>

### 40. Feedforward networks have no feedback

A **feedforward** network:

**A.** Has feedback loops when we run the forward pass without running the backward pass.  
**B.** Passes information from input toward output **without** recurrent cycles and lateral connections in the graph.  
**C.** Cannot have more than one layer.  
**D.** Is also called a recurrent network.  

<details>
<summary>Show answer</summary>

**B.** No **recurrent connections** along the main forward path (contrast with RNNs). Does not have feedback loops when we just run the forward pass.  

</details>

### 41. Multi-layer Perceptron (MLP)

An **MLP** is typically:

**A.** A purely linear layer with no activations.  
**B.** Only k-nearest neighbors.  
**C.** Alternating **linear** layers and **nonlinear activations** (sometimes dropout etc.).  
**D.** Only one convolution.  

<details>
<summary>Show answer</summary>

**C.** **Stacked** linear + nonlinear blocks (fully connected).  

</details>

### 42. Idea of dropout layer

**Dropout** during training:

**A.** Randomly zeros some units to reduce heavy dependence on a few units.  
**B.** Forces all activations to 1.  
**C.** Removes backpropagation.  
**D.** Deletes the dataset from disk.  

<details>
<summary>Show answer</summary>

**A.** **Random** neuron masking as **regularization**.  

</details>

### 43. Idea of convolution in 2D

2D convolution slides a **small filter** over an image so **each output location** combines information from:

**A.** Random pixels ignoring position.  
**B.** A **local neighborhood**.  
**C.** All pixels in the image.  
**D.** Only the top row of pixels.  

<details>
<summary>Show answer</summary>

**B.** **Local receptive field** + **weight sharing** across positions.  

</details>

### 44. 2D convolution example

If we have a 3x3 input image and a 3x3 filter, and we run 2D convolution once by aligning the filter at the center of the input image, the output size will be:

**A.** 4x4.  
**B.** 2x2.  
**C.** 1x1.  
**D.** 3x3.  

<details>
<summary>Show answer</summary>

**C.** 1x1.  

</details>

### 45. Main idea of CNNs

**CNNs** exploit:

**A.** Pixel sorting.  
**B.** Multiple small filters to extract useful information.  
**C.** Fully connecting every pixel to every other at all layers.  
**D.** Unsupervised clustering.  

<details>
<summary>Show answer</summary>

**B.** **Spatial structure** + **parameter sharing** → efficient vision models.  

</details>

### 46. Convolutional layers: filters = output channels

If a conv layer learns **K** distinct filters for the given input depth, you typically get:

**A.** **K** output feature maps (channels).  
**B.** K-1 output classes.  
**C.** Always one single-channel map regardless of K.  
**D.** K different datasets.  

<details>
<summary>Show answer</summary>

**A.** Number of filters ↔ **output channels** (per spatial location over depth).  

</details>

### 47. Pooling layer: max and average pooling

**Max pooling** over a small spatial window:

**A.** Computes a score for each class.  
**B.** Replaces all values with zero.  
**C.** Takes the **maximum** value in each window (often reducing spatial size).  
**D.** Increases the size of the image.  

<details>
<summary>Show answer</summary>

**C.** **Downsample** + **local summary**; average pooling uses the **mean** instead.  

</details>

### 48. Fully connected layer = linear layer

A **fully connected** layer means:

**A.** A non-linear mapping.  
**B.** No learnable parameters.  
**C.** Hard to learn.  
**D.** Every input unit connects to every output unit with its own weight.  

<details>
<summary>Show answer</summary>

**D.** Dense **linear** map $W\mathbf{x}+\mathbf{b}$.  

</details>

### 49. Why residual connections in ResNet?

Residual shortcuts in **ResNet** mainly help:

**A.** Remove all nonlinearities.  
**B.** Train **very deep** networks by easing gradient flow through identity skips.  
**C.** Remove convolutions entirely.  
**D.** Increase the size of the image.  

<details>
<summary>Show answer</summary>

**B.** **Identity + residual** blocks mitigate vanishing/optimization issues in depth.  

</details>

### 50. Strong training signal at every layer (ResNet)

Skip connections can give **earlier layers** more direct paths for gradients because:

**A.** Gradients can propagate along shortcuts.  
**B.** All layers share one single weight.  
**C.** Training skips the forward pass.  
**D.** The loss becomes better.  

<details>
<summary>Show answer</summary>

**A.** **Shortcut paths** improve **gradient** transport in deep stacks.  

</details>

### 51. What is U-Net

**U-Net** is best described as:

**A.** A **U-shaped** encoder–decoder with skip connections, often for **dense** per-pixel outputs the same size as the input image.  
**B.** A pure language model.  
**C.** Only logistic regression.  
**D.** A classifier that outputs only one label for the whole image only.  

<details>
<summary>Show answer</summary>

**A.** Common for **segmentation**-style **pixel-wise** prediction; skips preserve spatial detail.  

</details>
