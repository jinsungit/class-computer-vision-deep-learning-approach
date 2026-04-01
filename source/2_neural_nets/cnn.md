## Convolutional Neural Networks (CNN)

A **convolutional neural network (CNN)** is a feedforward model built from **convolution** layers (plus nonlinearities, and often **pooling** and **fully connected** readout layers). Instead of connecting every input pixel to every output unit like an MLP, a conv layer uses **small, shared filters** that slide across the image. That yields **far fewer parameters**, **local receptive fields**, and a useful inductive bias for natural images: similar patterns can appear at different locations.

### Convolution

```{figure} ../_static/imgs/neural_nets/conv.PNG
:width: 80%
:alt: Convolution

Convolution.
```


```{figure} ../_static/imgs/neural_nets/conv2.PNG
:width: 80%
:alt: Convolution

Convolution.
```

```{figure} ../_static/imgs/neural_nets/conv3.PNG
:width: 80%
:alt: Convolution

Convolution.
```




#### Idea of convolution

In deep learning frameworks, “convolution” usually means **2D discrete cross-correlation**: a **kernel** (filter) of learnable weights slides over the input feature map; at each position, we take an **elementwise product** with the patch under the kernel, **sum**, add a **bias**, and store the result in the output map.

Key properties:

- **Local connectivity**: each output depends only on a small spatial neighborhood—unlike a dense layer that mixes the whole image at once.
- **Weight sharing**: the **same** kernel is reused at every spatial location, which cuts parameters and encourages detectors that fire anywhere in the field (translation sensitivity in feature space).
- **Translation equivariance** (for a single conv layer with stride 1 and appropriate padding): shifting the input shifts the output feature map in the same way. (Pooling and strided convs change this story toward **invariance** at higher levels.)

**Convention:** libraries such as PyTorch implement **cross-correlation** (no flip of the kernel). Pure mathematical convolution flips the kernel; the distinction matters for signal-processing proofs but rarely changes how you *use* `nn.Conv2d`.

#### 2D illustration

The “convolution arithmetic” figures below (Dumoulin & Visin) show how kernel size, **stride**, and **padding** determine the output spatial size. Blue tiles are inputs; the green area is the receptive field moving over the map.

```{figure} https://upload.wikimedia.org/wikipedia/commons/6/6c/Convolution_arithmetic_-_No_padding_no_strides.gif
:width: 75%
:alt: Animation of 2D convolution with no padding and no strides

No padding, stride 1: the output map is **smaller** than the input (unless the kernel is $1\times 1$).
```

```{figure} https://upload.wikimedia.org/wikipedia/commons/e/ee/Convolution_arithmetic_-_Same_padding_no_strides.gif
:width: 75%
:alt: Animation of 2D convolution with same padding and no strides

“Same” padding (when stride $=1$): the output **height and width** match the input, so deeper stacks do not shrink every layer.
```

```{figure} ../_static/imgs/neural_nets/conv_example.PNG
:width: 80%
:alt: 2D convolution example

2D convolution example.
```




#### 2D example (one output location)

Let $x$ be a single-channel input and $w$ a $K\times K$ kernel. At output position $(i,j)$ (using the usual sliding-window indexing):

$$
y_{i,j} = \sum_{m=0}^{K-1} \sum_{n=0}^{K-1} x_{i+m,\,j+n} \, w_{m,n} + b.
$$

With **multiple input channels** $c$, each channel has its own slice of weights; those slices **add**:

$$
y_{i,j} = \sum_{c} \sum_{m=0}^{K-1} \sum_{n=0}^{K-1} x_{c,\,i+m,\,j+n} \, w_{c,m,n} + b.
$$

A conv layer with $C_{\mathrm{out}}$ filters stacks $C_{\mathrm{out}}$ such outputs—each filter is a full 3D tensor over $(C_{\mathrm{in}}, K, K)$.

For **stride** $S$ and **zero-padding** $P$ along one spatial axis of length $I$, the output length is

$$
O = \left\lfloor \frac{I + 2P - K}{S} \right\rfloor + 1
$$

(and similarly for height/width when $K,S,P$ differ per axis).

### Main idea of CNNs

A CNN stacks **conv $\rightarrow$ activation** blocks, often with **pooling** or **strided conv** for downsampling, so that early layers respond to **edges and textures**, and deeper layers build **parts and objects**. The final **spatial** maps are **vectorized** (global average pooling is common) and fed to a small **MLP** head for classification or regression.


```{figure} ../_static/imgs/neural_nets/cnn.PNG
:width: 80%
:alt: CNN

A CNN architecture (VGG-16).
```



#### Convolutional layers

- **Multiple filters** per layer learn different features in parallel (output **channels**).
- **$1\times 1$ convolutions** mix channels at each spatial location without enlarging the receptive field—cheap “channel mixing.”
- **Dilated convolutions** widen the receptive field without extra pooling by spacing kernel taps (common in segmentation).

#### Pooling layers

**Max pooling** and **average pooling** downsample: each output summarizes a neighborhood (e.g., $2\times 2$ with stride 2 halves resolution). Max pooling tends to preserve **strong activations** (salient edges); average pooling smooths.

Pooling increases **translation tolerance** in the pooled regions but **throws away** spatial detail—design choice depends on the task.


```{figure} ../_static/imgs/neural_nets/pooling.PNG
:width: 80%
:alt: Pooling

Pooling.
```



#### Fully connected layers

After enough conv layers, the network may **flatten** $C\times H\times W$ features into a vector and apply `Linear` layers. Many image classifiers instead use **global average pooling** over space to a $C$-dimensional vector, then a single linear layer to class logits—fewer parameters and less overfitting than huge FC layers on large spatial grids.

### Residual connections


```{figure} ../_static/imgs/neural_nets/res1.PNG
:width: 80%
:alt: Residual connections


Residual connections in neural networks allows to train very deep networks.
```

```{figure} ../_static/imgs/neural_nets/res2.PNG
:width: 80%
:alt: Residual connections


Residual connections in neural networks allows to train very deep networks.
```


```{figure} ../_static/imgs/neural_nets/res3.PNG
:width: 80%
:alt: Residual connections


Residual connections in neural networks allows to train very deep networks.
```




Very deep plain CNNs can be hard to optimize (**vanishing gradients**, degradation). **Residual networks (ResNet)** add **skip connections**: a block learns a **residual** mapping $F$ relative to identity,

$$
\mathbf{y} = F(\mathbf{x}) + \mathbf{x},
$$

so layers can learn small corrections and gradients can flow along the shortcut.

```{figure} https://upload.wikimedia.org/wikipedia/commons/6/6f/Resnet-18_architecture.svg
:width: 95%
:alt: ResNet-18 architecture diagram with residual skip connections

ResNet-18 stacks residual blocks; skips bypass convolutions and help train deep networks. Channel dimensions are matched with $1\times 1$ convolutions when needed.
```

When shapes differ, the skip is **projected** (e.g., $1\times 1$ conv) so $F(\mathbf{x})$ and $\mathbf{x}$ can be added elementwise.

### Summary

```{figure} ../_static/imgs/neural_nets/summary_cnn.PNG
:width: 80%
:alt: Summary of CNNs

Summary of CNNs.
```


- CNNs replace global dense mixing with **local, shared** filters—efficient and well matched to spatial structure in images.
- **Stride**, **padding**, and **kernel size** control output resolution and receptive field; multi-channel convs **sum** over input channels and **stack** output channels.
- **Pooling** (or strided conv) builds **hierarchy** and some **invariance**; **FC** or **global pooling + linear** heads produce task outputs.
- **Residual connections** stabilize and scale **depth**; they are standard in modern CNN backbones (ResNet, ConvNeXt, etc.).
