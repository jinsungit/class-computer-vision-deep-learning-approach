## Multi-layer perceptron (MLP)

A **multi-layer perceptron (MLP)** is a **fully connected feedforward** network: the input passes through a stack of layers, each of which applies a **linear map** (affine transform) followed by a **nonlinear activation**. There are no skip connections or convolutions inside a plain MLP—every output unit in a layer can depend on **every** input unit from the previous layer.

In computer vision, raw MLPs on full images are usually too parameter-heavy, but MLP **blocks** appear everywhere: classification **heads** on top of CNN/ViT features, **projection** layers in contrastive learning, and **MLP mixers** in hybrid architectures.

### Main idea

Stacking linear layers **without** nonlinearities is equivalent to **one** linear layer, because a composition of linear maps is still linear. **Nonlinear activations** between layers let the network approximate complex functions: each layer warps the feature space so the next layer can separate patterns that were not linearly separable in the input.

At a high level:

$$
\mathbf{h}^{(0)} = \mathbf{x}, \qquad
\mathbf{h}^{(\ell)} = \sigma\big(\, W^{(\ell)} \mathbf{h}^{(\ell-1)} + \mathbf{b}^{(\ell)} \,\big), \quad \ell = 1,\ldots,L,
$$

with output $\mathbf{y} = \mathbf{h}^{(L)}$ (or an additional readout layer). Here $W^{(\ell)}$ is a weight matrix, $\mathbf{b}^{(\ell)}$ a bias vector, and $\sigma$ an activation applied **element-wise** (unless noted otherwise).


### Typical structure


```{figure} ../_static/imgs/neural_nets/mlp.png
:width: 80%
:alt: MLP

An example of a MLP used in the transformer architecture.
```

A practical MLP block often repeats:

**Linear** $\rightarrow$ **Activation** $\rightarrow$ (optional) **Dropout** $\rightarrow$ **Linear** $\rightarrow$ …

Conceptual flow (same information as a layer diagram, without extra tooling):

```
Input vector  →  Linear + bias  →  Activation (e.g. ReLU)  →  Dropout (training only)
       →  Linear + bias  →  Activation  →  …  →  Output logits
```

- **Width**: number of units per hidden layer (hidden dimension).
- **Depth**: number of affine+activation stages (how many times you alternate linear and $\sigma$).
- **Output head**: for $C$-way classification, the last linear layer maps to $C$ logits; for regression, a single linear output or multiple outputs as needed.



### Layer-wise forward pass (notation)

For one layer with input $\mathbf{h} \in \mathbb{R}^{d_{\mathrm{in}}}$:

$$
\mathbf{z} = W \mathbf{h} + \mathbf{b}, \qquad W \in \mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{in}}}, \quad \mathbf{b} \in \mathbb{R}^{d_{\mathrm{out}}}.
$$

In **batch** form, with $B$ examples stacked as rows $H \in \mathbb{R}^{B \times d_{\mathrm{in}}}$:

$$
Z = H W^{\top} + \mathbf{1} \mathbf{b}^{\top}.
$$

Activations are applied to $Z$ (or to $HW^\top + b$ in row layout, depending on framework conventions).

#### Visualization of a linear layer

Each output coordinate is a **weighted sum** of all inputs plus a bias—hence “fully connected.” You can read $W$ as defining $d_{\mathrm{out}}$ linear **filters** over the input vector.

```{figure} https://upload.wikimedia.org/wikipedia/commons/e/eb/Matrix_multiplication_diagram_2.svg
:width: 55%
:alt: Matrix multiplication diagram

A linear layer is batch matrix multiplication: each row of the output combines the corresponding input row with the weight matrix (layout matches your framework’s convention for $W$ vs $W^{\top}$).
```

#### Visualization of a non-linear activation function

Common choices in vision models include **ReLU** $\sigma(z) = \max(0, z)$, **GELU**, and **SiLU/Swish**. They introduce **sparsity** (ReLU zeros out negative pre-activations) or smooth gates (GELU/SiLU), which changes which directions in feature space are active for the next layer.

```{figure} https://upload.wikimedia.org/wikipedia/commons/9/99/ReLU_Activation_Function_Plot.svg
:width: 65%
:alt: ReLU activation function plot

ReLU is piecewise linear: negative inputs map to zero; positive inputs pass through unchanged. That keeps gradients flowing for active units while inducing sparsity.
```

```{figure} https://upload.wikimedia.org/wikipedia/commons/4/42/ReLU_and_GELU.svg
:width: 72%
:alt: ReLU and GELU activation functions

ReLU (sharp corner at zero) vs GELU (smooth). Many Transformers and ViTs use GELU in MLP sublayers.
```

#### A dropout layer

**Dropout** randomly sets a subset of activations to zero during **training** (each forward pass, with probability $p$). At **evaluation** time, dropout is turned off; implementations often **scale** activations by $(1-p)$ during training or by $1/(1-p)$ at test time so that expected magnitudes match.

Intuition: dropout approximates an **ensemble** of many thinner sub-networks and reduces **co-adaptation** of features.

Standard dropout: randomly dropped units (crossed out) do not contribute to the forward pass for that iteration; different units may be dropped on the next step.


### Why depth and width matter (intuition)

- **Width** increases the dimension of intermediate representations—more “room” to store features before the next mix.
- **Depth** alternates linear mixes and nonlinearities, enabling **hierarchical** transforms: early layers can implement low-level combinations; later layers can build on them. Very wide shallow networks and narrower deep networks can both be expressive; depth often improves **sample efficiency** for structured tasks, at the cost of optimization challenges (vanishing/exploding gradients—addressed in practice with normalization, residual connections elsewhere, and good initialization).

### Summary

- An MLP is a **sequence of affine maps** separated by **element-wise nonlinearities**; without nonlinearities, the whole stack collapses to a single linear map.
- **Linear layers** implement global mixing of features (all inputs to all outputs); **activations** create nonlinearity; **dropout** is a regularizer active only in training.
- In vision pipelines, MLPs often appear as **heads** or **projectors** on top of spatial encoders; the same layer math applies wherever you see `nn.Linear` in code.
