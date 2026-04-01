## U-Net

**U-Net** is an **encoder–decoder** convolutional network for **dense prediction**: every input pixel gets a label (e.g., semantic class), not one vector for the whole image. A **contracting path** (encoder) downsamples to build **context** and semantics; an **expanding path** (decoder) upsamples back to full resolution. **Skip connections** copy high-resolution features from the encoder to the decoder so boundaries and small structures are not lost in the bottleneck.

U-Net and its variants remain a standard baseline for **medical segmentation**, **remote sensing**, and **small-data** settings; the same ideas (encoder–decoder + skips) appear inside larger systems.

### Main idea

Let $x \in \mathbb{R}^{H \times W \times C_{\mathrm{in}}}$ be an input image. The encoder produces a sequence of feature maps with **decreasing** spatial size and **increasing** channel width; the decoder produces maps with **increasing** spatial size until the output matches $(H, W)$. At each decoder level, features are **concatenated** with the **same-resolution** encoder map (skip), then passed through convolutions.

Informally:

$$
\text{encoder: } x \mapsto \big(\mathbf{s}^{(1)}, \ldots, \mathbf{s}^{(L)}, \mathbf{z}\big), \qquad
\text{decoder: } \big(\mathbf{z}, \mathbf{s}^{(L)}, \ldots, \mathbf{s}^{(1)}\big) \mapsto \mathbf{y},
$$

where $\mathbf{s}^{(\ell)}$ are **saved** skip tensors and $\mathbf{z}$ is the **bottleneck** representation. The final $\mathbf{y}$ has shape $H \times W \times C_{\mathrm{out}}$ (e.g., $C_{\mathrm{out}}$ class logits per pixel).

```{figure} https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.PNG
:width: 88%
:alt: U-Net encoder-decoder with skip connections between matching resolutions

Contracting path (left): repeated conv blocks and downsampling. Expanding path (right): upsampling, **concatenation** with skips (gray arrows), then convolutions. The output layer predicts a dense map at the input resolution.
```

### Typical structure

A common block pattern:

```
Input image (H×W×C_in)
  → Encoder: [ Conv–ReLU–Conv–ReLU → downsample ] × L   (save skip after each block)
  → Bottleneck: Conv blocks at lowest resolution
  → Decoder: [ Upsample → Concat(skip) → Conv–ReLU–Conv–ReLU ] × L
  → Head: 1×1 conv  →  per-pixel logits (H×W×C_out)
```

- **Downsampling**: max pooling or strided convolution (halves $H,W$ each step in the classic U-Net).
- **Upsampling**: transposed convolution, interpolation + conv, or learned upsampling.
- **Skips**: **channel-wise concatenation** along the filter dimension (decoder sees both **semantics** from below and **detail** from the encoder).
- **Head**: $1\times 1$ convolution maps channel dimension to $C_{\mathrm{out}}$ logits (one score per class per pixel).

**Depth** $L$ and **base channel width** (e.g., 32 → 64 → 128 → …) control capacity; segmentation quality and data needs grow with both.

### Encoder–decoder forward pass (notation)

Write a single encoder stage (conv block, then downsample) abstractly as

$$
\mathbf{s}^{(\ell)} = \mathrm{ConvBlock}^{\mathrm{enc}}_{\ell}\big(\mathbf{h}^{(\ell-1)}\big), \qquad
\mathbf{h}^{(\ell)} = D_{\ell}\big(\mathbf{s}^{(\ell)}\big),
$$

with $\mathbf{h}^{(0)} = x$. Here $\mathbf{s}^{(\ell)}$ is the map **before** pooling/strided conv (saved for the skip); $D_{\ell}$ is downsampling; $\mathbf{h}^{(\ell)}$ feeds the next encoder level.

The decoder at level $\ell$ merges upsampled features with the skip:

$$
\mathbf{u}^{(\ell)} = \mathrm{ConvBlock}^{\mathrm{dec}}_{\ell}\Big(\mathrm{Concat}\big(\mathrm{Up}(\mathbf{u}^{(\ell+1)}), \mathbf{s}^{(\ell)}\big)\Big),
$$

with $\mathbf{u}^{(L)} = \mathbf{z}$ at the bottleneck. **Up** denotes upsampling (often factor 2). **Concat** stacks channels: if $\mathrm{Up}(\mathbf{u}^{(\ell+1)})$ has $C'$ channels and $\mathbf{s}^{(\ell)}$ has $C''$ channels, the result has $C'+C''$ channels before the next convolutions.

The **segmentation head** applies a per-pixel linear mix of channels:

$$
\mathbf{Y}_{:,:,c} = \sum_{k=1}^{C_{\mathrm{hid}}} W_{c,k}\,\mathbf{U}_{:,:,k} + b_c, \qquad c = 1,\ldots,C_{\mathrm{out}},
$$

implemented as one $1\times 1$ convolution. For multi-class segmentation, a **softmax** over $c$ is taken **across classes at each pixel**; training often uses **per-pixel cross-entropy** (see the supervised semantics chapter on pixel-level labels).

#### Visualization of the encoder (contracting path)

The encoder is a **CNN stack** that trades spatial resolution for **receptive field** and **abstraction**: early maps keep fine edges; deeper maps encode object- and context-level cues. Downsampling is the same idea as in a classification CNN (see **Convolutional Neural Networks**), but here we **retain** shallow maps for the skips instead of discarding them.

#### Visualization of skip connections

Skip connections are **shortcuts** across the bottleneck. Without them, the decoder would only see a **low-resolution** code and would struggle to place **exact boundaries**. Concatenation lets the decoder **fuse** “where” (encoder, high $H,W$) with “what” (decoder pathway from $\mathbf{z}$).

```{figure} https://upload.wikimedia.org/wikipedia/commons/2/2b/Example_architecture_of_U-Net_for_producing_k_256-by-256_image_masks_for_a_256-by-256_RGB_image.PNG
:width: 82%
:alt: Example U-Net producing k spatial masks for an RGB input

U-shaped layout: encoder branches feed the decoder at matching resolutions; the network outputs **dense** masks (here, $k$ channels at full resolution).
```

#### Visualization of the decoder (expanding path)

Each decoder step **grows** the grid (e.g., $H/8 \to H/4$) and **refines** features using convolutions after concatenation. The effective **receptive field** is large thanks to the encoder; the skips supply **alignment** so class boundaries follow image edges.

#### Visualization of the segmentation head

The final $1\times 1$ convolution is exactly a **per-pixel linear classifier**: it does **not** mix different spatial locations—it only mixes **channels** at each $(i,j)$. Spatial context was already built inside the encoder–decoder.

### Why skip connections matter (intuition)

- **Information bottleneck**: aggressive pooling discards exact positions of edges; skips pass those positions forward without forcing the bottleneck to store everything.
- **Gradient flow**: shortcuts give shorter paths for backpropagation into the encoder, which stabilizes training in deep stacks.
- **Data efficiency**: with limited labels, a U-Net often generalizes better than an encoder-only network that decodes from a single global vector.

### Summary

- U-Net maps an image to a **dense** label map using an **encoder** (down), **decoder** (up), and **skip concatenations** at matching resolutions.
- **Contracting path** builds semantics and context; **expanding path** restores resolution; **$1\times 1$ conv** turns channel maps into **per-class logits**.
- Training objectives and metrics (e.g., cross-entropy, Dice, IoU) are covered where **pixel-level supervision** is discussed; the architecture here is the **spatial** backbone.
