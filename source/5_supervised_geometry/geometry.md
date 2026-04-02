## Geometry in computer vision

### Motivation

Almost every vision algorithm reasons about **where things are** on the image plane and how that relates to **where things are** in the world. Even when networks learn geometry implicitly (depth, poses, NeRF), the training signal and evaluation metrics still connect back to the same few ideas:

- **2D geometry**: pixel coordinates, edges, regions, and how they move when the viewpoint or the object moves.
- **3D geometry**: points, planes, and rigid motion in space—the objects we would like to recover from images.
- **Projection**: the mathematical link between 3D and 2D, usually through a **pinhole camera** model and its **intrinsic** and **extrinsic** parameters.

Why this matters for supervised learning:

- Depth and point-cloud supervision are defined in **metric or relative 3D**; losses compare predictions to ground truth in those spaces.
- Multi-view reconstruction, SLAM, and AR need consistent **cameras** across frames.
- Classical calibration and modern learning-based depth both assume (or learn a substitute for) the same projection structure.

Intuition: treat an image as a **measurement device**. Each pixel is a ray through the camera center; geometry tells you how to interpret that measurement and how measurements change when the camera or scene changes.

---

## 2D geometry

### Point

A **Euclidean** point in the image is $\mathbf{x} = (x, y)^\top$. In projective geometry we often use **homogeneous coordinates** $\tilde{\mathbf{x}} = (x, y, 1)^\top$ (defined up to scale: $(x,y,1) \sim (\lambda x, \lambda y, \lambda)$ for $\lambda \neq 0$).

Homogeneous points are the bookkeeping trick that lets us write many transforms—including projection from 3D—as a single matrix multiply. You multiply by a $3\times 3$ matrix on the left and then **dehomogenize** by dividing by the third coordinate when you need Euclidean $(x,y)$ again.

### Line

A line can be written as $ax + by + c = 0$, or in homogeneous form as $\tilde{\boldsymbol{\ell}} = (a, b, c)^\top$. A point $\tilde{\mathbf{x}}$ lies on the line when

$$
\tilde{\boldsymbol{\ell}}^\top \tilde{\mathbf{x}} = 0.
$$

Useful fact: in homogeneous coordinates, the **line through two points** is the cross product $\tilde{\mathbf{x}}_1 \times \tilde{\mathbf{x}}_2$, and the **intersection of two lines** is $\tilde{\boldsymbol{\ell}}_1 \times \tilde{\boldsymbol{\ell}}_2$ (up to scale). That is why homogeneous algebra shows up in multi-view geometry.

### Transformation

| Name | DoF | What it preserves | Typical matrix (2D homogeneous) |
|------|-----|-------------------|---------------------------------|
| **Translation** | 2 | Distances, angles | $\begin{bmatrix} I & \mathbf{t} \\ \mathbf{0}^\top & 1 \end{bmatrix}$ |
| **Rotation** (about origin) | 1 | Distances, angles, origin | $\begin{bmatrix} R & \mathbf{0} \\ \mathbf{0}^\top & 1 \end{bmatrix}$, $R \in \mathrm{SO}(2)$ |
| **Euclidean** (rigid in plane) | 3 | Distances, angles | Translation + rotation |
| **Similarity** | 4 | Angles, ratios of lengths | Uniform scale + Euclidean |
| **Affine** | 6 | Parallelism, ratios on lines | Linear part + translation |
| **Projective** (homography) | 8 | Incidence (points on lines) | General invertible $3\times 3$ |

Intuitions:

- **Rigid (Euclidean) motion** in 2D is what you get if a flat object rotates and translates in the plane; lengths and angles in the object do not change.
- A **homography** maps one perspective view of a **plane** to another perspective view of the **same plane**. That is why rectifying documents or stitching planar panoramas uses a $3\times 3$ **homography** matrix.

```{figure} https://upload.wikimedia.org/wikipedia/commons/9/94/Projection_geometry.svg
:width: 55%
:alt: Perspectivity between two planes

A **perspectivity** (a special projective map) sends points $A,B,C,D$ to $A',B',C',D'$ through a center of projection. Planar homographies in vision are built from compositions of perspective projections; straight lines stay straight, but lengths and angles need not be preserved.
```

---

## 3D geometry

### Point

A 3D point is $\mathbf{X} = (X, Y, Z)^\top$ in some world or camera coordinate system. Homogeneous form is $\tilde{\mathbf{X}} = (X, Y, Z, 1)^\top$ (or $(X,Y,Z,W)^\top$ with scale ambiguity). The extra coordinate again unifies rotation, translation, and projection into matrix form.

### Plane

A plane is $\pi_1 X + \pi_2 Y + \pi_3 Z + \pi_4 = 0$, or $\tilde{\boldsymbol{\pi}}^\top \tilde{\mathbf{X}} = 0$ in homogeneous 3D. The normal direction in Euclidean space is $(\pi_1,\pi_2,\pi_3)$. Planes are the 3D analogue of lines: they are the natural way to describe floors, walls, tables, and other large surfaces in scene reasoning.

### Transformation

| Name | DoF | What it encodes |
|------|-----|-----------------|
| **Translation** | 3 | Shift $\mathbf{t} \in \mathbb{R}^3$ |
| **Rotation** | 3 | Orientation $R \in \mathrm{SO}(3)$ (orthogonal, $\det(R)=1$) |
| **Rigid** (Euclidean) | 6 | $R$ and $\mathbf{t}$: preserves distances |
| **Similarity** | 7 | Rigid + uniform scale |
| **Affine** | 12 | Invertible linear map + translation |

**Rigid motion** in 3D is the workhorse for **camera pose** and **object pose**: a world point $\mathbf{X}_w$ rigidly moves to $\mathbf{X}_c = R \mathbf{X}_w + \mathbf{t}$ when expressed in a camera-centered frame (depending on convention, either $\mathbf{X}_{w\to c} = R\mathbf{X}_w+\mathbf{t}$ or the inverse relation describes “where the camera is in the world”; always draw the arrow World → Camera on your diagram once and stick to it).

```mermaid
flowchart LR
  Xw[World point] --> E["Extrinsics (R, t)"]
  E --> Xc[Camera coordinates]
  Xc --> Kv["Intrinsics (K)"]
  Kv --> xp[Pixel coordinates]
```

Extrinsics describe how the world frame relates to the camera frame ($R$, $\mathbf{t}$); intrinsics then turn camera rays into pixel indices. Conventions differ by textbook—the interactive **camera_params** demo below fixes one clear camera rig you can orbit around.

Rotations in 3D can be represented as **matrices** ($3\times 3$), **unit quaternions**, or **axis–angle**; different representations trade numerical stability and interpolation quality. For learning, **6D continuous representations** are common when MLPs regress rotation.

---

## Pinhole camera model

Mathematically, a camera is a map from a 3D point to a 2D image location. The **pinhole** model is the first accurate approximation used in most vision pipelines: all rays pass through a single **center of projection** (the pinhole), and the sensor records where each ray hits an **image plane** placed at focal distance.

Standard perspective projection (camera coordinates: $Z$ along optical axis, image plane at $Z=f$):

$$
x = f \frac{X}{Z}, \qquad y = f \frac{Y}{Z}.
$$

Parallel lines in the world meet at a **vanishing point** in the image because division by $Z$ is nonlinear: depth scales how “spread out” the image of a 3D object appears.

**Mistake to avoid**: mixing **camera-frame** vs **world-frame** coordinates without applying extrinsics first. Always transform $\mathbf{X}_w \to \mathbf{X}_c$ before applying the intrinsics that map $\mathbf{X}_c$ to pixels.

### Interactive: pinhole geometry (focal length and image plane)

The scene below is a **toy 3D pinhole setup** (not your dataset’s real calibration): orbit with the mouse to see the center of projection, focal length along the optical axis, and the image plane.

<iframe src="camera.html" width="100%" height="520" style="border:0; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.15);" title="Interactive pinhole camera demo"></iframe>

*Direct link if the embed is blocked:* open [`camera.html`](camera.html) in the browser.

---

## Camera parameters

We factor the projection into **extrinsic** (where the camera is) and **intrinsic** (how pixels relate to rays).

### Intrinsics

The **intrinsic matrix** $\mathbf{K}$ maps **camera-frame** 3D direction (from the center through $\mathbf{X}_c$) to **pixel coordinates**:

$$
\mathbf{K} =
\begin{bmatrix}
f_x & s & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix},
$$

where $f_x, f_y$ are focal lengths in **pixel units**, $(c_x, c_y)$ is the **principal point** (intersection of the optical axis with the sensor), and $s$ is skew (often $0$ for modern sensors). With homogeneous pixels $\tilde{\mathbf{x}}_p$ and homogeneous camera point $\tilde{\mathbf{X}}_c$:

$$
\lambda \, \tilde{\mathbf{x}}_p = \mathbf{K} \, [\, I \mid \mathbf{0} \,] \, \tilde{\mathbf{X}}_c = \mathbf{K} \mathbf{X}_c
$$

for some $\lambda = Z_c$. So intrinsics encode **field of view** and **pixel sampling**; they do not depend on where the camera sits in the world.

### Extrinsics

Extrinsics are a rigid transform from **world** to **camera** (or the inverse, depending on convention). A common form:

$$
\tilde{\mathbf{X}}_c \sim
\begin{bmatrix}
R & \mathbf{t} \\
\mathbf{0}^\top & 1
\end{bmatrix}
\tilde{\mathbf{X}}_w.
$$

Then the full **projection** is

$$
\lambda \, \tilde{\mathbf{x}}_p = \mathbf{K}
\begin{bmatrix}
R & \mathbf{t}
\end{bmatrix}
\tilde{\mathbf{X}}_w = \mathbf{P} \, \tilde{\mathbf{X}}_w,
$$

where $\mathbf{P}$ is the $3\times 4$ **projection matrix**. Depth $Z_c$ shows up implicitly in $\lambda$; recovering consistent $Z$ from one image alone is **ill-posed** without additional cues—that is why multi-view stereo, structure from motion, and learned priors matter.

### Interactive: intrinsics vs extrinsics

Use the controls in the panel to separate **orbiting the demonstration** (your view of the scene) from **moving the camera rig** (extrinsics) and **changing focal length / principal point** (intrinsics). Read the on-screen equations while you drag the sliders.

<iframe src="camera_params.html" width="100%" height="520" style="border:0; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.15);" title="Interactive camera intrinsics and extrinsics"></iframe>

*Direct link:* [`camera_params.html`](camera_params.html).

### Math formulation summary

Given world point $\mathbf{X}_w = (X_w, Y_w, Z_w)^\top$:

1. **World to camera:** $\mathbf{X}_c = R \mathbf{X}_w + \mathbf{t}$.
2. **Perspective:** $x_n = X_c / Z_c$, $y_n = Y_c / Z_c$ (normalized image plane).
3. **Pixels:** $x_p = f_x x_n + s y_n + c_x$, $y_p = f_y y_n + c_y$.

Stacked, this is exactly multiplying $\tilde{\mathbf{X}}_w$ by $\mathbf{P} = \mathbf{K}[R \mid \mathbf{t}]$ and dividing by the third coordinate of the result.

**Calibration** estimates $\mathbf{K}$ (and distortion, not covered here) from images of known patterns. **Pose estimation** finds $[R \mid \mathbf{t}]$ relative to a known 3D model or map. Many deep systems learn **surrogate** representations (e.g., depth maps, MLP densities) but still **evaluate** with the same projection geometry when ground truth is available.

---

## Concept map (how the pieces connect)

1. **3D rigid motion** ($R$, $\mathbf{t}$) moves points or places the camera in the world.
2. **Intrinsics** ($\mathbf{K}$) turn viewing rays into **pixel indices**.
3. Together they form $\mathbf{P}$, which explains **why** the same object looks different at different scales and positions in the screen—a single matrix encapsulates “where the camera is” and “how it images.”

That decomposition is what you will reuse in depth estimation (predict $Z$ per pixel), reconstruction (fuse multiple $\mathbf{P}_i$), and end-to-end geometric networks (differentiable rendering often implements the same $\mathbf{P}$).
