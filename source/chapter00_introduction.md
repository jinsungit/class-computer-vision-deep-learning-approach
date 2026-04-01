# Introduction

Course overview, philosophy, and what we will (and will not) cover.

## Welcome


<br>

This course focuses on modern computer vision as *representation learning*, 
emphasizing *foundation models*
and *unified architectures* that work across diverse visual tasks.


<br>
<br>


We prioritize understanding:
- How visual representations are learned
- Why certain architectures work
- How to effectively use and adapt state-of-the-art pretrained models


## Visual tasks in a single image


Take a look at this picture, how many visual tasks can you identify? 

```{figure} _static/imgs/vision_tasks.PNG
:width: 80%
:alt: Example scene for visual tasks
```

---
*Computer Vision* is about using computers to understand vision.

## Computer
- Computational
- Algorithm
- Verifiable
- Universal


## Vision
- Human vision
- Biological vision
- Visual perception
- Visual understanding



### Typical computer vision tasks

- Classification, detection, segmentation
- Tracking, counting
- 3D: reconstruction, decomposition, rendering
- Generation: image and video
- Visual Q&A, colorization, super resolution, denoising, dehazing
- Human analysis: pose, shape, recognition, behavior, action, intention
- And more...

## Classic vs modern computer vision


### *Classic* Computer Vision

- Canny edge detection
- Viola-Jones face detection
- HoG and DPM for pedestrian detection
- SIFT
- Eigen face

### *Modern* Computer Vision
#### Deep learning era (2010+)

- CNN
- UNet
- Vision transformers
- Diffusion
- NeRF and Gaussian Splatting
- Foundation vision models


In parralel: Large Language Models (LLMs) and Large Multi-modality Models (LMMs)

This will be our focus!

## Core philosophy


We view neural networks as the native solution to vision problems. 
We fully embrace the neural network approach, taking a fundamental view of learning representations in the vision domain.


## What this means


- **Representation Learning**: Understanding how to encode visual information
- **Unified Models**: One architecture for many tasks
- **Scale Matters**: Large models, large datasets
- **Foundation Models**: Pretrained models you can adapt

## Methodology

For each computer vision task, we:

- learn the most **basic** solution in neural network approach
- learn the proper way to **train** and **evaluate** it
- look at the bells and whistles of the **best** neural network approach


## What We Explicitly Deprioritize

- Hand-crafted features: SIFT, HOG, and classical feature engineering
- Classical pipelines without learning: Traditional computer vision approaches
- Excessive historical surveying: We focus on what works today, not comprehensive history
