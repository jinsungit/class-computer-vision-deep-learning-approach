## Bias and fairness in vision systems

### Where bias comes from

**Bias** in ML usually means **systematic** errors or outcomes that are **skewed** with respect to groups, contexts, or individuals—not random noise. In computer vision, bias often enters through:

1. **Data collection:** who appears in images, where cameras are placed, which products are photographed, which languages appear in text–image pairs.
2. **Annotation:** subjective labels (emotion, attractiveness, “suspicious” behavior), **rater demographics**, and **shortcut** cues (hospital ID in chest X-rays).
3. **Modeling choices:** optimizing **average** accuracy can hide **poor** performance on **tail** classes or subgroups.
4. **Deployment:** thresholds, camera placement, and **human-in-the-loop** workflows amplify or mitigate technical bias.

Fairness is not a single formula: different **definitions** (demographic parity, equalized odds, calibration across groups) can **conflict**. Stakeholders must decide which **normative** trade-offs are acceptable in a given domain (hiring, policing, healthcare).

---

### Vision-specific failure modes

| Area | Risk | Example pattern |
|------|------|------------------|
| **Face analysis** | Unequal error rates by **skin tone**, **gender presentation**, age | Higher false match or misclassification for some groups |
| **Object / scene classifiers** | **Co-occurrence** shortcuts | “Kitchen” ↔ certain cultures or layouts in training data |
| **Medical imaging** | **Site** or **scanner** as a spurious correlate | Model uses hospital metadata instead of pathology |
| **Generative models** | **Stereotypes** in text-to-image | Prompts like “architect” vs “housekeeper” can yield **occupational** and **gendered** imagery that reflects web priors, not reality |

The figures below illustrate **stereotype pressure** in a text-to-image system: different prompts can produce imagery that encodes **social associations** learned from training data. That is an **ethical** issue for product design, content policy, and **who** is harmed when such systems are deployed at scale.

```{figure} https://libapps-ca.s3.amazonaws.com/accounts/159843/images/SD_Architect.jpg
:width: 45%
:alt: Example generated image for the prompt architect

Example: **architect** (generated image; illustration from course materials).
```

```{figure} https://libapps-ca.s3.amazonaws.com/accounts/159843/images/SD_Housekeeper.jpg
:width: 45%
:alt: Example generated image for the prompt housekeeper

Example: **housekeeper** (generated image; illustration from course materials).
```

---

### Mitigations (high level)

- **Audit** performance **per subgroup** when attributes or proxies exist; document **limitations** when they do not.
- **Balance** or **reweight** training data cautiously—blind balancing can remove **legitimate** base rates in some tasks; domain expertise matters.
- **Robust training** (group DRO, subsampling, contrastive debiasing) and **post-hoc** calibration or **thresholding** per group where appropriate and legal.
- **Participatory** design with **affected communities** rather than purely technical fixes.

**Takeaway:** “debiasing” is not a one-click layer; it requires **clarifying harms**, **measuring** disparities honestly, and **governing** deployment—not only improving a global accuracy number.
