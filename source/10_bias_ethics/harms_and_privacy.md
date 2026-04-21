## Harms, privacy, and misuse

### Categories of harm

Vision systems can cause **individual** and **collective** harms even when benchmark accuracy is high:

- **Surveillance and tracking:** face recognition, **gait**, **re-identification** across cameras, and **license-plate** pipelines enable **mass monitoring**; risks rise when there is **no consent**, **no transparency**, or **disproportionate** targeting of neighborhoods or groups.
- **Manipulation and non-consensual imagery:** **deepfakes**, **face swap**, and **nudification** tools have been used for harassment, fraud, and **reputation** attacks.
- **Automated decisions:** lending, **border control**, **employment**, and **criminal justice** interfaces that use vision signals can **scale** errors and **encode** historical discrimination.
- **Labor and annotation economies:** large datasets depend on **precarious** workers, **content moderators**, and **clickworkers**; **informed consent** for biometric or sensitive images is often weak.

Ethical review asks not only “does it work?” but “**who benefits**, **who bears risk**, and **who was asked**?”

---

### Privacy concepts for vision

- **Data minimization:** collect and retain only what is **necessary** for the stated purpose; **retention** limits and **deletion** paths matter.
- **Re-identification:** even “anonymized” **face crops** or **unique** gait signatures can often be linked back to individuals when combined with auxiliary data.
- **Membership inference** and **model inversion:** strong models can **leak** training-set information; **differential privacy** during training trades utility for a formal **bound** on leakage (used in some large-scale training pipelines, not universal).

**Consent** for **biometric** processing is regulated in many jurisdictions (e.g. GDPR in Europe, BIPA in Illinois, sector-specific rules elsewhere). **Compliance** is necessary but not sufficient for **ethical** deployment.

---

### Misuse and dual-use

The same **foundation models** that help accessibility (e.g. image captioning) can accelerate **spam**, **scams**, and **synthetic propaganda**. **Guardrails** (refusal training, classifiers, provenance metadata such as **C2PA** where adopted) are partial mitigations; **adversaries** adapt.

**Takeaway:** privacy and misuse are **systems** problems—law, **product policy**, **logging**, **appeals**, and **human oversight** must sit alongside the neural network.
