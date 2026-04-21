## Responsible practice and governance

### Documentation: datasheets and model cards

**Datasheets for datasets** and **model cards** are structured disclosures: who collected data, **for what purpose**, **known biases**, **recommended uses**, and **out-of-scope** applications. They do not replace ethics review, but they **force** teams to articulate assumptions and **limitations** before release.

For vision projects, useful sections include: **sensor** and **geography** of capture, **label definitions**, **demographic** statistics if collected, **license**, and **update** cadence when the world changes.

---

### Transparency and contestability

- **Explainability:** saliency maps and attention overlays can be **misleading** “**UI theater**”; simpler **counterfactual** checks (“what if skin tone proxy changes?”) sometimes teach more. Still, **some** interface for **appeal** when an automated decision affects people is ethically important.
- **Versioning:** ship **model IDs**, **training data** snapshots (where legal), and **changelog** so failures can be **traced** and **rolled back**.

---

### Environmental and social cost

Training large models consumes **energy** and **hardware**; **inference** at planetary scale dominates **lifetime** carbon for many products. Responsible practice includes **efficient** architectures, **quantization**, **caching**, and **right-sizing** models for the task—not always the largest checkpoint.

**Labor:** credit and **compensation** for annotators, **community** benefit when using public or indigenous imagery, and **downstream** accountability when **open weights** are misused are active debates in policy and open-source norms.

---

### Governance inside organizations

Many universities and companies use **IRB** / ethics review for human subjects, **legal** review for biometric and **copyright** risk, and **AI ethics boards** with **veto** or **escalation** paths. For student projects: default to **public** datasets with **clear licenses**, avoid **scraping** identifiable people without protocol, and **document** limitations in the README.

**Takeaway:** responsible AI is **process**—documentation, **stakeholder** input, **monitoring** after launch, and **humility** about what metrics cannot measure.
