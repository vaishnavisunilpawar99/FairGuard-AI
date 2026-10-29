# FairGuard AI — Model Card
*Generated: 2026-10-28 | Standard: Mitchell et al. (2019) Model Cards*

---

## 1. Model Details

| | Model A (Unmitigated) | Model B (Fairness-Aware) |
|---|---|---|
| **Type** | HistGradientBoostingClassifier | ExponentiatedGradient + LogisticRegression |
| **Constraint** | None | Demographic Parity |
| **Framework** | scikit-learn | Fairlearn |
| **MLflow Run** | Model_A_Unmitigated | Model_B_FairnessMitigated |

---

## 2. Intended Use

**Primary Use:** Income bracket classification for demonstrating responsible AI evaluation techniques.

**Out-of-scope Use:** Real hiring, lending, or credit decisions without additional human review, bias auditing, and regulatory compliance checks. Model outputs MUST NOT be used to draw conclusions about protected groups.

---

## 3. Factors

**Sensitive Attributes Evaluated:** `sex`, `race`, `age_bin`

**Intersectional Groups:** `sex × race` combinations evaluated separately.

---

## 4. Performance Metrics

| Metric | Model A | Model B |
|--------|---------|---------|
| Accuracy | 0.84 | 0.785 |
| AUC-ROC | 0.8921 | 0.9156 |
| F1 Score | 0.6863 | 0.6767 |

---

## 5. Evaluation Data

- **Dataset:** UCI Adult Census Income (fetch_openml, version 2)
- **Split:** 80% train / 20% test (stratified by income)
- **Sample:** Up to 20,000 rows after NaN removal
- **Features:** 14 original features; age binned to `<30 / 30-50 / >50` for group analysis

---

## 6. Fairness & Privacy Findings

### Fairness Summary

| Metric | Model A | Status A | Model B | Status B |
|--------|---------|----------|---------|----------|
| Demographic Parity Diff | 0.2254 | Fail | 0.4012 | Fail |
| Equalized Odds Diff | 0.2434 | Fail | 0.6337 | Fail |
| Disparate Impact | 0.3038 | Fail | 0.2463 | Fail |
| Error Rate Disparity | 0.1123 | Warning | 0.1442 | Warning |

### Privacy Risk (Model A)

| Indicator | Value |
|-----------|-------|
| Min k-Anonymity (k) | 1 |
| Rows with k < 5 | 100.0% |
| MIA Train/Test AUC Gap | 0.1079 |
| Membership Inference Risk | **High** |

### Data Drift

| Indicator | Value |
|-----------|-------|
| Share of Drifted Features | 0.1667 |

## 7. Explainability

SHAP global and local explanations are available in the dashboard. Explanations describe model behaviour for individual records and feature groups; they do not describe people or groups.

## 8. Privacy

Privacy indicators include k-anonymity and a membership-inference proxy. These are screening measures and require review before deployment.

## 9. Drift

Evidently compares the training reference data with a shifted test sample. The generated drift report is retained with the artifacts.

## 10. Limitations

- UCI Adult data reflects 1994 US Census income patterns and may not generalise.
- Metrics depend on the selected sample, threshold, and sensitive attribute.

## 11. Ethical notes

This workbench reports measured model behaviour. It must not be used as the sole basis for decisions about individuals.

## 12. Approval status

Approval is pending until fairness, privacy, drift, legal, and governance checks are reviewed by an assigned reviewer.

---

## Fairness–Accuracy Trade-off Analysis

Model B applies the **Exponential Gradient** reduction with a **Demographic Parity** constraint, which directly optimises for equal selection rates across sensitive groups. This typically results in a measurable drop in raw accuracy and AUC compared to the unconstrained Model A, because the model is constrained away from patterns that, while predictive, correlate with protected attributes.

**Why Demographic Parity and Equalized Odds can conflict:** Demographic Parity requires equal *selection rates*, while Equalized Odds requires equal *true positive rates and false positive rates*. These objectives can conflict in a classifier, so FairGuard AI reports both measured behaviours and leaves the trade-off decision to human stakeholders.

---

## 8. Caveats

- UCI Adult data reflects 1994 US Census income patterns; distributions may not generalise.
- Fairness metrics are sensitive to how groups are defined and which sensitive attribute is chosen as the primary axis.
- SHAP values on Model B (mitigated) may differ significantly from Model A because the underlying model family changed.
- k-Anonymity is a weak privacy guarantee; re-identification risk may still exist.

---

## 9. Mitigations

- **Pre-processing:** Fairlearn sample reweighing (if enabled).
- **In-processing:** Exponential Gradient with Demographic Parity (Model B).
- **Post-processing:** Threshold calibration per group (recommended next step).
- **Monitoring:** Evidently AI drift reports on production data.
- **Human Oversight:** Sign-off checklist enforced before deployment.

---

## 10. Sign-Off Checklist

- [ ] Fairness audit reviewed by responsible AI lead
- [ ] All metrics above threshold reviewed and documented
- [ ] Drift report reviewed; no significant distributional shift confirmed
- [ ] Privacy risk assessed; k-anonymity deemed acceptable
- [ ] Model Card approved by data governance team
- [ ] Legal / regulatory review completed
- [ ] Deployment approved

---
*FairGuard AI · Responsible AI Assessment Workbench*
