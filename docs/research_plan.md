# Research fit and next experiments

## Credible positioning
This project demonstrates experimental design, data provenance, duplicate handling, baseline comparison, model selection without test tuning, uncertainty estimation, error analysis and reproducible Python engineering. It is an established-dataset benchmark, not a novel food safety detector. The AI-assisted code and report should be understood, checked and extended by the applicant before being discussed in applications or interviews.

A defensible project description: "Built and evaluated an AI-assisted, reproducible pipeline for sensory-quality grading from food chemistry measurements; compared four models using grouped cross-validation and documented minority-class errors and limits to generalisation."

## Focused next research question
Does an ordinal model improve higher-grade recall and reduce two-band errors without sacrificing macro-F1 on independently held-out production batches?

1. Freeze this benchmark as version 1. Explore ordinal regression or score regression inside nested cross-validation on training data. Use paired out-of-fold comparisons and confidence intervals; do not repeatedly optimise against the existing test set.
2. Obtain an external dataset with batch, winery and collection dates. Hold out entire batches and the latest period. Evaluate color-specific models and pooling with wine type, reporting subgroup supports and uncertainty.
3. Collect repeated sensory assessments to quantify inter-rater variability and investigate errors near score boundaries. Preserve label provenance and predefine binning alternatives.
4. If adding images, link each image to physical sample IDs and acquisition sessions. Split by sample/batch before augmentations; compare a frozen pretrained image encoder with tabular-only and combined models. Do not train/test on different photographs of the same sample.
5. For a genuine spoilage study, prospectively collect storage time and temperature with relevant validated spoilage endpoints. For contamination risk, obtain laboratory-confirmed pathogen/toxin outcomes and organism-specific thresholds. Sensory scores cannot substitute for microbiological labels.

## Suggested prospective design
Specify food, target organism or spoilage endpoint, measurement protocol, label assay, batch unit and intended use with a supervisor before collection. Reserve independent batches for final validation. Fit preprocessing only on training batches, handle prevalence explicitly, and measure sensitivity, specificity, precision-recall performance and probability calibration for the actual endpoint. Choose any operating threshold on validation data based on the consequence of errors, then freeze it. Determine sample size using prevalence and desired sensitivity precision rather than promising a performance target in advance.

## Interview preparation
Be able to explain why the majority baseline has 43.7% accuracy but only 0.203 macro-F1, why accuracy hides higher-grade misses, why deduplication changes the estimand, why feature importance is not causation, and why an internal bootstrap does not demonstrate external validity.
