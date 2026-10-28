# Classical Machine Learning for Motor Imagery Decoding

**Status:** Planned — no implementation or new benchmark has been executed in this subproject.

## Research question

How do classical feature extraction and machine learning methods compare for EEG-based motor imagery decoding under a leakage-safe, reproducible evaluation protocol?

## Candidate methods

Band-power features, Common Spatial Patterns (CSP), regularised CSP, filter-bank CSP (FBCSP), covariance-based Riemannian approaches, and appropriate machine learning baselines such as LDA and SVM. These are planned candidates, not validated findings.

## Candidate data and evaluation

Begin by checking suitability of the repository's existing EEG recordings. Additional public motor imagery datasets may be considered only after their specifications and conditions have been verified. Any supervised feature extraction, feature selection and model tuning must occur within the appropriate training folds.

## Intended outputs

Explanatory notebooks, reusable training and evaluation modules, reproducible baseline comparisons and clearly reported limitations. No results are asserted yet.

## Next stage

Agree a precise dataset, class definitions, split strategy, baseline model and the first notebook plan. Historical code remains unchanged unless a specific reuse action is approved.
