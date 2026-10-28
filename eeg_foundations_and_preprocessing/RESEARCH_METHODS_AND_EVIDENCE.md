# Research Methods and Evidence

**Study:** EEG Foundations, Signal Processing and Motor Imagery Analysis  
**Status:** Initial planning — experimental evidence has not yet been generated in this new project.

## Research objective

Establish an understandable and reproducible EEG motor imagery workflow, beginning with data validation, exploratory signal analysis and scientifically justified preprocessing before model-based decoding.

## Existing sources requiring verification

The historical repository contains MATLAB files in `../Data/` and `../Code/`, Python implementations in `../Code/`, and papers in `../PDF/`. These are pre-existing sources, not outputs of this new subproject. Keep them intact and check file schema, subject/session identity, classes, trial definitions, channel layouts, provenance and usage permissions before relying on them.

## Planned methodology

1. Inspect available source files and document their actual structure and metadata.
2. Explain EEG/BCI/MI principles with example signal visualisations.
3. Evaluate signal quality and recording characteristics, and assess trial boundaries and sampling parameters.
4. Introduce spectral and time-frequency representations, including ERD/ERS where the dataset supports meaningful analysis.
5. Implement and compare selected temporal and spatial preprocessing methods with clear parameters and justified reference choices.
6. Record assumptions, limitations, validation checks and evidence for each approved stage.

## Evaluation safeguards

- Separate descriptive signal analysis from supervised decoding experiments.
- Fit data-dependent transforms exclusively on permitted training partitions in any later model benchmark.
- Respect subject, session and trial independence according to the chosen evaluation question.
- Distinguish the effects of preprocessing from changes in classifier design.
- Do not infer generalisation or improved accuracy without a controlled, actually executed experiment.

## Initial dependencies and risks

- The legacy MATLAB data schema and class definitions need verification.
- Some legacy paths and code may depend on a particular local execution environment.
- Signal processing choices may alter physiologically meaningful activity.
- Any reused legacy code requires separate, specific approval before copy, migration or adaptation.

## Evidence record

No new experiments, metrics or figures are claimed at this stage. Add only actual executed configurations, verified observations, comparisons and limitations as development proceeds.
