# Evaluation And Data Roles

## Reference Data

The study uses an outcome-joint-stratified 80:20 split (18,604 training and 4,651 held-out visits). All four main generators fit the training pool. Fidelity uses the preprocessed real training reference. Classifier testing uses the held-out records. DCR compares synthetic-to-training distances with held-out-to-training distances.

Hospitalization is admission within seven days of the index visit. ICU Admission occurs within the same clinical episode and implies Hospitalization. The ICU task uses hospitalized records only; neither outcome is used as a predictive feature for the two documented tasks.

## Metrics

- Univariate fidelity: relative errors in continuous summaries, KS statistics and standardised Wasserstein distances; categorical coverage, TVD and Jensen-Shannon distance. Reported aggregate relative errors are means of per-variable relative errors.
- PWCC: Pearson correlation, Cramer's V or correlation ratio according to variable types. The summary is mean absolute error over valid variable pairs.
- CCD: continuous conditional means and SDs at each observed Age in Days value, standardised using real-reference SDs. Overall distances pool valid feature-age points. A split-half training reference supplies real-versus-real comparisons.
- Global structure: UMAP projections and local mixing summaries.
- Utility: logistic regression, random forest and XGBoost under TRTR and TSTR on the fixed real holdout.
- Empirical privacy: DCR and nearest-neighbour attribute inference. These are protocol-specific assessments, not formal privacy guarantees.

Missing categorical states use `__MISSING__` in evaluation outputs. Missing symptoms must not be recoded as absence.

## Sensitivity Analysis

In mixed training, dataset A contains 1,000 real classifier records and disjoint dataset B contains 17,604 generator-training records. Classifiers use A alone or A plus synthetic data at 0.5x and 1x through 10x A's size.

In synthetic-only training, the generator fits the full 18,604-record real training pool. Classifiers use only the generated data at 100-50,000 records. A supplies the real-only comparator. Requested sizes precede ICU-specific filtering. Summary means and SDs use successful repeats among the 100 scheduled repeats; repeat statuses are retained.

## Generated Outputs And Public Data

The code writes synthetic tables, run configurations, split and generation metadata, and evaluation outputs. Those runtime outputs are not bundled in this code folder.

Runtime evaluation can produce record-level diagnostics that are excluded from the public data record: CCD `details.csv`, DCR `details.csv`, attribute-inference `details.csv` and UMAP `embedding.csv`. Only the six all-variable CCD plots per run are included in the public data record. Other retained outputs include real-reference summaries and per-point LISI statistics.

The public data record's `docs/results_guide.pdf` and `file_manifest.csv` describe the released subset. The full ablation setting reuses the main Copula run, so its files are not independent replicates.
