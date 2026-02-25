# Model Card
## Black-Box Optimization Capstone Project

Last Updated: February 2026 (Week 11)
Model Version: 1.0

---

## Model Details

Developed by: Andrew Henson
Model Date: February 2026 (Week 11 of 13-week project)
Model Type: Ensemble Bayesian Optimization System
Model Version: 1.0

The model is an ensemble system combining Gaussian Process Regression with Automatic Relevance Determination (ARD), Random Forest for feature importance, and Bayesian Optimization with Expected Improvement and Upper Confidence Bound acquisition functions. The system sequentially optimizes eight unknown black-box functions with different dimensionalities (2D to 8D) to maximize total output across all functions.

License: Educational/Research Use Only
Contact: Andrew Henson

---

## Intended Use

Primary Intended Uses:
The model is designed for sequential optimization of expensive-to-evaluate black-box functions where query budget is limited. Ideal applications include hyperparameter tuning, engineering design optimization, and AutoML scenarios where each function evaluation has real cost (time, money, or resources). The system works best on smooth deterministic functions with 2-6 dimensions and clear gradients.

Primary Intended Users:
Machine learning researchers, optimization practitioners, and students learning Bayesian Optimization principles. Users should have understanding of Gaussian Processes, acquisition functions, and exploration-exploitation trade-offs.

Out-of-Scope Use Cases:
Not intended for high-dimensional optimization beyond 10 dimensions (tested only on 2-8D), real-time optimization requiring immediate feedback (system designed for delayed evaluation), safety-critical applications where catastrophic failures are unacceptable, noisy/stochastic functions without noise kernel adaptation, or production deployment without extensive validation and human oversight.

The model failed catastrophically on regime-shift functions (Week 5: 40x unexpected jump) and on sensitive boundary changes (Week 8: negative 1,742 points from single variable change). Do not use GP predictions at function boundaries without empirical validation. Do not change variables with ARD length scales below 0.1 without extreme caution.

---

## Factors

Relevant Factors:
The model's performance depends heavily on function characteristics including dimensionality (2D to 8D tested), determinism versus stochasticity (F2 showed 80% variance rendering all models useless), presence of regime shifts (F5 jumped 40x from low regime approximately 137 to high regime approximately 6,500), smoothness versus sharp peaks (F7 peak at x6=0.734 with decline on both sides), and optimization landscape (single optimum versus multi-modal).

Evaluation Factors:
Model evaluated across heterogeneous function types: smooth quadratic bowl (F4), stochastic noisy (F2), regime-shift dominant (F5), multi-dimensional sparse signal (F1, F8), and boundary-sensitive (F7). Performance measured by total score improvement over 11 weeks and individual function optimization success.

---

## Metrics

Model Performance Metrics:
Primary metric is total output summed across all eight functions. Secondary metrics include individual function improvement versus baseline, GP prediction accuracy (predicted versus actual output), and ARD length scale analysis for variable importance.

Decision Thresholds:
GP predictions used when uncertainty (2-sigma confidence interval) is below 20% of predicted value. Acquisition function switches from exploration (UCB with beta=2.0) to exploitation (Expected Improvement) after 7 weeks of data collection. Variables with ARD length scale below 0.1 classified as critical and excluded from optimization attempts.

Variation Approaches:
Performance varies significantly by function type. Deterministic functions (F1, F4, F5, F7, F8) show consistent GP prediction accuracy within 5%. Stochastic functions (F2) show 80% variance requiring noise kernel or model abandonment. Functions with regime shifts (F5) require regime-specific training to avoid kernel collapse.

---

## Training Data

Dataset: 88 query-response pairs collected sequentially over 11 weeks (8 functions times 11 weeks). Each week, one query submitted per function with one-week delay for response.

Preprocessing: For F5, low-regime data (output below 1,000) dropped from high-regime GP training to prevent kernel collapse. Week 10 exact duplicates removed from training set. All inputs normalized to range 0 to 1 by design (constraint of problem). GP trained with normalize_y=True to handle output scale differences (F5 outputs 6,526 while F3 outputs negative 0.06).

---

## Evaluation Data

Training and Evaluation: Progressive training where Week N model uses data from Weeks 1 through N minus 1. No held-out validation set due to limited query budget (only 88 total queries). Each submitted query serves as live test point.

Evaluation performed weekly by comparing predicted optimal input versus actual response. F5 Week 11 prediction: 6,288 plus/minus 109, actual: 6,526 (within 2-sigma, 3.8% error). F4, F6, F8 predictions within 0.5% of actual. F2 predictions unreliable due to stochasticity (80% variance on identical inputs).

---

## Quantitative Analyses

Unitary Results:
Total score progression: Week 1 baseline 150.0 points, Week 5 breakthrough 5,575.3 points (positive 3,617% improvement), Week 7 pre-x4 peak 6,164.7 points, Week 8 disaster 4,447.2 points (negative 27.8% regression), Week 11 x4=1.0 breakthrough 6,532.8 points (all-time high). Total improvement: positive 6,382.8 points representing 4,255% gain from baseline.

Individual function best results: F1 0.5259 (Week 11), F2 0.1300 (Week 5), F3 negative 0.0614 (Week 11), F4 negative 3.9857 (Week 8), F5 6,526.44 (Week 11, contributing 99.9% of total score), F6 negative 1.0641 (Week 8), F7 1.4783 (Week 7), F8 9.6921 (Week 7).

Intersectional Results:
F5 dominance creates imbalanced optimization where F5 contributes 6,526 points while other seven functions contribute only 6.36 points combined. Resource allocation strategy adapted: 75% effort on F5 plus F7 (still improving), 25% on others (plateaued or stochastic). This intentional bias maximizes total score but abandons optimization of low-value functions.

GP prediction accuracy varies by function characteristics. Deterministic smooth functions (F4, F6, F8) show under 1% prediction error. Functions with boundaries or peaks (F7) show 5-10% error when extrapolating. Regime-shift functions (F5) underestimate by 3.8% even with regime-specific training. Stochastic functions (F2) render GP predictions useless (80% variance overwhelms signal).

---

## Ethical Considerations

The model exhibits strong optimization bias toward F5 (99.9% of score) resulting in resource allocation strategy that intentionally abandons optimization of seven other functions. This is strategic given problem constraints but creates "function abandonment" bias where F2, F4, F6, F8 receive minimal optimization effort after Week 7.

Sequential active learning inherently biases sampling toward high-performing regions. F5 low-regime (output approximately 137) only sampled 4 times out of 11 weeks while high-regime (output 4,416 to 6,526) sampled 7 times. Alternative optima in under-explored regions may exist but remain undiscovered.

No privacy, security, or fairness concerns as dataset consists of synthetic black-box functions with no real-world data, personally identifiable information, or protected attributes. Computational cost minimal (under 1 hour total over 11 weeks, approximately 0.1 seconds per GP training iteration).

Model failures documented for transparency: Week 5 regime shift (GP failed to predict 40x jump), Week 8 catastrophic variable change (negative 1,742 points from misinterpreting ARD length scale), Week 11 F7 boundary extrapolation (negative 11% from peak when GP suggested improvement), F2 stochasticity (all models failed, abandoned modeling entirely).

---

## Caveats and Recommendations

Known Limitations:
Regime shift blindness: GP cannot predict discontinuities or transitions between performance regimes. Small sample size: 11 points per function insufficient for reliable modeling in high dimensions (F8: 8D). Boundary extrapolation failure: GP assumes smoothness and fails at sharp peaks or discontinuities. No global convergence guarantee: Bayesian Optimization provides local optimization with possibility of getting stuck in suboptimal regions.

Recommendations:
Use this model for smooth deterministic functions with 2-6 dimensions, expensive evaluations where query budget matters, and single-optimum landscapes. Adapt the approach for regime-shift functions by training regime-specific GPs. For stochastic functions, add noise kernel to GP or increase query budget for repeated sampling. For high-dimensional functions beyond 8D, consider dimensionality reduction or trust region methods.

Do not use for real-time optimization (one-week feedback delay in original problem), safety-critical applications without extensive validation (catastrophic failures observed), or scenarios requiring guaranteed global optimum (no convergence proof).

Critical lessons: Short ARD length scales indicate variable sensitivity not tunability (do not optimize variables with length scale below 0.1). Trust empirical patterns over GP extrapolation at boundaries. Implement single-variable isolation for causal attribution. Accept losses on stochastic or plateaued functions to focus resources on improvements. Validate all model predictions against empirical data before deployment.

---

## Model Performance

GP Prediction Accuracy (Week 11):
- F3: Predicted negative 0.058, Actual negative 0.061 (5.2% error)
- F4: Predicted negative 3.986, Actual negative 3.986 (0.0% error)  
- F5: Predicted 6,288 plus/minus 109, Actual 6,526 (3.8% error, within 2-sigma)
- F6: Predicted negative 1.065, Actual negative 1.069 (0.4% error)
- F8: Predicted 9.68, Actual 9.68 (0.0% error)

Feature Importance (F5 ARD Length Scales):
- x1: 0.035 (critical boundary, must equal 1.0)
- x2: 0.182 (moderate sensitivity, optimum at 0.853)  
- x3: 0.041 (critical boundary, must equal 1.0)
- x4: 0.256 (exploratory, tested boundary at 1.0 in Week 11)

Shorter length scale indicates higher variable importance. x1 and x3 are critical non-tunable boundaries.

Major Failures:
Week 5: GP trained on F5 low regime (approximately 137) predicted no improvement. Actual result 5,549 (40x jump due to regime shift to high regime). Week 8: Misinterpreted ARD short length scale as "optimize this variable." Changed x1 from 1.0 to 0.855, output crashed from 6,158 to 4,416 (negative 1,742 points). Week 11 F7: GP suggested x6=0.800 based on smooth extrapolation. Actual result at x6=0.760 was 1.316 (negative 11% from peak at 0.734). F2 all weeks: Stochasticity (80% variance) rendered all models useless, abandoned modeling approach.

Success Factors:
Regime-specific GP training (drop F5 low-regime data). ARD length scale interpretation for identifying critical variables. Single-variable isolation after Week 8 for causal attribution. Adaptive resource allocation (75% on F5 plus F7). Hybrid empirical-model approach using GP for smooth interpolation and empirical patterns for boundaries. Conservative step sizes in late weeks (0.01 to 0.02 range). Accepting losses on F2 stochastic function rather than wasting queries.

