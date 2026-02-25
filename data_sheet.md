# Machine Learning Datasheet
## Black-Box Optimization Capstone Project

Document Version: 1.0  
Last Updated: February 2026 (Week 11)  
Created By: Andrew Henson

---

## 1. Model Overview

Model Family: Ensemble Bayesian Optimization System

Primary Components:
- Gaussian Process Regressor with Automatic Relevance Determination (ARD)
- Random Forest Regressor (100 estimators, max_depth=5)
- Bayesian Optimization framework with Expected Improvement and Upper Confidence Bound acquisition functions

Model Purpose: Predict optimal input configurations for eight unknown black-box functions to maximize total output across all functions.

Training Data: 88 query-response pairs collected over 11 weeks (8 functions × 11 weeks)

Development Timeline:
- Weeks 1-2: Manual exploration
- Weeks 3-4: Initial GP models with 3-5 data points per function
- Weeks 5-7: GP with ARD, Random Forest feature importance
- Weeks 8-11: Hybrid empirical-model approach with noise characterization

Software: scikit-learn 1.3.0, NumPy 1.24.0, Python 3.10+

---

## 2. Intended Use

Primary Applications:
- Sequential optimization of unknown expensive-to-evaluate functions
- Query selection for black-box systems with limited evaluation budget
- Multi-objective optimization across heterogeneous function landscapes

Research and Educational Applications:
- Studying exploration-exploitation trade-offs in Bayesian Optimization
- Analyzing GP performance on functions with regime shifts
- Teaching optimization principles and adaptive resource allocation

Out-of-Scope Uses:
- High-dimensional optimization (over 10 dimensions) - models trained on 2-8D functions only
- Real-time optimization - sequential nature requires feedback delay
- Safety-critical applications - models failed catastrophically on regime shifts (Week 5: 40x unexpected jump)
- Guaranteed global optimum - provides local optimization with no convergence guarantees
- Noisy functions without noise modeling - standard GP failed on F2 (80% variance)
- Production deployment without validation - Week 8 showed models can fail catastrophically (negative 1742 points)

Specific Warnings:
- Do not use GP predictions at function boundaries without empirical validation
- Do not trust GP on regime-shift functions without regime-specific training
- Do not change variables with short ARD length scales (under 0.1) without extreme caution

---

## 3. Training Data

Dataset: BBO Capstone Sequential Query Dataset containing 88 query-response pairs across 8 functions over 11 weeks with progressive training (Week N uses data from Weeks 1 through N-1). No held-out validation data; all queries used for training with live evaluation each week.

Function Characteristics:
- F1 (2D): Output 0.0-0.526, deterministic, best 0.5259
- F2 (2D): Output negative 0.025-0.130, stochastic (80% variance), best 0.1300
- F3 (3D): Output negative 1.154 to negative 0.061, mostly deterministic (4% noise), best negative 0.0614
- F4 (4D): Output negative 30.1 to negative 3.986, deterministic, best negative 3.9857
- F5 (4D): Output 131.8-6526.4, deterministic, regime shift (low approximately 137, high approximately 6500), best 6526.44
- F6 (5D): Output negative 2.067 to negative 1.064, moderate noise (9%), best negative 1.0641
- F7 (6D): Output 0.00003-1.478, deterministic, best 1.4783
- F8 (8D): Output 4.18-9.692, deterministic, best 9.6921

Input Constraints: All dimensions in range 0 to 1 with 6 decimal precision. No duplicate queries allowed.

Label Distribution: F5 contributes 99.9% of total score (6526.44 out of 6532.80). Other 7 functions contribute 0.1% combined. Highly imbalanced optimization problem.

Preprocessing:
- F5 regime separation: Low-regime data (y under 1000) dropped from GP training to prevent kernel collapse
- Duplicate removal: Week 10 exact duplicates excluded from unique training set
- Output normalization: GP trained with normalize_y=True to handle scale differences

---

## 4. Model Performance

Primary Metric: Total output across all 8 functions

Performance Progression:
- Week 1 (Baseline): 150.0 points
- Week 5 (F5 Breakthrough): 5,575.3 points (positive 3,617%)
- Week 7 (Pre-x4 Peak): 6,164.7 points
- Week 8 (Disaster): 4,447.2 points (negative 27.8%)
- Week 11 (x4=1.0 Breakthrough): 6,532.8 points (all-time high)

Total Improvement: positive 6,382.8 points (positive 4,255% from baseline)

Gaussian Process Prediction Accuracy (Week 11):
- F3: GP predicted negative 0.058, actual negative 0.061 (5.2% error)
- F4: GP predicted negative 3.986, actual negative 3.986 (0.0% error)
- F5: GP predicted 6288 plus/minus 109, actual 6526 (3.8% error, within 2 sigma)
- F6: GP predicted negative 1.065, actual negative 1.069 (0.4% error)
- F8: GP predicted 9.68, actual 9.68 (0.0% error)

GP highly accurate for smooth deterministic functions (F4, F6, F8) but underestimated F5 and could not model F2 stochasticity.

Feature Importance (ARD Length Scales for F5 High-Regime):
- x1: 0.035 (critical boundary, must equal 1.0)
- x2: 0.182 (moderate sensitivity, optimum at 0.853)
- x3: 0.041 (critical boundary, must equal 1.0)
- x4: 0.256 (exploratory, tested up to 1.0)

Shorter length scale indicates higher importance. x1 and x3 are critical variables.

---

## 5. Major Failures and Limitations

Failure 1 - Week 5 Regime Shift: GP trained on F5 low-regime data (y approximately 137 plus/minus 5) predicted flat landscape with no improvement expected. Testing corner region gave output 5549.45 (40x jump). Root cause: GP cannot predict regime shifts when trained on one regime and tested on another. Mitigation: Regime-specific training by dropping low-regime data for high-regime GP.

Failure 2 - Week 8 ARD Misinterpretation: F5 x1 had shortest ARD length scale (0.035), misinterpreted as "tunable variable." Changed x1 from 1.0 to 0.855, causing output to crash from 6158 to 4416 (negative 1742 points, negative 28%). Root cause: Short length scale means "highly sensitive boundary, do not touch" not "optimize this." Mitigation: Single-variable isolation; never change critical variables.

Failure 3 - Week 11 F7 Overextrapolation: GP suggested x6=0.800+ might improve based on smooth extrapolation. Actual result at x6=0.760 was 1.316 (negative 11% from peak at 0.734). Root cause: GP assumes smoothness and cannot model sharp peaks followed by decline. Mitigation: Trust empirical patterns over GP extrapolation at boundaries.

Failure 4 - F2 Stochasticity: F2 showed 80% variance on identical inputs. Tried GP with noise kernel, Bayesian Ridge, Random Forest ensemble - all models failed. Mitigation: Abandon modeling and use best historical value (0.130).

Known Limitations:
- Regime shift blindness: GP cannot predict discontinuities or multi-modal landscapes
- Small sample size: 11 points per function insufficient for high-dimensional spaces
- No global convergence guarantee: Bayesian Optimization is greedy and can get stuck in local optima
- Boundary extrapolation failure: GP assumes smoothness and fails at sharp peaks
- Noise sensitivity: GP requires noise kernel for stochastic functions
- One-week feedback delay: Cannot rapidly iterate
- No duplicate queries: Cannot re-sample for noise reduction on stochastic functions
- Function-specific tuning: Optimal strategy for F5 does not work for F2
- Low transferability: Models trained on these 8 functions unlikely to generalize to new black-box functions

---

## 6. Recommendations

Strong Recommendations (High Confidence):
- Smooth deterministic functions with 2-6 dimensions
- Expensive-to-evaluate functions where query budget matters
- Functions with clear gradients and single global optimum
- Educational demonstrations of Bayesian Optimization

Conditional Recommendations (Requires Adaptation):
- Functions with regime shifts: Use regime-specific GP training
- Stochastic functions: Use noise kernel and budget for repeated samples
- High-dimensional functions (over 8D): Consider dimensionality reduction or trust region methods
- Multi-modal functions: Use multiple restarts or population-based methods

Not Recommended:
- Real-time optimization (one-week delay in capstone; models not designed for speed)
- Safety-critical applications (catastrophic failures observed)
- Guaranteed global optimum (no convergence proof)

---

## 7. Ethical Considerations

Data Collection Bias: Sequential active learning inherently biases toward high-performing regions. Low-performing regions under-sampled (F5 low-regime only 4 points out of 11).

Model Bias: F5 dominates total score (99.9%), causing models to over-optimize F5 and under-optimize others. Resource allocation strategy (75% effort on F5 and F7) intentionally neglects other functions. This is strategic given constraints but creates "abandoned function" bias.

Privacy and Security: Not applicable (synthetic black-box functions, no real-world data). Model could potentially be reverse-engineered to infer function structure. No personally identifiable information involved.

Environmental Impact: GP training approximately 0.1 seconds per function per week on standard CPU. Total compute under 1 hour over 11 weeks. Minimal environmental impact.

---

## 8. Key Takeaways

This Bayesian Optimization system achieved 4,255% improvement (150 to 6,533 points) over 11 weeks through ensemble modeling combining Gaussian Processes with ARD, Random Forest feature importance, and hybrid empirical-model decision making.

Critical Success Factors:
- Regime-specific GP training to avoid kernel collapse on F5
- ARD length scale interpretation for variable importance
- Single-variable isolation for causal attribution after Week 8 disaster
- Adaptive resource allocation (75% effort on improving functions)
- Hybrid approach: GP for smooth interpolation, empirical patterns for boundaries

Main Lessons:
- Models excel at interpolation but fail at predicting regime shifts
- Short ARD length scales indicate sensitivity, not tunability
- Empirical gap-filling (testing obvious untested boundaries like x4=1.0) can outperform model optimization
- Accept losses on stochastic or plateaued functions to focus resources
- Validate model predictions against empirical data, especially at boundaries

This datasheet documents both successes (positive 6,383 points) and failures (Week 8 negative 1,742 points) to guide future applications of Bayesian Optimization in similar expensive black-box optimization scenarios.

---

