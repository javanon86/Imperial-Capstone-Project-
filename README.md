# Black-Box Optimization Capstone Project

## Table of Contents
1. [Project Overview](#project-overview)
2. [Inputs and Outputs](#inputs-and-outputs)
3. [Challenge Objectives](#challenge-objectives)
4. [Technical Approach](#technical-approach)
5. [Results and Achievements](#results-and-achievements)
6. [Key Learnings](#key-learnings)

---

## Section 1: Project Overview

### Project Description
The Black-Box Optimization (BBO) capstone project is a 13-week function maximization challenge involving eight multi-dimensional unknown functions. Each week, I submitted input queries and received performance feedback, using machine learning to iteratively improve outputs.

### Overall Goal
The goal is to use increasingly sophisticated machine learning models to optimize input variable queries and achieve maximum output from unknown black-box functions. As data accumulates weekly, models learn patterns and improve prediction accuracy, guiding smarter query selection.

### Real-World Relevance
Black-box optimization is fundamental to many real-world ML applications:
- **Hyperparameter tuning**: Optimizing model configurations without knowing the underlying performance landscape
- **Drug discovery**: Finding optimal molecular structures without complete biochemical understanding
- **Engineering design**: Optimizing complex systems (aircraft wings, engines) where simulations are expensive
- **A/B testing**: Maximizing user engagement metrics with limited experiments
- **Automated machine learning (AutoML)**: Searching neural architecture spaces efficiently

This project teaches essential skills for working with unknown, expensive-to-evaluate systems where data is scarce and each query has real cost.

### Career Impact
This capstone has strengthened critical professional capabilities:
- **Strategic planning**: Managing 13-week timeline with limited queries (11 submissions per function = 88 total experiments)
- **Model selection under uncertainty**: Choosing appropriate ML techniques as data accumulates
- **Iterative refinement**: Learning from failures (Week 8's x1=0.855 disaster) and adapting strategy
- **Resource allocation**: Focusing 75% of effort on high-value functions (F5) while accepting losses on others (F2)
- **Empirical vs. model-driven decisions**: Knowing when to trust data patterns over model predictions
- **Technical communication**: Documenting methodology, justifying decisions, explaining trade-offs

---

## Section 2: Inputs and Outputs

### Input Format
Each function receives a different number of input variables (dimensions):
- **F1, F2**: 2 dimensions (x1, x2)
- **F3**: 3 dimensions (x1, x2, x3)
- **F4, F5**: 4 dimensions (x1, x2, x3, x4)
- **F6**: 5 dimensions (x1, x2, x3, x4, x5)
- **F7**: 6 dimensions (x1, x2, x3, x4, x5, x6)
- **F8**: 8 dimensions (x1, x2, x3, x4, x5, x6, x7, x8)

### Query Constraints
- All input values must be in range [0, 1]
- Values submitted with 6 decimal places
- Format: dimension values separated by hyphens (e.g., `0.410000-0.433000`)

### Example Queries

**Function 1 (2D):**
```
Week 8:  0.405000-0.428000  →  Output: 0.462531
Week 11: 0.410000-0.433000  →  Output: 0.525860
```

**Function 5 (4D) - The Breakthrough:**
```
Week 7:  1.000000-0.853000-1.000000-0.977000  →  Output: 6158.08
Week 11: 1.000000-0.853000-1.000000-1.000000  →  Output: 6526.44
```

### Output Format
Each function returns a single continuous numerical value representing performance. Higher values indicate better optimization (maximization problem). The total weekly score is the sum of all eight function outputs.

### Performance Tracking
Success is measured by:
1. **Individual function improvement**: Comparing each function's output to historical best
2. **Total weekly score**: Sum of all eight outputs (current best: 6532.80 in Week 11)
3. **Cumulative progress**: Overall improvement from Week 1 baseline (150 points) to current state (6533 points)

---

## Section 3: Challenge Objectives

### Primary Goal
Achieve the highest possible total output across all eight functions by Week 13 (final submission) through strategic query selection guided by machine learning models and empirical analysis.

### Optimization Type
**Maximization problem**: All functions reward higher output values. The challenge is finding the global maximum in each function's unknown landscape with minimal queries.

### Constraints and Limitations

**Query Budget:**
- 13 weeks total (11 weeks completed)
- 1 query per function per week
- 88 total queries across all functions (8 functions × 11 weeks)
- Each query is irreversible and expensive

**Operational Constraints:**
- Input values constrained to [0, 1] with 6 decimal precision
- No duplicate submissions allowed (portal rejects exact repeats)
- One-week delay between submission and results
- Unknown function structure (black-box)

**Strategic Challenges:**
- **Regime shifts**: Some functions have multiple performance regimes (F5: low-regime ~137, high-regime ~6500)
- **Stochasticity**: Some functions are noisy (F2 shows ±80% variance on identical inputs)
- **Dimensionality**: High-dimensional functions (F8: 8D) require exponentially more data for coverage
- **Exploration-exploitation trade-off**: Limited queries force choice between finding new regions vs. refining known peaks

---

## Section 4: Technical Approach

### Evolution of Strategy (Weeks 1-12)

#### **Weeks 1-2: Pure Exploration**
- **Methods**: Random sampling, grid-based exploration, corner testing
- **Results**: Discovered F4 is a simple quadratic bowl, F5 stuck in low regime

#### **Weeks 3-4: Introduction to ML Models**
- **Methods**: Gaussian Processes with RBF and Matérn kernels, Random Forest for feature importance
- **Key Learning**: GP requires 5+ points for reliability

#### **Week 5: The F5 Regime Discovery**
- **Result**: Testing [0.99, 0.90, 0.98, 0.93] jumped from 137 → 5549 (40× increase)
- **Strategic Pivot**: F5 became 99% of total score

#### **Weeks 6-7: Model Refinement**
- **Methods**: GP with ARD, Bayesian Optimization with Expected Improvement and UCB
- **Results**: Found F5 optimum at x2=0.853, F7 peak at x6=0.734

#### **Week 8: Catastrophic Model Misinterpretation**
- **Mistake**: Changed F5 x1 from 1.0 → 0.855
- **Result**: Output crashed from 6158 → 4416 (−1742 points)
- **Lesson**: Short ARD length scale means "don't touch"—it's critical, not tunable

#### **Week 11: Empirical Gap-Filling**
- **Breakthrough**: Tested x4=1.0 (had tested 0.977, 0.979, 0.988 but never 1.0)
- **Result**: 6526 (+368 points), largest single-variable gain

#### **Week 12: Regime-Shift Hypothesis Testing**
- **Test**: x2=0.920 with x4=1.0 to see if regime boundaries shifted
- **Risk Management**: 6526 floor locked from Week 11

### Current ML Toolkit

**Primary Models:**
1. **Gaussian Processes**: ARD for variable importance, noise modeling for stochastic functions
2. **Random Forest**: Feature importance for regime discovery
3. **Bayesian Optimization**: Expected Improvement, Upper Confidence Bound acquisition

**Abandoned Approaches:**
- Neural Networks (insufficient data), SVMs (less interpretable), Bayesian Ridge (couldn't handle noise)

### Exploration vs. Exploitation Balance

- **Weeks 1-4**: 80% exploration
- **Weeks 5-7**: 50% exploration
- **Weeks 8-10**: 30% exploration
- **Week 11**: 20% exploration, 80% exploitation
- **Week 12**: 40% exploration (aggressive hypothesis testing)

**Resource Allocation:**
- 75% effort on F5+F7 (still improving)
- 25% on others (plateaued or stochastic)

---

## Section 5: Results and Achievements

### Overall Performance

**Total Score Progression:**
- Week 1: 150.0 (baseline)
- Week 5: 5,575.3 (F5 breakthrough)
- Week 7: 6,164.7 (pre-x4 peak)
- Week 11: 6,532.8 (x4=1.0 breakthrough)
- **Total Improvement: +6,383 points (+4,255%)**

### Function-by-Function Best Results

| Function | Best Output | Week | Input Variables |
|----------|-------------|------|-----------------|
| **F1** | 0.5259 | W11 | [0.410, 0.433] |
| **F2** | 0.1300 | W5 | [0.111, 0.100] |
| **F3** | -0.0614 | W11 | [0.949, 0.966, 0.830] |
| **F4** | -3.9857 | W8 | [0.498, 0.502, 0.500, 0.500] |
| **F5** | 6526.44 | W11 | [1.0, 0.853, 1.0, 1.0] |
| **F6** | -1.0641 | W8 | [0.245, 0.162, 0.507, 0.482, 0.418] |
| **F7** | 1.4783 | W7 | [0.038, 0.462, 0.239, 0.171, 0.378, 0.734] |
| **F8** | 9.6921 | W7 | [0.177, 0.194, 0.170, 0.194, 0.294, 0.143, 0.109, 0.208] |

---

## Section 6: Key Learnings

### Technical Insights

1. **Regime Shifts Are Unpredictable**: F5's 40× jump was invisible to models trained on low-regime data
2. **ARD Length Scales Indicate Sensitivity**: Short length scale = critical boundary, not tunable
3. **Empirical Gap-Filling Beats Model Optimization at Boundaries**: Week 11's x4=1.0 was obvious from data
4. **Single-Variable Isolation Is Critical**: Changing one variable at a time enables causal attribution
5. **Noise Characterization Enables Strategy Adaptation**: Different approaches for deterministic vs. stochastic functions

### Strategic Insights

1. **Adaptive Resource Allocation**: Focus 75% on functions still improving
2. **Conservative Steps in Mature Phase**: Step sizes 0.010-0.023 in late weeks
3. **Trust Empirical Patterns Over Models at Boundaries**: F7 data showed peak at 0.734
4. **Floor Protection Enables Aggressive Exploration**: Week 11 locked floor allows Week 12 risk-taking
5. **Know When to Abandon Modeling**: F2 stochasticity makes all models useless

### Career-Applicable Skills

1. **Decision-Making Under Uncertainty**
2. **Model Selection and Validation**
3. **Strategic Planning and Resource Allocation**
4. **Communication and Justification**
5. **Resilience and Adaptation**

---

## Conclusion

This 13-week journey from 150 points to 6,533 points demonstrates the power of combining rigorous ML modeling with empirical pattern recognition and strategic resource allocation. The largest lessons came from failures: Week 8's disaster taught ARD interpretation, Week 10's duplicate taught noise characterization. With Week 12 results pending and Week 13 approaching, this project has delivered both technical mastery (GP/ARD, Bayesian Optimization) and strategic wisdom (when to trust models vs. data).

---

**Last Updated**: Week 11 (February 2026)  
**Current Best**: 6,532.80 total points  
**Next Submission**: Week 12 (Regime-shift hypothesis: x2=0.920 + x4=1.0)
