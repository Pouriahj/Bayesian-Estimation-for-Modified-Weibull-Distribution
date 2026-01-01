# Bayesian Estimation for Modified Weibull Distribution

This repository contains MATLAB and Python implementations of **Approximate Bayesian Computation (ABC)** incorporated with  
**Metropolis–Hastings MCMC** and **Sequential Monte Carlo (SMC)** to estimate the parameters of a modified  
3-parameter Weibull distribution.

The code accompanies the paper:

> **M. Ravandi & P. Hajizadeh**,  
> *Application of approximate Bayesian computation for estimation of modified Weibull distribution parameters  
> for natural fiber strength with high uncertainty*, Journal of Materials Science, 57, 2731–2743 (2022).  
> DOI: https://doi.org/10.1007/s10853-021-06850-w  [oai_citation:0‡Springer Nature Link](https://link.springer.com/article/10.1007/s10853-021-06850-w?utm_source=chatgpt.com)

---


## 1. Model

We model the tensile strength $$(\sigma)$$ of a fiber with volume \( V \) using a volume-dependent
modified Weibull distribution. The cumulative distribution function (CDF) is

$$
F(\sigma|\sigma_{0}, \beta, m) = 1 - exp(-(V/V_{0})^{\beta}*(\sigma/\sigma_{0})^m)
$$

where

- \( \sigma_0 > 0 \) – scale parameter  
- \( m > 0 \) – shape parameter (Weibull modulus)  
- \( \beta \) – geometry-sensitivity exponent  
- \( V \) – fiber volume, \( V_0 \) – reference volume (fixed)

The corresponding probability density function (PDF) is

\[
f(\sigma \mid \sigma_0, \beta, m)
= \frac{m}{\sigma_0}
  \left(\frac{V}{V_0}\right)^{\beta}
  \left(\frac{\sigma}{\sigma_0}\right)^{m-1}
  \exp\!\left[
  - \left(\frac{V}{V_0}\right)^{\beta}
    \left(\frac{\sigma}{\sigma_0}\right)^m
  \right].
\]

For a dataset \(\{(\sigma_i, V_i)\}_{i=1}^N\), the full likelihood is the product of these PDFs.
In practice, we avoid evaluating this likelihood explicitly and instead use **ABC** with simulated data.

---

## 2. Priors

The parameter vector is

\[
\theta = (\sigma_0, \beta, m).
\]

We use independent **exponential priors** for each component:

\[
\sigma_0 \sim \text{Exp}(\lambda_1), \quad
\beta    \sim \text{Exp}(\lambda_2), \quad
m       \sim \text{Exp}(\lambda_3),
\]

so that the prior density factorizes as

\[
\pi(\theta) = \pi(\sigma_0)\,\pi(\beta)\,\pi(m).
\]

The actual values of \( \lambda_1, \lambda_2, \lambda_3 \) are chosen based on reasonable scales for the
data (and can be edited directly in the MATLAB and Python code).

---

## 3. ABC distance and tolerance

Given a candidate parameter vector \( \theta \):

1. **Simulate synthetic strengths**  
   For each observed volume \( V_i \), simulate a strength
   \[
   \tilde{\sigma}_i \sim f(\cdot \mid \theta, V_i).
   \]

2. **Sort by length / volume**  
   The paper groups strengths by gauge length / volume and sorts them before comparison.
   The code mirrors that procedure.

3. **Distance measure**  
   We define the ABC distance as the Euclidean norm between observed and simulated strengths:
   \[
   d(\theta)
   = \left\| \tilde{\boldsymbol{\sigma}}(\theta)
     - \boldsymbol{\sigma}_{\text{obs}} \right\|_2.
   \]

4. **Tolerance**  
   - ABC–MCMC uses a **single tolerance** \( \varepsilon \).  
   - ABC–SMC uses a **sequence of decreasing tolerances**
     \( \varepsilon_1 > \varepsilon_2 > \dots > \varepsilon_T \).

A proposed parameter value is considered **compatible with the data** if  
\( d(\theta) \leq \varepsilon \) (or \( \varepsilon_t \) at stage \( t \) in SMC).

---

## 4. ABC–MCMC (Metropolis–Hastings)

We run a **component-wise Metropolis–Hastings chain** for
\(\theta = (\sigma_0, \beta, m)\).

For each component \( \theta_j \in \{\sigma_0, \beta, m\} \):

1. **Proposal distribution (Gamma)**  
   From the current value \( \theta_j \), propose
   \[
   \theta_j^\* \sim \text{Gamma}(\alpha_j, \beta_j),
   \]
   where the gamma parameters are set so the mean is near \( \theta_j \) and ensures positivity.
   This is implemented using a **precision parameter** in the MATLAB code (e.g. `tau1`, `tau2`, `tau3`).

2. **Prior ratio**  
   Compute the ratio of prior densities:
   \[
   r_{\text{prior}}
   = \frac{\pi(\theta_j^\*)}{\pi(\theta_j)}.
   \]

3. **Likelihood ratio (via modified Weibull PDF)**  
   Using the user-defined `mod_wblpdf` (MATLAB) or its Python equivalent, compute
   \[
   r_{\text{like}}
   = \prod_{i=1}^N \frac{f(\sigma_i \mid \theta^\*, V_i)}{f(\sigma_i \mid \theta, V_i)}.
   \]

4. **Proposal asymmetry correction**  
   Because the gamma proposal is **asymmetric**, we correct using
   \[
   r_{\text{prop}}
   = \frac{q(\theta_j \mid \theta_j^\*)}{q(\theta_j^\* \mid \theta_j)},
   \]
   where \( q(\cdot \mid \cdot) \) is the gamma PDF used for proposals.

5. **Metropolis–Hastings acceptance probability**  
   \[
   \alpha_{\text{MH}}
   = \min\left( 1,\; r_{\text{prior}} \cdot r_{\text{like}} \cdot r_{\text{prop}} \right).
   \]

6. **ABC filtering**  
   - Generate synthetic data with \( \theta^\* \) and compute \( d(\theta^\*) \).  
   - If \( d(\theta^\*) > \varepsilon \), treat the proposal as automatically **rejected** (ABC filter).  
   - Otherwise, accept with probability \( \alpha_{\text{MH}} \).

7. **Accept / reject**  
   Draw \( u \sim \text{Uniform}(0,1) \).  
   - If \( u \leq \alpha_{\text{MH}} \) and \( d(\theta^\*) \leq \varepsilon \), set \( \theta_j \leftarrow \theta_j^\* \).  
   - Else, keep \( \theta_j \).

This is repeated for \( \sigma_0 \), \( \beta \), and \( m \) at each MCMC iteration, producing a Markov chain
whose stationary distribution approximates the **ABC posterior**.

---

## 5. ABC–SMC (Sequential Monte Carlo)

The ABC–SMC implementation propagates a **population of particles**
\(\{\theta_i^{(t)}\}_{i=1}^N\) with weights \(\{w_i^{(t)}\}_{i=1}^N\) through a sequence of tolerances.

### Stage 1 (t = 1)

1. Sample particles from the prior:
   \[
   \theta_i^{(1)} \sim \pi(\theta).
   \]
2. For each \( \theta_i^{(1)} \), simulate synthetic strengths and compute \( d(\theta_i^{(1)}) \).
3. Keep only particles with \( d(\theta_i^{(1)}) \leq \varepsilon_1 \).
4. Assign equal weights \( w_i^{(1)} = 1/N \).

### Stage t ≥ 2

Given particles \( \{\theta_k^{(t-1)}, w_k^{(t-1)}\}_{k=1}^N \):

1. **Resampling and perturbation**  
   - Draw an index \( k \) with probability \( w_k^{(t-1)} \).  
   - Set a base parameter \( \theta_k^{(t-1)} \).  
   - Perturb each component with a **uniform kernel**:
     \[
     \theta_j^\*
     \sim \text{Uniform}\big(\theta_j^{\text{min}} - r_j^{(t)},\
                              \theta_j^{\text{max}} + r_j^{(t)}\big),
     \]
     where \( r_j^{(t)} \) is typically half the spread of accepted values for component \( j \) at stage \( t-1 \).

2. **ABC acceptance**  
   - Simulate strengths with \( \theta^\* \) and compute \( d(\theta^\*) \).  
   - Accept \( \theta^\* \) as a new particle \( \theta_i^{(t)} \) if \( d(\theta^\*) \leq \varepsilon_t \).

3. **Weight update**  
   The new weight is
   \[
   w_i^{(t)} \propto
   \frac{\pi(\theta_i^{(t)})}
        {\sum_{k=1}^N w_k^{(t-1)} K(\theta_i^{(t)} \mid \theta_k^{(t-1)})},
   \]
   where \( K(\cdot \mid \cdot) \) is the uniform perturbation kernel.

4. **Normalize weights**  
   Scale so that \( \sum_i w_i^{(t)} = 1 \).

As \( t \) increases and \( \varepsilon_t \) decreases, the particle cloud contracts around regions of parameter
space that generate synthetic data close to the observations, approximating the ABC posterior more tightly.

---

## 6. Flowchart

Insert a high-quality flowchart image for the ABC–MCMC and/or ABC–SMC algorithms here.

For example, after adding your PNG file to `assets/`:

```markdown
![Flowchart of the ABC–MCMC algorithm](assets/abc_mcmc_flowchart.png)

## Overview

We consider a **volume-based modified 3-parameter Weibull model** for fiber tensile strength.  
The cumulative distribution function is

\[
F(\sigma \mid \sigma_0, \beta, m)
= 1 - \exp\!\left[-\left(\frac{V}{V_0}\right)^{\beta}
\left(\frac{\sigma}{\sigma_0}\right)^m \right],
\]

where:

- \( \sigma \) – fiber strength  
- \( \sigma_0 \) – scale parameter  
- \( m \) – shape (Weibull modulus)  
- \( \beta \) – geometry sensitivity exponent  
- \( V \) – fiber volume, \( V_0 \) – reference volume  [oai_citation:1‡ResearchGate](https://www.researchgate.net/publication/357822194_Application_of_approximate_Bayesian_computation_for_estimation_of_modified_weibull_distribution_parameters_for_natural_fiber_strength_with_high_uncertainty?utm_source=chatgpt.com)  

The corresponding probability density function is

\[
f(\sigma \mid \sigma_0, \beta, m)
= \frac{m}{\sigma_0}
\left(\frac{V}{V_0}\right)^{\beta}
\left(\frac{\sigma}{\sigma_0}\right)^{m-1}
\exp\!\left[-\left(\frac{V}{V_0}\right)^{\beta}
\left(\frac{\sigma}{\sigma_0}\right)^m \right].
\]

Because the likelihood is expensive / intractable for the full hierarchical model, the paper uses
**likelihood-free Bayesian inference** via Approximate Bayesian Computation.

---

## Algorithms

The repository implements two ABC algorithms:

1. **ABC–MCMC (Metropolis–Hastings)**  
   - Propose new parameters component-wise from Gaussian random walks.  
   - Simulate synthetic fiber strengths from the modified Weibull model.  
   - Compute a distance between simulated and observed strength distributions.  
   - Accept or reject the proposal based on an ABC tolerance \( \epsilon \).  

2. **ABC–SMC (Sequential Monte Carlo)**  
   - Evolve a population of particles through a sequence of decreasing tolerances
     \( \epsilon_1 > \epsilon_2 > \dots > \epsilon_T \).  
   - Use importance weights and resampling to focus particles near high-posterior-density regions.  
   - Provides a richer view of posterior scatter and predictive distributions.  [oai_citation:2‡ResearchGate](https://www.researchgate.net/publication/357822194_Application_of_approximate_Bayesian_computation_for_estimation_of_modified_weibull_distribution_parameters_for_natural_fiber_strength_with_high_uncertainty?utm_source=chatgpt.com)  

Both algorithms return posterior samples of \( \sigma_0, \beta, m \) and allow you to compute
summary statistics, credible intervals, and posterior predictive curves.

---

## Flowchart

You can include your high-quality flowchart from the paper here. For example, after placing the image at:

`assets/abc_mcmc_flowchart.png`

add:

```markdown
![Flowchart of the ABC–MCMC algorithm](assets/abc_mcmc_flowchart.png)


