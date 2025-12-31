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


