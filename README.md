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

We model the tensile strength $$(\sigma)$$ of a fiber with volume $$(V)$$ using a volume-dependent
modified Weibull distribution. The cumulative distribution function (CDF) is:

$$
\Large
F(\sigma \mid \sigma_{0}, \beta, m)
= 1 - \exp\left(
  -\left(\frac{V}{V_{0}}\right)^{\beta}
   \left(\frac{\sigma}{\sigma_{0}}\right)^{m}
\right)
$$

where

- $$\sigma_{0} > 0$$ – scale parameter  
- $$m > 0$$ – shape parameter (Weibull modulus)  
- $$\beta$$ – geometry-sensitivity exponent  
- $$V$$ – fiber volume
- $$V_{0}$$ – reference volume (fixed, here is equal to 1)

The corresponding probability density function (PDF) is

$$
\Large
f(\sigma \mid \sigma_{0}, \beta, m)
= \frac{m}{\sigma_{0}}
  \left(\frac{V}{V_{0}}\right)^{\beta}
  \left(\frac{\sigma}{\sigma_{0}}\right)^{m-1}
  \exp\left(
    -\left(\frac{V}{V_{0}}\right)^{\beta}
     \left(\frac{\sigma}{\sigma_{0}}\right)^{m}
  \right)
$$

For a dataset $$\(\{(\sigma_i, V_i)\}_{i=1}^N\)$$, the full likelihood is the product of these PDFs.
In practice, we avoid evaluating this likelihood explicitly and instead use **ABC** with simulated data.

---

## 2. Priors

The parameter vector is

$$ 
\Large
\theta = (\sigma_0, \beta, m).
$$

We let $$\pi(\theta)$$ to be the priors. We use independent **exponential priors** and sample from them component-wise:

$$
\Large
\sigma_0 \sim \text{Exp}(\lambda_1), \quad
\beta    \sim \text{Exp}(\lambda_2), \quad
m       \sim \text{Exp}(\lambda_3),
$$

The actual values of $$(\lambda_1, \lambda_2, \lambda_3)$$ are chosen based on reasonable scales for the
data (and can be edited directly in the MATLAB code).

---

## 3. ABC distance and tolerance

We compare the observed strengths $$\sigma_{\text{obs}}$$ with synthetic strengths
$$\tilde{\sigma}(\theta)$$ simulated under parameters $$\theta = (\sigma_{0}, \beta, m)$$
using the modified Weibull model. 
ABC uses the Euclidean distance

$$
\Large
d(\theta) = 
\tilde{\sigma}(\theta)\sigma_{\text{obs}}
$$

and accepts parameters that satisfy

$$
\Large
d(\theta) \le \varepsilon.
$$

ABC–MCMC uses a single tolerance $$\varepsilon$$, while ABC–SMC uses a decreasing sequence  
$$\varepsilon_{1} > \varepsilon_{2} > \dots > \varepsilon_{T}.$$

---

## 4. ABC–MCMC (Metropolis–Hastings)

ABC–MCMC runs a component-wise Metropolis–Hastings chain on

$$
\Large
\theta = (\sigma_{0}, \beta, m).
$$

For each component $$\theta_{j} \in \{\sigma_{0}, \beta, m\}$$:

1. **Proposal**  
   Propose $$\theta_{j}^{\*}$$ from a Gamma distribution centered near the current $$\theta_{j}$$.

2. **Compute acceptance ratio** using prior, likelihood and proposal terms:

$$
\Large
\alpha_{\mathrm{MH}}
=
\min\Biggl(
  1,\;
  \frac{\pi(\theta^{\*})}{\pi(\theta)}
  \prod_{i=1}^{N}
  \frac{
    f\!\left(\sigma_{i} \mid \theta^{\*}, V_{i}\right)
  }{
    f\!\left(\sigma_{i} \mid \theta, V_{i}\right)
  }
  \frac{
    q(\theta_{j} \mid \theta_{j}^{\*})
  }{
    q(\theta_{j}^{\*} \mid \theta_{j})
  }
\Biggr)
$$

3. **ABC filter and accept/reject**  
   - Simulate data with $$\theta^{\*}$$, compute $$d(\theta^{\*})$$.  
   - If $$d(\theta^{\*}) \le \varepsilon$$, accept $$\theta_{j}^{\*}$$ with probability $$\alpha_{\mathrm{MH}}$$;  
     otherwise reject and keep $$\theta_{j}$$.

After burn-in, the chain provides ABC posterior samples of $$\theta$$.

**Flowchart (ABC–MCMC)**  


---

## 5. ABC–SMC (Sequential Monte Carlo)

ABC–SMC maintains a population of particles $$\{\theta_{i}^{(t)}, w_{i}^{(t)}\}_{i=1}^{N}$$  
at each stage $$t$$ with tolerance $$\varepsilon_{t}$$.

- **Stage 1 (t = 1)**  
  Sample $$\theta_{i}^{(1)} \sim \pi(\theta)$$ from the prior, simulate data and compute  
  $$d\big(\theta_{i}^{(1)}\big)$$. Keep only particles with
  $$d\big(\theta_{i}^{(1)}\big) \le \varepsilon_{1}$$ and set equal weights
  $$w_{i}^{(1)} = 1/N.$$

- **Stages t ≥ 2**  
  At each stage $$t$$:
  1. Resample a previous particle $$\theta_{k}^{(t-1)}$$ with probability $$w_{k}^{(t-1)}$$.  
  2. Perturb it with a kernel $$K\big(\theta \mid \theta_{k}^{(t-1)}\big)$$ to obtain a proposal
     $$\theta_{i}^{(t)}$$.  
  3. Simulate data with $$\theta_{i}^{(t)}$$ and compute $$d\big(\theta_{i}^{(t)}\big)$$.  
     Accept the proposal only if
     $$d\big(\theta_{i}^{(t)}\big) \le \varepsilon_{t}.$$
  4. Update the weight:

     $$
     \Large
     w_{i}^{(t)} \propto
     \frac{
       \pi\!\left(\theta_{i}^{(t)}\right)
     }{
       \displaystyle
       \sum_{k=1}^{N}
       w_{k}^{(t-1)}
       K\!\left(
         \theta_{i}^{(t)}
         \mid
         \theta_{k}^{(t-1)}
       \right)
     }
     $$

  5. Normalize so that
     $$\sum_{i=1}^{N} w_{i}^{(t)} = 1.$$

As $$t$$ increases and $$\varepsilon_{t}$$ decreases, the particles concentrate in regions that
generate synthetic data close to the observations, approximating the ABC posterior.



