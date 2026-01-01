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
d(\theta)=
\text{norm}\left(
\tilde{\sigma}(\theta), \sigma_{\text{obs}}
\right)
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
   Propose $$\theta_{j}^{\*}$$ from a Gamma distribution $$(q)$$ centered near the current $$\theta_{j}$$.

2. **Compute acceptance ratio** using prior, likelihood and proposal terms:

$$
\Large
\alpha_{\mathrm{MH}}=\min\Biggl(
  1,
  \frac{\pi(\theta^{\*})}{\pi(\theta)}
  \prod_{i=1}^{N}
  \frac{
    f\left(\sigma_{i} \mid \theta^{\*}, V_{i}\right)
  }{
    f\left(\sigma_{i} \mid \theta, V_{i}\right)
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

ABC–SMC maintains a population of particles $\{\theta_i^{(t)}, w_i^{(t)}\}_{i=1}^N$
at each stage $t$ with tolerance $\varepsilon_t$.

- **Stage 1 (t = 1)**  
  Sample $\theta_i^{(1)} \sim \pi(\theta)$ from the prior, simulate data and compute
  $d(\theta_i^{(1)})$. Keep only particles with $d(\theta_i^{(1)}) \le \varepsilon_1$ and
  set equal weights $w_i^{(1)} = 1/N$.

- **Stages t ≥ 2**  
  At each stage $t$:

  i. Resample a previous particle $\theta_k^{(t-1)}$ with probability $w_k^{(t-1)}$.  

  ii. Perturb it with a kernel $K(\theta \mid \theta_k^{(t-1)})$ to obtain a proposal
      $\theta_i^{(t)}$.  

  iii. Simulate data with $\theta_i^{(t)}$ and compute $d(\theta_i^{(t)})$.
       Accept the proposal only if $d(\theta_i^{(t)}) \le \varepsilon_t$.  

  iv. Update the weight:

  $$
  \Large
  w_i^{(t)} \propto
  \frac{
    \pi\!\left(\theta_i^{(t)}\right)
  }{
    \displaystyle
    \sum_{k=1}^{N}
    w_k^{(t-1)}
    K\!\left(
      \theta_i^{(t)}
      \mid
      \theta_k^{(t-1)}
    \right)
  }
  $$

  v. Normalize so that $\sum_{i=1}^{N} w_i^{(t)} = 1$.

As $t$ increases and $\varepsilon_t$ decreases, the particles concentrate in regions that
generate synthetic data close to the observations, approximating the ABC posterior.



