# Statistical Foundations of `utils.py`

## 1. Why Monte Carlo Works — Law of Large Numbers (LLN)

Each simulated path produces a random payoff $X_i$ (i.i.d.). The sample mean converges to the true option price:

$$\hat{P}_N = \frac{1}{N} \sum_{i=1}^{N} X_i \xrightarrow{N \to \infty} \mu = P^*$$

This justifies why MC pricing works at all — but it does not tell us *how far off* we are for a finite N.

---

## 2. Quantifying the Error — Central Limit Theorem (CLT)

The CLT quantifies the fluctuations around the true mean. For large N:

$$\frac{\hat{P}_N - \mu}{\sigma / \sqrt{N}} \xrightarrow{d} \mathcal{N}(0, 1)$$

### Conditions Required
1. **Independence** — each path uses independent random draws ✅
2. **Identically distributed** — same model, same parameters ✅
3. **Finite variance** — $\text{Var}(X_i) < \infty$ ✅ *(for vanilla options)*

The individual payoffs $X_i$ do **not** need to be Gaussian. The CLT applies regardless of payoff shape, as long as the three conditions hold.

---

## 3. What Does a Single Payoff Look Like?

For a European call, each payoff is:

$$X_i = e^{-rT} \max(S_T^{(i)} - K,\ 0)$$

This is **not Gaussian** — it is a mixture:

$$X_i = \begin{cases} 0 & \text{with probability } P(S_T < K) \\ \text{lognormal tail} & \text{with probability } P(S_T \geq K) \end{cases}$$

It has a large point mass at zero and a heavy right tail. Yet the CLT still applies because the variance remains finite.

---

## 4. Confidence Interval — From CLT to Probability Statement

From the standard normal table:

$$P\left(-1.96 \leq \frac{\hat{P}_N - \mu}{\sigma/\sqrt{N}} \leq 1.96\right) = 95\%$$

Rearranging for $\mu$:

$$P\left(\hat{P}_N - 1.96 \cdot \frac{\sigma}{\sqrt{N}} \leq \mu \leq \hat{P}_N + 1.96 \cdot \frac{\sigma}{\sqrt{N}}\right) = 95\%$$

Since the true $\sigma$ is unknown, we substitute the sample standard deviation $s$:

$$SE = \frac{s}{\sqrt{N}}, \qquad \text{CI} = \hat{P}_N \pm z_{1-\alpha/2} \cdot SE$$

### Key Insight — Why Error ∝ 1/√N

The CI width shrinks as $1/\sqrt{N}$. To halve the error, you need **4× more paths**. This is the fundamental cost of MC simulation.

---

## 5. Speed of Convergence to Normality — Berry-Esseen Theorem

The CLT is asymptotic. The Berry-Esseen theorem bounds how fast convergence occurs:

$$\sup_x \left| P\left(\frac{\hat{P}_N - \mu}{SE}\leq x\right) - \Phi(x) \right| \leq \frac{C \cdot \mathbb{E}|X_i - \mu|^3}{\sigma^3 \sqrt{N}}$$

Convergence to normality is **slower** when:
- The payoff distribution is highly skewed (e.g. deep OTM options)
- The third moment $\mathbb{E}|X_i - \mu|^3$ is large

---

## 6. Practical Validity of the Gaussian CI

| Scenario | CLT Validity | Risk |
|---|---|---|
| ATM vanilla call, large N | ✅ Strong | Low |
| Deep OTM call, small N | ⚠️ Weak | CI may be too narrow |
| Barrier / digital options | ⚠️ Weak | Discontinuous payoff → large skew |
| Variance swaps | ❌ Dangerous | Variance may not be finite |

---

## 7. Efficiency Ratio — Work-Normalised Variance

For two estimators, the efficiency is:

$$\text{Efficiency} = \frac{\text{Var}_{\text{base}} \times t_{\text{base}}}{\text{Var}_{\text{imp}} \times t_{\text{imp}}}$$

For **antithetic variates**, the variance becomes:

$$\text{Var}_{\text{antithetic}} = \frac{\sigma^2(1 + \rho)}{2}$$

When $\rho \to -1$, the variance $\to 0$ and efficiency $\to \infty$.

A ratio **> 1** means the improved estimator delivers more precision per unit of compute time.

---

## 8. Convergence Rate — Log-Log OLS Regression

Standard MC theory states:

$$\text{error} \propto N^{\beta}, \quad \beta = -0.5$$

Taking logs: $\log(\text{error}) = \beta \cdot \log(N) + c$

The slope is estimated by OLS:

$$\hat{\beta} = \frac{\text{Cov}(\log N,\ \log e)}{\text{Var}(\log N)} = \frac{\sum(\log N_i - \overline{\log N})(\log e_i - \overline{\log e})}{\sum(\log N_i - \overline{\log N})^2}$$

---

## 9. Black-Scholes Benchmark

The analytical benchmark $P^*_{BS}$ used for error computation is:

$$P^*_{BS} = S_0 \Phi(d_1) - K e^{-rT} \Phi(d_2)$$

$$d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$$

Errors are defined as:

$$\text{abs\_error} = |\hat{P}_N - P^*_{BS}|, \qquad \text{rel\_error} = \frac{|\hat{P}_N - P^*_{BS}|}{P^*_{BS}} \times 100\%$$