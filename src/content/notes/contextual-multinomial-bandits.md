## Problem Formulation

At round $t=1,2,\ldots$, the interaction protocol is as follows.

- The agent observes feature vectors $x_{ti}\in\mathbb{R}^d$ for every item $i\in[N]$.
- Based on this contextual information, the agent chooses an action subset $S_t=\{i_1,\ldots,i_l\}\in\mathcal{S}$, where $\mathcal{S}=\{S\subset[N]:|S|\leq K\}$.
- The agent observes the final decision $c_t\in S_t\cup\{0\}$, where $0$ represents the outside option, and receives the corresponding reward $r_{tc_t}$.

The distribution of these selections follows a multinomial logit (MNL) choice model. For an unknown parameter $\mathbf{w}^\star\in\mathbb{R}^d$, the probability of choosing an item $i_t\in S_t$, or the outside option, is

$$
p_t(i_t\mid S_t,\mathbf{w}^\star)=
\frac{\exp(x_{ti_t}^\top \mathbf{w}^\star)}
{1+\sum_{j\in S_t}\exp(x_{tj}^\top\mathbf{w}^\star)},
\quad
p_t(0\mid S_t,\mathbf{w}^\star)=
\frac{1}
{1+\sum_{j\in S_t}\exp(x_{tj}^\top\mathbf{w}^\star)}.
$$

The choice response for each item $i\in S_t\cup\{0\}$ is defined as $y_{ti}=\mathbb{1}\{c_t=i\}\in\{0,1\}$. Therefore, the choice feedback vector $\mathbf{y}_t$ can be viewed as a single-trial multinomial sample:

$$
\mathbf{y}_t\sim\text{Multinomial}(1,\mathbf{p}_t),
$$

where $\mathbf{p}_t=[p_t(0\mid S_t,\mathbf{w}^\star),\ldots,p_t(i_l\mid S_t,\mathbf{w}^\star)]$.

For each $i\in S_t\cup\{0\}$, define the noise

$$
\epsilon_{ti}=y_{ti}-p_t(i\mid S_t,\mathbf{w}^\star).
$$

Since $y_{ti}$ is bounded, $\epsilon_{ti}$ is centered and bounded; equivalently, it is sub-Gaussian with a constant variance proxy. In particular, under the usual Bernoulli normalization, the variance proxy is at most $1/4$.

At round $t$, the reward $r_{ti}$ for each item $i$ is also given, with $r_{t0}\equiv0$. Therefore, the expected reward of choosing set $S$ at round $t$ is

$$
R_t(S,\mathbf{w}^\star)=
\sum_{i\in S}p_t(i\mid S,\mathbf{w}^\star)r_{ti}
=
\frac{\sum_{i\in S}\exp(x_{ti}^\top\mathbf{w}^\star)r_{ti}}
{1+\sum_{j\in S}\exp(x_{tj}^\top \mathbf{w}^\star)}.
$$

Define

$$
S_t^\star=\arg\max_{S\in\mathcal{S}}R_t(S,\mathbf{w}^\star).
$$

Our goal is to minimize the cumulative pseudo-regret over $T$ rounds:

$$
\text{Reg}_T
=
\sum_{t=1}^T
R_t(S_t^\star,\mathbf{w}^\star)-R_t(S_t,\mathbf{w}^\star).
$$

Consistent with previous work on contextual MNL bandits, we make the following assumptions.

- Boundedness: assume $\|\mathbf{w}^\star\|_2\leq 1$, and for all $t\geq 1$, $i\in[N]$, $\|x_{ti}\|_2\leq 1$ and $0<\rho\leq r_{ti}\leq 1$.
- Problem-dependent constant: there exists $0<\kappa\leq 1$ such that for every item $i\in S$, every $S\in\mathcal{S}$, and every round $t$,

$$
\inf_{\mathbf{w}:\|\mathbf{w}\|_2\leq 1}
p_t(i\mid S,\mathbf{w})p_t(0\mid S,\mathbf{w})
\geq\kappa.
$$

This problem-dependent constant represents the non-linearity of the multinomial function. Note that $1/\kappa$ depends on the maximum choice size $K$; typically, $1/\kappa=\mathcal{O}(K^2)$.

## Previous Work

See [Lee and Oh, 2024](https://arxiv.org/abs/2502.10020) for related online confidence bounds and optimistic algorithms for multinomial logistic bandits.

Define the multinomial logistic loss function at round $t$ as

$$
\ell_t(\mathbf{w})
=
-\sum_{i\in S_t}y_{ti}\log p_t(i\mid S_t,\mathbf{w}).
$$

The corresponding Hessian matrix at round $t$ is

$$
H_t(\mathbf{w})
=
\sum_{i\in S_t}p_t(i\mid S_t,\mathbf{w})x_{ti}x_{ti}^\top
-
\sum_{i\in S_t}\sum_{j\in S_t}
p_t(i\mid S_t,\mathbf{w})p_t(j\mid S_t,\mathbf{w})
x_{ti}x_{tj}^\top.
$$

The OFU-MNL+ style algorithm uses an Online Mirror Descent (OMD) update following this procedure.

- Initialize $\mathcal{H}_1=\lambda \mathbf{I}_d$ and choose $\mathbf{w}_1$ such that $\|\mathbf{w}_1\|_2\leq 1$.
- At round $t=1,2,\ldots,T$, compute

$$
\alpha_{ti}
=
x_{ti}^\top\mathbf{w}_t
+
\beta_t(\delta)\|x_{ti}\|_{\mathcal{H}_t^{-1}},
\quad \forall i\in[N].
$$

- Choose $S_t=\arg\max_{S\in\mathcal{S}}\tilde{R}_t(S)$, where the optimistic expected reward function is

$$
\tilde{R}_t(S)
=
\frac{\sum_{i\in S}\exp(\alpha_{ti})r_{ti}}
{1+\sum_{j\in S}\exp(\alpha_{tj})}.
$$

- Get the choice response $\mathbf{y}_t$.
- Update $\tilde{\mathcal{H}}_t=\mathcal{H}_t+\eta H_t(\mathbf{w}_t)$, and update the estimator $\mathbf{w}_{t+1}$ by the OMD update

$$
\mathbf{w}_{t+1}
=
\underset{\mathbf{w}:\|\mathbf{w}\|_2\leq 1}{\arg\min}
\left\langle \nabla\ell_t(\mathbf{w}_t),\mathbf{w}\right\rangle
+
\frac{1}{2\eta}\|\mathbf{w}-\mathbf{w}_t\|_{\tilde{\mathcal{H}}_t}^2.
$$

- Update the look-ahead Hessian $\mathcal{H}_{t+1}=\mathcal{H}_t+H_t(\mathbf{w}_{t+1})$.

In this algorithm, $\beta_t(\delta)=\mathcal{O}(\sqrt{d}\log t\log K)$ is the radius of the confidence set. With probability at least $1-\delta$,

$$
\mathbf{w}^\star\in\mathcal{C}_t(\delta)
:=
\left\{
\mathbf{w}\mid
\|\mathbf{w}\|_2\leq 1,\,
\|\mathbf{w}_t-\mathbf{w}\|_{\mathcal{H}_t}\leq\beta_t(\delta)
\right\}.
$$

## Algorithm

Following the recipe of [Boudart et al., 2025](https://arxiv.org/abs/2507.05306), we consider the following algorithm.

- Initialize $\mathcal{H}_1=\lambda\mathbf{I}_d$ and $\mathbf{w}_1$ as any vector in $\mathcal{W}$, where $\mathcal{W}$ is returned by an exploration routine.
- At round $t=1,2,\ldots,T$, compute

$$
\alpha_{ti}
=
x_{ti}^\top\mathbf{w}_t
+
\beta_t(\delta)\|x_{ti}\|_{\mathcal{H}_t^{-1}},
\quad \forall i\in[N].
$$

- Choose $S_t=\arg\max_{S\in\mathcal{S}}\tilde{R}_t(S)$, where

$$
\tilde{R}_t(S)
=
\frac{\sum_{i\in S}\exp(\alpha_{ti})r_{ti}}
{1+\sum_{j\in S}\exp(\alpha_{tj})}.
$$

- Get the choice response $\mathbf{y}_t$.
- Update $\tilde{\mathcal{H}}_t=\mathcal{H}_t+\eta H_t(\mathbf{w}_t)$, and update the estimator $\mathbf{w}_{t+1}$ by

$$
\mathbf{w}_{t+1}
=
\underset{\mathbf{w}\in\mathcal{W}}{\arg\min}
\left\langle\nabla\ell_t(\mathbf{w}_t),\mathbf{w}\right\rangle
+
\frac{1}{2\eta}\|\mathbf{w}-\mathbf{w}_t\|_{\tilde{\mathcal{H}}_t}^2.
$$

- Update the look-ahead Hessian $\mathcal{H}_{t+1}=\mathcal{H}_t+H_t(\mathbf{w}_{t+1})$.

Here, the exploration routine, or warm-up phase, outputs a parameter space $\mathcal{W}$ such that $\text{diam}(\mathcal{W})\leq 1$, where

$$
\text{diam}(\mathcal{W})
=
\max_{t\geq1,i\in[N]}
\max_{\mathbf{w}_1,\mathbf{w}_2\in\mathcal{W}}
\left|x_{ti}^\top(\mathbf{w}_1-\mathbf{w}_2)\right|.
$$

This enables us to leverage the self-concordance property of the MNL function without incurring an exponential constant. More specifically, the exploration routine is as follows.

- Initialize $V_0=\lambda \mathbf{I}_d$.
- At round $t=1,2,\ldots,\tau$, iteratively add items $x_{ti}$ to $S_t$ with the largest leverage score $x_{ti}^\top V_{t-1}^{-1}x_{ti}$ until $|S_t|=K$.
- Get the choice response $\mathbf{y}_t$.
- Update

$$
V_t=V_{t-1}+\kappa \sum_{i\in S_t} x_{ti}x_{ti}^\top.
$$

- Compute

$$
\hat{\mathbf{w}}
=
\arg\min_{\mathbf{w}:\|\mathbf{w}\|_2\leq 1}
\sum_{t=1}^\tau\ell_t(\mathbf{w})
+
\frac{\lambda}{2}\|\mathbf{w}\|_2^2,
$$

and output

$$
\mathcal{W}
=
\left\{
\mathbf{w}\in\mathbb{R}^d:
\|\mathbf{w}\|_2\leq 1,\,
\|\mathbf{w}-\hat{\mathbf{w}}\|_{V_\tau}^2
\leq \mathcal{O}(\lambda)
\right\}.
$$

## Analysis

### Uniform Reward Setting

We first consider the uniform-reward case, i.e., $r_{ti}\equiv 1$. In some scenarios such as dynamic pricing or assortment, as studied in [Perivier and Goyal, 2022](https://arxiv.org/abs/2110.10018), we only consider the assortment with no relative reward.

Using optimism and the greedy assortment choice, we have

$$
\text{Reg}_T
=
\sum_{t=1}^T
R_t(S_t^\star,\mathbf{w}^\star)-R_t(S_t,\mathbf{w}^\star)
\leq
\sum_{t=1}^T
\tilde{R}_t(S_t)-R_t(S_t,\mathbf{w}^\star),
$$

where

$$
\tilde{R}_t(S_t)
=
\frac{\sum_{i\in S_t}\exp(\alpha_{ti})}
{1+\sum_{j\in S_t}\exp(\alpha_{tj})},
\quad
R_t(S_t,\mathbf{w}^\star)
=
\frac{\sum_{i\in S_t}\exp(x_{ti}^\top\mathbf{w}^\star)}
{1+\sum_{j\in S_t}\exp(x_{tj}^\top\mathbf{w}^\star)}.
$$

Let $Q:\mathbb{R}^K\rightarrow\mathbb{R}$ be

$$
Q(\mathbf{u})
=
\frac{\sum_{i=1}^K\exp(u_i)}
{1+\sum_{j=1}^K\exp(u_j)}.
$$

In the uniform reward setting, $S_t$ always contains $K$ elements, i.e., $S_t=\{i_1,\ldots,i_K\}$. Let

$$
\mathbf{u}_t=(\alpha_{ti_1},\ldots,\alpha_{ti_K})^\top,
\quad
\mathbf{u}_t^\star
=
(x_{ti_1}^\top\mathbf{w}^\star,\ldots,x_{ti_K}^\top\mathbf{w}^\star)^\top.
$$

Then the regret can be expressed as

$$
\begin{aligned}
\text{Reg}_T
&\leq
\sum_{t=1}^T Q(\mathbf{u}_t)-Q(\mathbf{u}_t^\star)\\
&=
\sum_{t=1}^T
\nabla Q(\mathbf{u}_t^\star)^\top(\mathbf{u}_t-\mathbf{u}_t^\star)
+
\frac{1}{2}\sum_{t=1}^T
(\mathbf{u}_t-\mathbf{u}_t^\star)^\top
\nabla^2 Q(\bar{\mathbf{u}}_t)
(\mathbf{u}_t-\mathbf{u}_t^\star).
\end{aligned}
$$

The first-order error can be bounded by

$$
\sum_{t=1}^T
\nabla Q(\mathbf{u}_t^\star)^\top(\mathbf{u}_t-\mathbf{u}_t^\star)
\leq
2\beta_T(\delta)
\sum_{t=1}^T\sum_{i\in S_t}
p_t(i\mid S_t,\mathbf{w}^\star)p_t(0\mid S_t,\mathbf{w}^\star)
\|x_{ti}\|_{\mathcal{H}_t^{-1}},
$$

and the second-order error can be bounded by

$$
\frac{1}{2}\sum_{t=1}^T
(\mathbf{u}_t-\mathbf{u}_t^\star)^\top
\nabla^2 Q(\bar{\mathbf{u}}_t)
(\mathbf{u}_t-\mathbf{u}_t^\star)
\leq
10\beta_T(\delta)^2
\sum_{t=1}^T
\max_{i\in S_t}
\|x_{ti}\|_{\mathcal{H}_t^{-1}}^2.
$$

Here, $\beta_t(\delta)$ is the radius of the $t$-th round confidence set. With probability at least $1-\delta$,

$$
\mathbf{w}^\star\in \mathcal{C}_t(\delta)
:=
\left\{
\mathbf{w}\in\mathcal{W}:
\|\mathbf{w}-\mathbf{w}_t\|_{\mathcal{H}_t}
\leq\beta_t(\delta)
\right\},
$$

with $\beta_t(\delta)=\mathcal{O}(\sqrt{d}\log t\log K)$.

Recall the elliptical potential lemma in contextual MNL bandits from [Lee and Oh, 2024](https://arxiv.org/abs/2502.10020). Since the cumulative Hessian is

$$
\mathcal{H}_t
=
\lambda \mathbf{I}_d+\sum_{s=1}^{t-1}H_s(\mathbf{w}_{s+1}),
$$

we have

- $\sum_{s=1}^t\sum_{i\in S_s}p_s(i\mid S_s,\mathbf{w}_{s+1})p_s(0\mid S_s,\mathbf{w}_{s+1})\|x_{si}\|_{\mathcal{H}_s^{-1}}^2\leq 2d\log(1+\frac{t}{d\lambda})$.
- $\sum_{s=1}^t\max_{i\in S_s}\|x_{si}\|_{\mathcal{H}_{s}^{-1}}^2\leq \frac{2}{\kappa}d\log(1+\frac{t}{d\lambda})$.

Using the second elliptical-potential inequality directly bounds the second-order error. The first-order error is more subtle. In [Lee and Oh, 2024](https://arxiv.org/abs/2502.10020), this first-order error is decomposed into three terms and each term is bounded using the elliptical potential lemma.

Here, we use a different analysis: directly apply the Cauchy-Schwarz inequality, in a style similar to [Boudart et al., 2025](https://arxiv.org/abs/2507.05306):

$$
\begin{aligned}
&\sum_{t=1}^T\sum_{i\in S_t}
p_t(i\mid S_t,\mathbf{w}^\star)p_t(0\mid S_t,\mathbf{w}^\star)
\|x_{ti}\|_{\mathcal{H}_t^{-1}}\\
&\leq
\sqrt{
\sum_{t=1}^T\sum_{i\in S_t}
p_t(i\mid S_t,\mathbf{w}^\star)p_t(0\mid S_t,\mathbf{w}^\star)
}\\
&\quad\cdot
\sqrt{
\sum_{t=1}^T\sum_{i\in S_t}
p_t(i\mid S_t,\mathbf{w}^\star)p_t(0\mid S_t,\mathbf{w}^\star)
\|x_{ti}\|_{\mathcal{H}_t^{-1}}^2
}.
\end{aligned}
$$

For the first term, following [Perivier and Goyal, 2022](https://arxiv.org/abs/2110.10018), we have

$$
\sum_{t=1}^T\sum_{i\in S_t}
p_t(i\mid S_t,\mathbf{w}^\star)p_t(0\mid S_t,\mathbf{w}^\star)
\leq
\text{Regret}_T+\sum_{t=1}^T\kappa_t^\star,
\tag{$\star$}
$$

where

$$
\kappa_t^\star
=
\sum_{i\in S_t^\star}
p_t(i\mid S_t^\star,\mathbf{w}^\star)p_t(0\mid S_t^\star,\mathbf{w}^\star)
$$

is a problem-dependent term.

For the second term, let

$$
\mathcal{H}_t^\star
=
\lambda' \mathbf{I}_d+\sum_{s=1}^{t-1}H_s(\mathbf{w}^\star).
$$

Since $\ell_t(\mathbf{w})$ is a $3\sqrt{2}$-self-concordance-like function, for all $\mathbf{w}_1,\mathbf{w}_2\in\mathcal{W}$,

$$
H_t(\mathbf{w}_1)
\preceq
\exp(3\sqrt{2}\text{diam}(\mathcal{W}))
H_t(\mathbf{w}_2)
\leq
\exp(3\sqrt{2})H_t(\mathbf{w}_2).
$$

This gives credit to the exploration routine, which makes $\text{diam}(\mathcal{W})\leq 1$.

Therefore, setting $\lambda'=e^{3\sqrt{2}}\lambda$, we have $\mathcal{H}_t^\star\preceq e^{3\sqrt{2}}\mathcal{H}_t$, which means that $\|x_{ti}\|_{\mathcal{H}_t^{-1}}^2$ can be controlled by $\|x_{ti}\|_{(\mathcal{H}_t^\star)^{-1}}^2$ up to this constant. Now we can use the elliptical potential lemma by setting all $\mathbf{w}_{s+1}=\mathbf{w}^\star$.

Plugging in $\beta_T(\delta)=\mathcal{O}(\sqrt{d}\log T)$, we get the quadratic inequality

$$
\text{Reg}_T
\lesssim
d\sqrt{\sum_{t=1}^T\kappa_t^\star+\text{Reg}_T}
+
\frac{d^2}{\kappa}.
$$

Solving it yields the regret bound under the uniform reward setting:

$$
\text{Reg}_T
=
\tilde{\mathcal{O}}
\left(
d\sqrt{\sum_{t=1}^T\kappa_t^\star}
+
\frac{1}{\kappa}d^2
\right).
$$

### Non-Uniform Reward Setting

Now we turn to the non-uniform reward setting. Similarly, let $\tilde{Q}:\mathbb{R}^{|S_t|}\rightarrow\mathbb{R}$ be

$$
\tilde{Q}(\mathbf{u})
=
\sum_{i\in S_t}
\frac{\exp(u_i) r_{ti}}
{1+\sum_{j\in S_t}\exp(u_j)}.
$$

We express the regret as

$$
\begin{aligned}
\text{Regret}_T
&\leq
\sum_{t=1}^T\tilde{Q}(\mathbf{u}_t)-\tilde{Q}(\mathbf{u}_t^\star)\\
&=
\sum_{t=1}^T
\nabla \tilde{Q}(\mathbf{u}_t^\star)^\top(\mathbf{u}_t-\mathbf{u}_t^\star)
+
\frac{1}{2}\sum_{t=1}^T
(\mathbf{u}_t-\mathbf{u}_t^\star)^\top
\nabla^2 \tilde{Q}(\bar{\mathbf{u}}_t)
(\mathbf{u}_t-\mathbf{u}_t^\star).
\end{aligned}
$$

A possible way to deal with non-uniform rewards is simply to drop the reward factor when upper bounding. For instance, for the second-order error, since $|r_{ti}|\leq 1$ for all $t\in[T]$ and $i\in[N]$, we have

$$
\left|
\frac{\partial^2\tilde{Q}(\mathbf{u})}{\partial u_i\partial u_j}
\right|
\leq
\left|
\frac{\partial^2Q(\mathbf{u})}{\partial u_i\partial u_j}
\right|.
$$

Therefore, the same second-order bound still holds:

$$
\frac{1}{2}\sum_{t=1}^T
(\mathbf{u}_t-\mathbf{u}_t^\star)^\top
\nabla^2 \tilde{Q}(\bar{\mathbf{u}}_t)
(\mathbf{u}_t-\mathbf{u}_t^\star)
\leq
10\beta_T(\delta)^2
\sum_{t=1}^T
\max_{i\in S_t}
\|x_{ti}\|_{\mathcal{H}_t^{-1}}^2.
$$

The first-order error can be written in two different forms in the non-uniform reward setting. One form is

$$
\sum_{t=1}^T
\nabla \tilde{Q}(\mathbf{u}_t^\star)^\top(\mathbf{u}_t-\mathbf{u}_t^\star)
\leq
\beta_T(\delta)
\sum_{t=1}^T\sum_{i\in S_t}
p_t(i\mid S_t,\mathbf{w}^\star)
\left(
r_{ti}
-
\sum_{j\in S_t}
p_t(j\mid S_t,\mathbf{w}^\star)r_{tj}
\right)
\|x_{ti}\|_{\mathcal{H}_t^{-1}}.
$$

In [Lee and Oh, 2024](https://arxiv.org/abs/2502.10020), another form is used to bound the first-order error:

$$
\sum_{t=1}^T
\nabla \tilde{Q}(\mathbf{u}_t^\star)^\top(\mathbf{u}_t-\mathbf{u}_t^\star)
\leq
\beta_T(\delta)
\sum_{t=1}^T\sum_{i\in S_t}
p_t(i\mid S_t,\mathbf{w}^\star)
\left\|
x_{ti}
-
\mathbb{E}_{j\sim p_t(\cdot\mid S_t,\mathbf{w}^\star)}[x_{tj}]
\right\|_{\mathcal{H}_t^{-1}}.
$$

They introduce a new form of the elliptical potential lemma. Let

$$
\tilde{x}_{si}
=
x_{si}
-
\mathbb{E}_{j\sim p_s(\cdot\mid S_s,\mathbf{w}_{s+1})}[x_{sj}].
$$

Then

- $\sum_{s=1}^t\sum_{i\in S_s}p_s(i\mid S_s,\mathbf{w}_{s+1})\|\tilde{x}_{si}\|_{\mathcal{H}_s^{-1}}^2\leq 2d\log(1+\frac{t}{d\lambda})$.
- $\sum_{s=1}^t\max_{i\in S_s}\|\tilde{x}_{si}\|_{\mathcal{H}_{s}^{-1}}^2\leq \frac{2}{\kappa}d\log(1+\frac{t}{d\lambda})$.

By decomposing through $\tilde{x}_{si}$ and

$$
\bar{x}_{si}
=
x_{si}
-
\mathbb{E}_{j\sim p_s(\cdot\mid S_s,\mathbf{w}^\star)}[x_{sj}],
$$

they bound the first-order error using these elliptical potential inequalities. This gives a general regret bound, but it does not enjoy the non-linearity in the leading term; namely, it gives $\text{Reg}_T=\mathcal{O}(d\sqrt{T}+\kappa^{-1}d^2)$ up to logarithmic factors.

Actually, we can directly upper bound the first-order error by reusing the result $(\star)$ from the uniform reward setting:

$$
\begin{aligned}
&\sum_{t=1}^T\sum_{i\in S_t}
p_t(i\mid S_t,\mathbf{w}^\star)
\left(
r_{ti}
-
\sum_{j\in S_t}
p_t(j\mid S_t,\mathbf{w}^\star)r_{tj}
\right)\\
&\leq
\sum_{t=1}^T\sum_{i\in S_t}
p_t(i\mid S_t,\mathbf{w}^\star)p_t(0\mid S_t,\mathbf{w}^\star)\\
&\leq
\sum_{t=1}^T\kappa_t^\star
+
\text{Reg}_T^\text{u},
\end{aligned}
$$

where $\text{Reg}_T^\text{u}$ denotes the regret under the uniform reward setting. This means

$$
\rho\text{Reg}_T^\text{u}
\leq
\text{Reg}_T
=
\sum_{t=1}^T
\left(
\sum_{i\in S_t^\star}p_t(i\mid S_t^\star,\mathbf{w}^\star)
-
\sum_{i\in S_t}p_t(i\mid S_t,\mathbf{w}^\star)
\right)r_{ti}
\leq
\text{Reg}_T^\text{u}.
$$

Therefore, we can apply the Cauchy-Schwarz inequality to bound the first-order error:

$$
\sum_{t=1}^T
\nabla \tilde{Q}(\mathbf{u}_t^\star)^\top(\mathbf{u}_t-\mathbf{u}_t^\star)
\leq
\beta_T(\delta)
\left(
d\log\left(1+\frac{T}{d\lambda}\right)
\sqrt{
\sum_{t=1}^T\kappa_t^\star
+
\frac{1}{\rho}\text{Reg}_T
}
\right).
$$

Solving the resulting quadratic inequality gives

$$
\text{Reg}_T
=
\tilde{\mathcal{O}}
\left(
\frac{d}{\rho}
\sqrt{\sum_{t=1}^T\kappa_t^\star}
+
\frac{1}{\kappa\rho}d^2
\right).
$$

### Other Technical Points

Confidence set. Set $\eta=\frac{1}{2}\log(K+1)+2$ and $\lambda =84\sqrt{2}d\eta$. Define

$$
\mathcal{C}_t(\delta)
=
\left\{
\mathbf{w}\in\mathcal{W}:
\|\mathbf{w}_t-\mathbf{w}\|_{\mathcal{H}_t}
\leq\beta_t(\delta)
\right\},
$$

where $\beta_t(\delta)=\mathcal{O}(\sqrt{d}\log t\log K)$. Then

$$
\Pr[\forall t\geq 1,\mathbf{w}^\star\in\mathcal{C}_t(\delta)]
\geq 1-\delta.
$$

Concentration. Let $\{\mathcal{F}_t\}_{t=1}^\infty$ be a filtration and let $\{z_t\}_{t=1}^\infty$ be a stochastic process with $z_t\in\mathbb{R}^K$ and $\|z_t\|_\infty\leq 1$, such that $z_t$ is $\mathcal{F}_t$-measurable. Let $\{\varepsilon_t\}_{t=1}^\infty$ be a martingale difference sequence such that $\varepsilon_t\in\mathbb{R}^K$ is $\mathcal{F}_{t+1}$-measurable. Furthermore, assume that conditional on $\mathcal{F}_t$, we have $\|\varepsilon_t\|_1\leq 2$ almost surely. Let

$$
\Sigma_t
=
\mathbb{E}[\varepsilon_t\varepsilon_t^\top\mid \mathcal{F}_t]
$$

and let $\lambda>0$. For any $t\geq 1$, define

$$
U_t=\sum_{s=1}^{t-1}\langle\varepsilon_s,z_s\rangle,
\quad
H_t=\lambda+\sum_{s=1}^{t-1}\|z_s\|_{\Sigma_s}^2.
$$

Then, for any $\delta\in(0,1]$,

$$
\Pr\left[
\exists t\geq 1,\,
U_t
\geq
\sqrt{H_t}
\left(
\frac{\sqrt{\lambda}}{4}
+
\frac{4}{\sqrt{\lambda}}\log\sqrt{\frac{H_t}{\lambda}}
+
\frac{4}{\sqrt{\lambda}}\log\frac{2}{\delta}
\right)
\right]
\leq
\delta.
$$

## References

- [Boudart et al., 2025, *Enjoying Non-linearity in Multinomial Logistic Bandits: A Minimax-Optimal Algorithm*](https://arxiv.org/abs/2507.05306).
- [Lee and Oh, 2024/2025, *Improved Online Confidence Bounds for Multinomial Logistic Bandits*](https://arxiv.org/abs/2502.10020).
- [Perivier and Goyal, 2022, *Dynamic pricing and assortment under a contextual MNL demand*](https://arxiv.org/abs/2110.10018).
- [Zhang and Sugiyama, 2023, *Online (Multinomial) Logistic Bandit: Improved Regret and Constant Computation Cost*](https://papers.nips.cc/paper_files/paper/2023/hash/5ef04392708bb2340cb9b7da41225660-Abstract-Conference.html).
- [Li et al., 2024, *Provably Efficient Reinforcement Learning with Multinomial Logit Function Approximation*](https://arxiv.org/abs/2405.17061).
