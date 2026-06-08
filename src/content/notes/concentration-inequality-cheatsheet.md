
- [Preliminaries](#preliminaries)
  - [Markov Inequality](#markov-inequality)
  - [Chebyshev Inequality](#chebyshev-inequality)
  - [Chernoff Bound](#chernoff-bound)
- [Part I. Common Concentration Inequalities](#part-i-common-concentration-inequalities)
  - [Hoeffding Inequality](#hoeffding-inequality)
  - [Bernstein Inequality](#bernstein-inequality)
  - [Sub-Gaussian Concentration](#sub-gaussian-concentration)
  - [Sub-Exponential Concentration](#sub-exponential-concentration)
  - [Gaussian Maxima](#gaussian-maxima)
  - [Empirical Process + Union Bound](#empirical-process-union-bound)
- [Part II. Martingale and Adaptive Concentration](#part-ii-martingale-and-adaptive-concentration)
  - [Azuma-Hoeffding Inequality](#azuma-hoeffding-inequality)
  - [Freedman Inequality](#freedman-inequality)
  - [Self-Normalized Martingale Concentration](#self-normalized-martingale-concentration)
- [Part III. Matrix Concentration](#part-iii-matrix-concentration)
  - [Matrix Bernstein Inequality](#matrix-bernstein-inequality)
  - [Matrix Hoeffding Inequality](#matrix-hoeffding-inequality)
  - [Sample Covariance Concentration](#sample-covariance-concentration)
- [Part IV. Concentration for Functions of Random Variables](#part-iv-concentration-for-functions-of-random-variables)
  - [McDiarmid Inequality](#mcdiarmid-inequality)
  - [Gaussian Lipschitz Concentration](#gaussian-lipschitz-concentration)
  - [Chi-Square Concentration](#chi-square-concentration)
- [Part V. Uniform Concentration](#part-v-uniform-concentration)
  - [Symmetrization Inequality](#symmetrization-inequality)
  - [Rademacher Complexity Bound](#rademacher-complexity-bound)
  - [Empirical Rademacher Complexity Bound](#empirical-rademacher-complexity-bound)
  - [Ledoux-Talagrand Contraction Inequality](#ledoux-talagrand-contraction-inequality)
  - [Dudley Entropy Integral](#dudley-entropy-integral)
  - [Talagrand's Concentration Inequality](#talagrands-concentration-inequality)
  - [Talagrand-Bousquet Inequality](#talagrand-bousquet-inequality)
- [Part VI. Heavy-Tailed Concentration](#part-vi-heavy-tailed-concentration)
  - [Median-of-Means Concentration](#median-of-means-concentration)
  - [Catoni-Type Concentration](#catoni-type-concentration)
- [Part VII. Other Concentration Inequalities](#part-vii-other-concentration-inequalities)
  - [Bennett Inequality](#bennett-inequality)
  - [Cantelli Inequality](#cantelli-inequality)
  - [Paley-Zygmund Inequality](#paley-zygmund-inequality)
  - [Sudakov-Fernique Inequality](#sudakov-fernique-inequality)
  - [Bernoulli KL Concentration](#bernoulli-kl-concentration)

## Preliminaries

### Markov Inequality

Let $X$ be a nonnegative random variable (that is, $X\ge 0$ almost surely). Then, for any $t>0$,
$$
\mathbb{P}[X\ge t]\le\frac{\mathbb{E}[X]}{t}.
$$

### Chebyshev Inequality

Let $X$ have mean $\mu$ and variance $\sigma^2$. Then for any $t>0$,
$$
\mathbb{P}[|X-\mu|\ge t]\le\frac{\sigma^2}{t^2}.
$$
Equivalently, $\mathbb{P}[|X-\mu|\ge k\sigma]\le k^{-2}$.

### Chernoff Bound

Let $X=X_1+\ldots+X_n$, where the $X_i$'s are independent. Consider the Bernoulli case where $X_i\in\{0,1\}$ with $\mathbb{P}[X_i=1]=p$ (that is $X\sim B(n,p)$). Let $\mu=np$. Then for $0<\delta\le 1$,
$$
\mathbb{P}[X\ge (1+\delta)\mu]\le e^{-\mu\delta^2/3},\qquad \mathbb{P}[X\le (1-\delta)\mu]\le e^{-\mu\delta^2/2}.
$$
A more precise version is that, for independent Bernoulli sums with mean $\mu$,
$$
\mathbb{P}[X\ge (1+\delta)\mu]\le\left(\frac{e^\delta}{(1+\delta)^{1+\delta}}\right)^\mu,\qquad \mathbb{P}[X\le(1-\delta)\mu]\le\left(\frac{e^{-\delta}}{(1-\delta)^{1-\delta}}\right)^{\mu}.
$$

## Part I. Common Concentration Inequalities

### Hoeffding Inequality

**Condition:** let $X_1,\ldots,X_n$ be independent random variables such that $X_i\in[a_i,b_i]$ almost surely, and define $R_n^2=\sum_{i=1}^n(b_i-a_i)^2$.

For any $\delta\in(0,1)$, with probability at least $1-\delta$,

$$
\left|\sum_{i=1}^n\left(X_i-\mathbb{E}[X_i]\right)\right|
\le
\sqrt{\frac{R_n^2}{2}\log\frac{2}{\delta}}.
$$

In particular, if $X_1,\ldots,X_n$ are independent and $X_i\in[a,b]$ almost surely for all $i$, then with probability at least $1-\delta$,

$$
\left|\frac1n\sum_{i=1}^n X_i-\frac1n\sum_{i=1}^n\mathbb{E}[X_i]\right|
\le
(b-a)\sqrt{\frac{\log(2/\delta)}{2n}}.
$$

### Bernstein Inequality

**Condition:** let $Y_1,\ldots,Y_n$ be independent random variables such that $\mathbb{E}[Y_i]=0$, $|Y_i|\le M$ almost surely, and $\sum_{i=1}^n\mathrm{Var}[Y_i]\le v$.

For any $\delta\in(0,1)$, with probability at least $1-\delta$,

$$
\left|\sum_{i=1}^nY_i\right|
\le
\sqrt{2v\log\frac{2}{\delta}}
+
\frac{2M}{3}\log\frac{2}{\delta}.
$$

If additionally $\frac1n\sum_{i=1}^n\mathrm{Var}[Y_i]\le\sigma^2$, then with probability at least $1-\delta$,

$$
\left|\frac1n\sum_{i=1}^nY_i\right|
\le
\sqrt{\frac{2\sigma^2\log(2/\delta)}{n}}
+
\frac{2M\log(2/\delta)}{3n}.
$$

### Sub-Gaussian Concentration

**Condition:** let $X_1,\ldots,X_n$ be independent mean-zero random variables such that each $X_i$ is sub-Gaussian with variance proxy $\sigma_i^2$, i.e., $\mathbb{E}\left[\exp(\lambda X_i)\right]
\le
\exp\left(\frac{\lambda^2\sigma_i^2}{2}\right)$ for all $\lambda\in\mathbb{R}$. 

Define $\bar\sigma_n^2=\frac1{n^2}\sum_{i=1}^n\sigma_i^2$. For any $\delta\in(0,1)$, with probability at least $1-\delta$,

$$
\left|\frac1n\sum_{i=1}^nX_i\right|
\le
\sqrt{2\bar\sigma_n^2\log\frac{2}{\delta}}
=
\frac{\sqrt{2\log(2/\delta)\sum_{i=1}^n\sigma_i^2}}{n}.
$$

### Sub-Exponential Concentration

**Condition:** let $X_1,\ldots,X_n$ be independent mean-zero random variables satisfying $\mathbb{E}[\exp(\lambda X_i)]\le\exp(\nu_i^2\lambda^2/2)$ for all $|\lambda|\le1/\alpha_i$; equivalently, a sub-exponential tail condition has the form $\mathbb{P}[|X_i|\ge t]\le c_1\exp(-c_2t)$ for absolute constants $c_1,c_2>0$ after rescaling.

Define $V=\sum_{i=1}^n\nu_i^2$ and $\alpha=\max_{1\le i\le n}\alpha_i$. For any $\delta\in(0,1)$, with probability at least $1-\delta$,

$$
\left|\sum_{i=1}^nX_i\right|
\le
\sqrt{2V\log\frac{2}{\delta}}
+
2\alpha\log\frac{2}{\delta}.
$$

Equivalently, with probability at least $1-\delta$,

$$
\left|\frac1n\sum_{i=1}^nX_i\right|
\le
\sqrt{\frac{2V\log(2/\delta)}{n^2}}
+
\frac{2\alpha\log(2/\delta)}{n}.
$$

### Gaussian Maxima

**Condition:** let $Z_1,\ldots,Z_N$ be centered Gaussian random variables, not necessarily independent, such that $\max_{1\le j\le N}\mathrm{Var}[Z_j]\le\sigma^2$.

The expectation satisfies

$$
\mathbb{E}\left[\max_{1\le j\le N}Z_j\right]
\le
\sigma\sqrt{2\log N}.
$$

For any $\delta\in(0,1)$, with probability at least $1-\delta$,

$$
\max_{1\le j\le N}Z_j
\le
\sigma\sqrt{2\log N}
+
\sigma\sqrt{2\log\frac1\delta}.
$$

### Empirical Process + Union Bound

**Condition:** let $X_1,\ldots,X_n$ be i.i.d. samples, let $\mathcal{F}$ be a finite class of measurable functions with $|\mathcal{F}|=N$, and assume $f(X_i)\in[a,b]$ almost surely for every $f\in\mathcal{F}$ and every $i$.

For any $\delta\in(0,1)$, with probability at least $1-\delta$,

$$
\sup_{f\in\mathcal{F}}
\left|
\frac1n\sum_{i=1}^n f(X_i)-\mathbb{E}[f(X)]
\right|
\le
(b-a)\sqrt{\frac{\log(2N/\delta)}{2n}}.
$$

## Part II. Martingale and Adaptive Concentration

### Azuma-Hoeffding Inequality

**Condition:** let $(\mathcal{F}_t)_{t=0}^T$ be a filtration and let $X_1,\ldots,X_T$ be a martingale difference sequence such that $X_t$ is $\mathcal{F}_t$-measurable, $\mathbb{E}[X_t\mid\mathcal{F}_{t-1}]=0$, and $|X_t|\le c_t$ almost surely.

For any $\delta\in(0,1)$, with probability at least $1-\delta$,

$$
\left|\sum_{t=1}^T X_t\right|
\le
\sqrt{2\left(\sum_{t=1}^T c_t^2\right)\log\frac{2}{\delta}}.
$$

### Freedman Inequality

**Condition:** let $(\mathcal{F}_t)_{t=0}^T$ be a filtration and let $X_1,\ldots,X_T$ be a martingale difference sequence such that $X_t$ is $\mathcal{F}_t$-measurable, $\mathbb{E}[X_t\mid\mathcal{F}_{t-1}]=0$, $X_t\le M$ almost surely, and $V_T=\sum_{t=1}^T\mathbb{E}[X_t^2\mid\mathcal{F}_{t-1}]$.

For any $v>0$ and $\delta\in(0,1)$,

$$
\mathbb{P}\left[
\sum_{t=1}^T X_t
\ge
\sqrt{2v\log\frac1\delta}
+
\frac{M}{3}\log\frac1\delta
\ \text{and}\
V_T\le v
\right]
\le
\delta.
$$

Equivalently, for any fixed $v>0$, with probability at least $1-\delta$, on the event $V_T\le v$,

$$
\sum_{t=1}^T X_t
\le
\sqrt{2v\log\frac1\delta}
+
\frac{M}{3}\log\frac1\delta.
$$

### Self-Normalized Martingale Concentration

**Condition:** let $(\mathcal{F}_t)_{t=0}^T$ be a filtration, let $x_t\in\mathbb{R}^d$ be $\mathcal{F}_{t-1}$-measurable with $\|x_t\|_2\le L$, let $\eta_t$ be conditionally $R$-sub-Gaussian in the sense that $\mathbb{E}[\exp(\lambda\eta_t)\mid\mathcal{F}_{t-1}]\le\exp(\lambda^2R^2/2)$ for all $\lambda\in\mathbb{R}$, and define $V_t=\lambda I_d+\sum_{s=1}^t x_sx_s^\top$ for some regularization parameter $\lambda>0$.

For any $\delta\in(0,1)$, with probability at least $1-\delta$, simultaneously for all $t\ge0$,

$$
\left\|
\sum_{s=1}^t x_s\eta_s
\right\|_{V_t^{-1}}
\le
R\sqrt{
2\log\left(
\frac{\det(V_t)^{1/2}}{\det(\lambda I_d)^{1/2}\delta}
\right)
}.
$$

Using $\|x_t\|_2\le L$, the preceding bound implies that with probability at least $1-\delta$, simultaneously for all $t\ge0$,

$$
\left\|
\sum_{s=1}^t x_s\eta_s
\right\|_{V_t^{-1}}
\le
R\sqrt{
d\log\left(1+\frac{tL^2}{\lambda d}\right)
+2\log\frac1\delta
}.
$$

## Part III. Matrix Concentration

### Matrix Bernstein Inequality

**Condition:** let $Y_1,\ldots,Y_n$ be independent self-adjoint random matrices in $\mathbb{R}^{d\times d}$ such that $\mathbb{E}[Y_i]=0$, $\|Y_i\|_{\mathrm{op}}\le M$ almost surely, and $v=\left\|\sum_{i=1}^n\mathbb{E}[Y_i^2]\right\|_{\mathrm{op}}$.

For any $\delta\in(0,1)$, with probability at least $1-\delta$,

$$
\left\|
\sum_{i=1}^nY_i
\right\|_{\mathrm{op}}
\le
\sqrt{2v\log\frac{2d}{\delta}}
+
\frac{2M}{3}\log\frac{2d}{\delta}.
$$

### Matrix Hoeffding Inequality

**Condition:** let $Y_1,\ldots,Y_n$ be independent self-adjoint random matrices in $\mathbb{R}^{d\times d}$ such that $\mathbb{E}[Y_i]=0$, $\|Y_i\|_{\mathrm{op}}\le M_i$ almost surely, and $\sigma^2=\sum_{i=1}^nM_i^2$.

For any $\delta\in(0,1)$, with probability at least $1-\delta$,

$$
\left\|
\sum_{i=1}^nY_i
\right\|_{\mathrm{op}}
\le
\sqrt{8\sigma^2\log\frac{2d}{\delta}}.
$$

### Sample Covariance Concentration

**Condition:** let $X_1,\ldots,X_n\in\mathbb{R}^d$ be i.i.d. mean-zero sub-Gaussian random vectors with covariance matrix $\Sigma=\mathbb{E}[X_iX_i^\top]$, and assume there is a constant $K\ge1$ such that $\|\langle u,X_i\rangle\|_{\psi_2}\le K\sqrt{u^\top\Sigma u}$ for every $u\in\mathbb{R}^d$.

There exists a universal constant $C>0$ such that, for any $\delta\in(0,1)$, with probability at least $1-\delta$,

$$
\left\|
\frac1n\sum_{i=1}^nX_iX_i^\top-\Sigma
\right\|_{\mathrm{op}}
\le
C K^2\|\Sigma\|_{\mathrm{op}}
\left(
\sqrt{\frac{d+\log(2/\delta)}{n}}
+
\frac{d+\log(2/\delta)}{n}
\right).
$$

## Part IV. Concentration for Functions of Random Variables

### McDiarmid Inequality

**Condition:** let $X_1,\ldots,X_n$ be independent random variables and let $f$ satisfy the bounded differences condition $|f(x_1,\ldots,x_i,\ldots,x_n)-f(x_1,\ldots,x_i',\ldots,x_n)|\le c_i$ for every $i$ and every pair $x_i,x_i'$.

For any $\delta\in(0,1)$, with probability at least $1-\delta$,

$$
\left|
f(X_1,\ldots,X_n)-\mathbb{E}[f(X_1,\ldots,X_n)]
\right|
\le
\sqrt{\frac12\left(\sum_{i=1}^n c_i^2\right)\log\frac{2}{\delta}}.
$$

### Gaussian Lipschitz Concentration

**Condition:** let $g\sim N(0,I_d)$ and let $f:\mathbb{R}^d\to\mathbb{R}$ be $L$-Lipschitz with respect to the Euclidean norm, i.e. $|f(x)-f(y)|\le L\|x-y\|_2$ for all $x,y\in\mathbb{R}^d$.

For any $\delta\in(0,1)$, with probability at least $1-\delta$,

$$
\left|
f(g)-\mathbb{E}[f(g)]
\right|
\le
L\sqrt{2\log\frac{2}{\delta}}.
$$

### Chi-Square Concentration

**Condition:** let $Z\sim\chi_d^2$.

For any $\delta\in(0,1)$, with probability at least $1-\delta$,

$$
d-2\sqrt{d\log\frac{2}{\delta}}
\le
Z
\le
 d+2\sqrt{d\log\frac{2}{\delta}}+2\log\frac{2}{\delta}.
$$

## Part V. Uniform Concentration

### Symmetrization Inequality

**Condition:** let $X_1,\ldots,X_n$ be i.i.d. samples, let $\mathcal{F}$ be a measurable function class with $f(X)\in[a,b]$ almost surely for every $f\in\mathcal{F}$, and let $\varepsilon_1,\ldots,\varepsilon_n$ be independent Rademacher random variables independent of the data.

For any $\delta\in(0,1)$, with probability at least $1-\delta$,

$$
\sup_{f\in\mathcal{F}}
\left|
\frac1n\sum_{i=1}^n f(X_i)-\mathbb{E}[f(X)]
\right|
\le
2\mathbb{E}_{X,\varepsilon}\left[
\sup_{f\in\mathcal{F}\cup(-\mathcal{F})}
\frac1n\sum_{i=1}^n\varepsilon_i f(X_i)
\right]
+
(b-a)\sqrt{\frac{\log(1/\delta)}{2n}}.
$$

### Rademacher Complexity Bound

**Condition:** let $X_1,\ldots,X_n$ be i.i.d. samples, let $\mathcal{F}$ be a measurable function class with $f(X)\in[a,b]$ almost surely for every $f\in\mathcal{F}$, and define $\mathfrak{R}_n(\mathcal{G})=\mathbb{E}_{X,\varepsilon}[\sup_{g\in\mathcal{G}}\frac1n\sum_{i=1}^n\varepsilon_i g(X_i)]$.

For any $\delta\in(0,1)$, with probability at least $1-\delta$,

$$
\sup_{f\in\mathcal{F}}
\left|
\frac1n\sum_{i=1}^n f(X_i)-\mathbb{E}[f(X)]
\right|
\le
2\mathfrak{R}_n(\mathcal{F}\cup(-\mathcal{F}))
+
(b-a)\sqrt{\frac{\log(1/\delta)}{2n}}.
$$

### Empirical Rademacher Complexity Bound

**Condition:** let $X_1,\ldots,X_n$ be i.i.d. samples, let $\mathcal{F}$ be a measurable function class with $f(X)\in[a,b]$ almost surely for every $f\in\mathcal{F}$, and define $\widehat{\mathfrak{R}}_n(\mathcal{G})=\mathbb{E}_{\varepsilon}[\sup_{g\in\mathcal{G}}\frac1n\sum_{i=1}^n\varepsilon_i g(X_i)]$ conditional on the observed data.

For any $\delta\in(0,1)$, with probability at least $1-\delta$,

$$
\sup_{f\in\mathcal{F}}
\left|
\frac1n\sum_{i=1}^n f(X_i)-\mathbb{E}[f(X)]
\right|
\le
2\widehat{\mathfrak{R}}_n(\mathcal{F}\cup(-\mathcal{F}))
+3(b-a)\sqrt{\frac{\log(2/\delta)}{2n}}.
$$

### Ledoux-Talagrand Contraction Inequality

**Condition:** fix $x_1,\ldots,x_n$, let $\varepsilon_1,\ldots,\varepsilon_n$ be independent Rademacher random variables, let $\mathcal{F}$ be a class of real-valued functions on the sample points, and let $\phi_i:\mathbb{R}\to\mathbb{R}$ be $L$-Lipschitz functions satisfying $\phi_i(0)=0$.

Conditionally on $x_1,\ldots,x_n$,

$$
\mathbb{E}_{\varepsilon}\left[
\sup_{f\in\mathcal{F}}
\frac1n\sum_{i=1}^n\varepsilon_i\phi_i(f(x_i))
\right]
\le
L\mathbb{E}_{\varepsilon}\left[
\sup_{f\in\mathcal{F}}
\frac1n\sum_{i=1}^n\varepsilon_i f(x_i)
\right].
$$

### Dudley Entropy Integral

**Condition:** fix $x_1,\ldots,x_n$, let $\varepsilon_1,\ldots,\varepsilon_n$ be independent Rademacher random variables, let $\mathcal{F}$ be a class of functions with empirical diameter $D=\sup_{f,g\in\mathcal{F}}\sqrt{\frac1n\sum_{i=1}^n(f(x_i)-g(x_i))^2}$, and let $N(u,\mathcal{F},L_2(x_{1:n}))$ be the covering number under the empirical $L_2$ metric.

Conditionally on $x_1,\ldots,x_n$,

$$
\mathbb{E}_{\varepsilon}\left[
\sup_{f\in\mathcal{F}}
\frac1n\sum_{i=1}^n\varepsilon_i f(x_i)
\right]
\le
\inf_{\alpha>0}
\left\{
4\alpha+
\frac{12}{\sqrt n}\int_{\alpha}^{D}
\sqrt{\log N(u,\mathcal{F},L_2(x_{1:n}))}\,du
\right\}.
$$

### Talagrand's Concentration Inequality

**Condition:** let $X_1,\ldots,X_n$ be independent random variables, let $\mathcal{F}$ be a countable class of measurable functions satisfying $|f(X_i)-\mathbb{E}[f(X_i)]|\le b$ almost surely and $\frac1n\sum_{i=1}^n\mathrm{Var}[f(X_i)]\le\sigma^2$ for all $f\in\mathcal{F}$.

There exists a universal constant $C>0$ such that, for any $\delta\in(0,1)$, with probability at least $1-\delta$,

$$
\sup_{f\in\mathcal{F}}
\left|
\frac1n\sum_{i=1}^n f(X_i)-\frac1n\sum_{i=1}^n\mathbb{E}[f(X_i)]
\right|
\le
C\left(
\mathbb{E}\left[
\sup_{f\in\mathcal{F}}
\left|
\frac1n\sum_{i=1}^n f(X_i)-\frac1n\sum_{i=1}^n\mathbb{E}[f(X_i)]
\right|
\right]
+
\sqrt{\frac{\sigma^2\log(1/\delta)}{n}}
+
\frac{b\log(1/\delta)}{n}
\right).
$$

### Talagrand-Bousquet Inequality

**Condition:** let $X_1,\ldots,X_n$ be independent random variables, let $\mathcal{F}$ be a countable class of measurable functions satisfying $\mathbb{E}[f(X_i)]=0$, $f(X_i)\le b$ almost surely, and $\frac1n\sup_{f\in\mathcal{F}}\sum_{i=1}^n\mathbb{E}[f(X_i)^2]\le\sigma^2$; define $Z=\sup_{f\in\mathcal{F}}\sum_{i=1}^n f(X_i)$.

For any $\delta\in(0,1)$, with probability at least $1-\delta$,

$$
\frac{Z}{n}
\le
\frac{\mathbb{E}[Z]}{n}
+
\sqrt{
\frac{2\log(1/\delta)}{n}
\left(
\sigma^2+\frac{2b\mathbb{E}[Z]}{n}
\right)
}
+
\frac{b\log(1/\delta)}{3n}.
$$

## Part VI. Heavy-Tailed Concentration

### Median-of-Means Concentration

**Condition:** let $X_1,\ldots,X_n$ be i.i.d. real-valued random variables with $\mathbb{E}[X_i]=\mu$ and $\mathrm{Var}[X_i]\le\sigma^2$; assume $K$ divides $n$, split the data into $K$ equal blocks of size $m=n/K$, let $\bar X_k$ be the empirical mean of block $k$, and define $\widehat\mu_{\mathrm{MOM}}=\mathrm{median}(\bar X_1,\ldots,\bar X_K)$.

If $K\ge8\log(1/\delta)$, then with probability at least $1-\delta$,

$$
|\widehat\mu_{\mathrm{MOM}}-\mu|
\le
2\sigma\sqrt{\frac{K}{n}}.
$$

### Catoni-Type Concentration

**Condition:** let $X_1,\ldots,X_n$ be i.i.d. real-valued random variables with $\mathbb{E}[X_i]=\mu$ and $\mathrm{Var}[X_i]\le\sigma^2$, let $\psi$ be nondecreasing and satisfy $-\log(1-x+x^2/2)\le\psi(x)\le\log(1+x+x^2/2)$, and let $\widehat\mu_{\mathrm{Catoni}}$ solve $\sum_{i=1}^n\psi(\alpha(X_i-\widehat\mu_{\mathrm{Catoni}}))=0$ with a valid tuning parameter $\alpha$ depending on $n,\sigma^2,\delta$.

For $n>2\log(2/\delta)$, one standard tuning gives, with probability at least $1-\delta$,

$$
|\widehat\mu_{\mathrm{Catoni}}-\mu|
\le
\sigma\sqrt{
\frac{2\log(2/\delta)}{n-2\log(2/\delta)}
}.
$$

## Part VII. Other Concentration Inequalities

### Bennett Inequality

**Condition:** let $X_1,\ldots,X_n$ be independent mean-zero random variables such that $|X_i|\le M$ almost surely, let $v=\sum_{i=1}^n\mathrm{Var}[X_i]$, and define $h(u)=(1+u)\log(1+u)-u$ for $u\ge0$.

For any $\delta\in(0,1)$, with probability at least $1-\delta$,

$$
\left|\sum_{i=1}^nX_i\right|
\le
\frac{v}{M}\,
h^{-1}\left(
\frac{M^2\log(2/\delta)}{v}
\right),
$$

where $h^{-1}$ denotes the inverse of $h$ on $[0,\infty)$.

### Cantelli Inequality

**Condition:** let $X$ be a real-valued random variable with $\mathbb{E}[X]=\mu$ and $\mathrm{Var}[X]=\sigma^2<\infty$.

For any $\delta\in(0,1)$, with probability at least $1-\delta$, one of the following two holds:

$$
X-\mu
\le
\sigma\sqrt{\frac{1-\delta}{\delta}},\qquad \mu-X
\le
\sigma\sqrt{\frac{1-\delta}{\delta}}.
$$

### Paley-Zygmund Inequality

**Condition:** let $X\ge0$ be a random variable with $0<\mathbb{E}[X^2]<\infty$.

For any $\theta\in(0,1)$, with probability at least $(1-\theta)^2\frac{\mathbb{E}[X]^2}{\mathbb{E}[X^2]}$,

$$
X
\ge
\theta\mathbb{E}[X].
$$

### Sudakov-Fernique Inequality

**Condition:** let $(X_t)_{t\in T}$ and $(Y_t)_{t\in T}$ be centered Gaussian processes indexed by the same set $T$, and assume $\mathbb{E}[(X_s-X_t)^2]\le\mathbb{E}[(Y_s-Y_t)^2]$ for all $s,t\in T$.

Then

$$
\mathbb{E}\left[\sup_{t\in T}X_t\right]
\le
\mathbb{E}\left[\sup_{t\in T}Y_t\right].
$$

### Bernoulli KL Concentration

**Condition:** let $X_1,\ldots,X_n$ be i.i.d. Bernoulli$(p)$ random variables, let $\widehat p=\frac1n\sum_{i=1}^nX_i$, and define $D_\mathrm{KL}(q\|p)=q\log(q/p)+(1-q)\log((1-q)/(1-p))$ for $q,p\in(0,1)$, with the standard continuous extensions at the boundary.

For any $\delta\in(0,1)$, with probability at least $1-\delta$,

$$
D_\mathrm{KL}(\widehat p\|p)
\le
\frac{\log((n+1)/\delta)}{n}.
$$

Equivalently, for any fixed $q>p$ and any fixed $r<p$,

$$
\mathbb{P}[\widehat p\ge q]
\le
\exp\left(-n\,D_\mathrm{KL}(q\|p)\right),
\qquad
\mathbb{P}[\widehat p\le r]
\le
\exp\left(-n\,D_\mathrm{KL}(r\|p)\right).
$$

## References

- [Wainwright, 2019, *High-Dimensional Statistics: A Non-Asymptotic Viewpoint*](https://www.cambridge.org/core/books/highdimensional-statistics/8A91ECEEC38F46DAB53E9FF8757C7A4E). Cambridge University Press.
