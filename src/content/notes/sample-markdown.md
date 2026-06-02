# A Short Note on Policy Gradients, RLHF, RLVR, PPO, DPO, and GRPO

This sample note is meant to test both Markdown images and mathematical formulas. Images used by a note can live in a matching folder under `public/notes`. For this note, the image below is served from `/notes/sample-markdown/myphoto.png`.

![Sample note image](/notes/sample-markdown/myphoto.png)

## Policy Optimization Setup

Consider a parameterized policy $\pi_\theta(a \mid s)$ in a discounted Markov decision process. The usual objective is

$$
J(\theta)
=
\mathbb{E}_{\tau \sim \pi_\theta}
\left[
\sum_{t=0}^{\infty} \gamma^t r(s_t,a_t)
\right],
$$

where $\tau=(s_0,a_0,s_1,a_1,\ldots)$ is a trajectory sampled from the current policy. The policy gradient theorem gives

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{s \sim d^{\pi_\theta}, a \sim \pi_\theta}
\left[
\nabla_\theta \log \pi_\theta(a \mid s) Q^{\pi_\theta}(s,a)
\right].
$$

In practice, $Q^{\pi_\theta}(s,a)$ is often replaced by an advantage estimate

$$
A^{\pi_\theta}(s,a)
=
Q^{\pi_\theta}(s,a)-V^{\pi_\theta}(s),
$$

which preserves the gradient direction while reducing variance:

$$
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
\nabla_\theta \log \pi_\theta(a \mid s) A^{\pi_\theta}(s,a)
\right].
$$

## PPO as a Conservative Policy Gradient Method

PPO updates the policy using samples from an old policy $\pi_{\theta_{\mathrm{old}}}$. Define the probability ratio

$$
\rho_t(\theta)
=
\frac{\pi_\theta(a_t \mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)}.
$$

The clipped PPO objective is

$$
L^{\mathrm{PPO}}(\theta)
=
\mathbb{E}_t
\left[
\min
\left(
\rho_t(\theta)\hat A_t,
\operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right].
$$

The clipping term discourages updates that move the new policy too far from the data-generating policy. This is especially useful when reward estimates are noisy or the advantage estimator is imperfect.

## RLHF and RLVR

In RLHF, a language model policy $\pi_\theta(y \mid x)$ generates a response $y$ to a prompt $x$, and a learned reward model $r_\phi(x,y)$ supplies the optimization signal. A common objective includes a KL penalty to keep the policy close to a reference model $\pi_{\mathrm{ref}}$:

$$
\max_\theta
\mathbb{E}_{x,y \sim \pi_\theta}
\left[
r_\phi(x,y)
-
\beta \operatorname{KL}
\left(
\pi_\theta(\cdot \mid x)
\;\|\;
\pi_{\mathrm{ref}}(\cdot \mid x)
\right)
\right].
$$

RLVR uses verifiable rewards instead of, or in addition to, learned preference rewards. For example, in math or code tasks, the reward may be computed from answer correctness:

$$
r(x,y)
=
\mathbf{1}\{\mathrm{verify}(x,y)=\mathrm{correct}\}.
$$

The optimization problem still resembles policy gradient learning, but the reward source changes. This difference matters because verifiable rewards can be sparse, high variance, and strongly shaped by the task distribution.

## DPO as Preference-Based Optimization

DPO starts from pairwise preference data $(x,y_w,y_l)$, where $y_w$ is preferred to $y_l$. Instead of explicitly training a reward model and then running RL, DPO optimizes a classification-style objective:

$$
L^{\mathrm{DPO}}(\theta)
=
-
\mathbb{E}
\left[
\log \sigma
\left(
\beta
\left[
\log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)}
-
\log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)}
\right]
\right)
\right].
$$

Intuitively, DPO increases the relative likelihood of preferred responses while controlling movement away from the reference model.

## GRPO and Group-Relative Advantages

GRPO is often described as a PPO-style method that avoids a separate learned value function by comparing several sampled responses for the same prompt. Suppose for a prompt $x$ we sample a group of responses $\{y_i\}_{i=1}^G$ with rewards $\{r_i\}_{i=1}^G$. A simple group-relative advantage is

$$
\hat A_i
=
\frac{r_i-\operatorname{mean}(r_1,\ldots,r_G)}
{\operatorname{std}(r_1,\ldots,r_G)+\delta}.
$$

Then a PPO-like objective can use the response-level ratio

$$
\rho_i(\theta)
=
\frac{\pi_\theta(y_i \mid x)}
{\pi_{\theta_{\mathrm{old}}}(y_i \mid x)}
$$

and optimize a clipped surrogate similar to

$$
L^{\mathrm{GRPO}}(\theta)
=
\mathbb{E}
\left[
\min
\left(
\rho_i(\theta)\hat A_i,
\operatorname{clip}(\rho_i(\theta),1-\epsilon,1+\epsilon)\hat A_i
\right)
-
\beta
\operatorname{KL}
\left(
\pi_\theta(\cdot \mid x)
\;\|\;
\pi_{\mathrm{ref}}(\cdot \mid x)
\right)
\right].
$$

This note is only a compact sketch, but it exercises the main rendering cases I care about for the website: paragraphs, links, images, inline math like $\pi_\theta(a \mid s)$, and multi-line display equations.
