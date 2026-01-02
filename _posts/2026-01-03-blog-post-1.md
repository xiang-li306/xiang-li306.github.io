---
title: 'Sharpening or Discovery? The Role of RL in LLM Reasoning'
date: 2026-01-01
permalink: /posts/2026/01/rl-reasoning/
tags:
  - reinforcement learning
  - large language models
---



The year 2025 has witnessed a renewed surge of interest in reinforcement learning (RL), driven by the remarkable success of reasoning models such as OpenAI o1 and DeepSeek-R1, which leverage RL to achieve impressive performance. Nevertheless, the role of RL—more specifically, reinforcement learning with verifiable rewards (RLVR)—in large language models has remained a subject of long-standing debate.

Does RL merely act as a “sharpener,” reinforcing behaviors that the base (pre-trained) model already possesses? Or can it truly discover new behaviors beyond the base model’s capabilities, as classical RL does in domains like games and robotics? This question has persisted throughout 2025. In this blog post, we will examine several recent research papers that present different perspectives on this debate.

Sharpening or Discovery? The Role of RL in LLM Reasoning
======

### A Quick Review on RLVR

Reinforcement Learning with Verifiable Rewards (RLVR) refers to a class of reinforcement learning methods where the reward signal can be **automatically and unambiguously verified**. This setting has become especially important for reasoning-oriented large language models (LLMs), such as those used for math problem solving, coding, and symbolic reasoning.

Formally, let a language model define a policy $\pi_\theta(y\mid x)$, where $x$ is the input prompt (e.g., a math problem) and $y=(y_1,\ldots,y_T)$ is a generated response or reasoning trajectory. In RLVR, we assume access to a **verifier** $r(x,y)\in\{0,1\}$ which deterministically check whether the output is correct. Examples include:

- exact answer matching for math problems,
- compilation or unit tests for code,
- formal proof checkers.

Crucially, the reward does **not** depend on human preferences (as in RLHF) or learned reward models. The learning objective is typically to maximize the **KL-regularized** reward. That is,
$$
\max_\theta \mathbb{E}_{x\sim\mathcal{X},y\sim\pi_\theta}[r(x,y)]-\beta\cdot\mathbb{E}_{x\sim\mathcal{X}}[\mathrm{KL}(\pi_\theta(\cdot\mid x)\|\pi_\mathrm{ref}(\cdot\mid x))].
$$
In practice, RLVR is usually implemented using policy-gradient methods, such as PPO, GRPO, REINFORCE++, and DAPO. For instance, Deepseek R1 model uses **GRPO (Group-Relative Policy Optimization)** algorithm to train the model. It performs as following steps:

1. Given an input prompt \(x\), GRPO samples a *group* of $G$ responses in a single rollout: $\{y^{(1)}, y^{(2)}, \dots, y^{(G)}\} \sim \pi_\theta(\cdot \mid x)$.
2. Each response is then evaluated by a verifier, producing rewards $r^{(i)} = r(x, y^{(i)})$. 
3. Instead of using an explicit value function or critic (as in PPO), GRPO constructs a **group-relative advantage** by normalizing rewards within the group: $\hat{A}^{(i)} = r^{(i)} - \frac{1}{G}\sum_{j=1}^G r^{(j)}$.
4. This relative baseline removes common-mode effects across the group and significantly reduces gradient variance. The policy-gradient objective for GRPO can then be written as

$$
\mathcal{L}_{\text{GRPO}}(\theta)
=
\mathbb{E}_{x}\left[
\frac{1}{G}\sum_{i=1}^G
\hat{A}^{(i)} \log \pi_\theta(y^{(i)} \mid x)
\right].
$$

