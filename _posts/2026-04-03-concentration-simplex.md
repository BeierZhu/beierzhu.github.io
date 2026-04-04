---
layout: post
title: "On the Concentration of Empirical Means on the Simplex"
date: 2026-04-03
description:  
tags: [machine learning]
categories: [blog]
math: true
toc:
  beginning: true
---

Many statistical or machine learning problems start from the same basic question:

<p align="center"><code>Given finitely many samples, how accurately can we estimate an expectation?</code></p>

Let us begin with the simplest case.

## 1. Estimating the mean of a single random variable

Suppose $$X \in [0,1]$$ is a scalar random variable with mean

$$
\mu=\mathbb{E}[X].
$$

Given $$N$$ i.i.d. samples $$X^{(1)},\dots,X^{(N)}$$, the natural estimator is the sample mean

$$
\hat{\mu}=\frac1N\sum_{n=1}^N X^{(n)}.
$$

A classic result, [Hoeffding's inequality](/assets/pdf/Inequalities.pdf), tells us that for any $$\varepsilon > 0$$, 
the probability that the absolute difference between $$\mu$$ and $$\hat{\mu}$$ exceeds $$\varepsilon > 0$$  is at most $$2\exp(-2N\varepsilon^2)$$,

$$
\mathbb{P}\left(|\hat{\mu}-\mu| \geq \varepsilon \right) \leq 2\exp(-2N\varepsilon^2)
$$