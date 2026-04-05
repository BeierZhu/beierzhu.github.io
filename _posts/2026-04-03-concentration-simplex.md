---
layout: post
title: "On the Concentration of Empirical Means on the Simplex"
date: 2026-04-03
description:  This post starts from the familiar problem of estimating the mean of a bounded random variable, and then follows what changes as the object of interest becomes more structured. For simplex-valued means, it is more natural to think in terms of distributional concentration, which leads to sharper and more meaningful bounds. Weissman's inequality captures this especially well in the one-hot categorical setting, while a McDiarmid-based argument extends the picture to general soft simplex-valued samples. 
tags: [machine learning]
categories: [blog]
math: true
toc:
  beginning: true
---

Many statistical or machine learning problems start from the same basic question:

<p align="center"><code>Given finitely many samples, how accurately can we estimate an expectation?</code></p>

Let us begin with the simplest case.

---
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
\mathbb{P}\left(|\hat{\mu}-\mu| \geq \varepsilon \right) \leq 2\exp(-2N\varepsilon^2)。
$$

This  captures the basic message of finite-sample mean estimation: 
**as the number of samples grows, the empirical mean concentrates around the true mean, with deviation on the order of $$1/\sqrt{N}$$**.

So far, everything is completely standard. We are estimating one number, and Hoeffding gives a clean answer.

---
## 2. From a scalar to a vector

Now let us move from a scalar to a $$K$$-dimensional random vector $$
X=(X_1,...,X_K) \in \mathbb{R}^K,
$$
and suppose we want to estimate its mean
$$
\boldsymbol{\mu} = \mathbb{E}[X].
$$
Given i.i.d. samples $$X^{(1)},\dots,X^{(N)}$$, we again use the empirical mean
$$
\hat{\boldsymbol{\mu}}=\frac1N\sum_{n=1}^N X^{(n)}.
$$

At first glance, the most natural strategy is coordinate-wise estimation.
If each coordinate satisfies $$X_i \in [0,1]$$, then Hoeffding applies coordinate by coordinate:

$$
\mathbb{P}\left(|\hat{\mu}_i-\mu_i| \geq \varepsilon \right) \leq 2\exp(-2N\varepsilon^2)。
$$

This is fine as far as it goes. But it only tells us that each coordinate is accurate separately. It does **not** yet say how well the whole vector is estimated.
If we care about the entire vector, a natural quantity is
$$
\|\hat{\boldsymbol{\mu}}-\boldsymbol{\mu}\|_1=\sum_{i=1}^K|\hat{\mu}_i-\mu_i|.
$$

One can still start from coordinate-wise Hoeffding and then combine the results using a union bound.

<div class="theorem-block">
<div class="theorem-title">Theorem 1 Hoeffding's inequality for vector samples </div>
<div class="theorem-content" markdown="1">
  
For any $$\varepsilon > 0$$, the probability that the $$L_1$$ distance between $${\boldsymbol{\mu}}$$ and $$\hat{\boldsymbol{\mu}}$$ exceeds $$\varepsilon$$ is at most $$2K\exp(-\frac{2N\epsilon^2}{K^2})$$:

$$
\mathbb{P}\left(\|\hat{\boldsymbol{\mu}}-\boldsymbol{\mu}\|_1 \geq \varepsilon \right) \leq 2K\exp(-\frac{2N\varepsilon^2}{K^2}).
$$

</div>
</div>

<details class="proof-block">
<summary class="proof-title">Proof (Click to expand)</summary>
<div class="proof-content" markdown="1">
If
$$
\|\hat{\boldsymbol{\mu}}-\boldsymbol{\mu}\|_1 \ge t,
$$
then at least one coordinate must satisfy
$$
|\hat{\mu}_i-\mu_i| \ge \frac{\varepsilon}{K},
$$
since otherwise all $$K$$ coordinates would be smaller than $$\varepsilon/K$$, and their sum could not reach $$\varepsilon$$. Hence
  
$$
\mathbb{P}\!\left(\|\hat{\boldsymbol{\mu}}-\boldsymbol{\mu}\|_1 \ge t\right)
\le
\sum_{i=1}^K
  \mathbb{P}\!\left(|\hat{\mu}_i-\mu_i| \ge \frac{\varepsilon}{K}\right)
\le
2K\exp\!\left(-\frac{2N\varepsilon^2}{K^2}\right),
$$

where the last step follows from Hoeffding's inequality applied coordinate-wise.

  <div class="proof-end">□</div>
  </div>
  </details>

But this approach is somewhat crude: it treats the vector simply as a collection of **unrelated coordinates**, without using any additional structure the vector might have. 
And in many problems, that **structure matters**.

---
## 3. A special case: simplex-valued random vectors

Now suppose that $$X=(X_1,...,X_K)$$ is not arbitrary, but lies in the probability simplex ($$\Delta^{K-1}$$):

$$
X_i \geq 0,\quad \sum_{i=1}^K X_i= 1.
$$

This setting appears naturally when $$X$$ represents a label distribution, a soft label, or a probabilistic prediction. 
In this case, the mean and the empirical mean also lie in the simplex

$$
\mathbf{p}=\mathbb{E}[X] \in \Delta^{K-1}, \quad \hat{\mathbf{p}}=\frac1N \sum_{n=1}^N X^{(n)} \in \Delta^{K-1}. 
$$

This is the point where coordinate-wise Hoeffding begins to feel mismatched to the problem. 
It is still usable, but it does not fully reflect the fact that the object of interest is a distribution (simplex).

A concentration result for general simplex-valued samples can be obtained from [McDiarmid’s inequality](/assets/pdf/Inequalities.pdf).

<div class="theorem-block">
<div class="theorem-title">Theorem 2 Concentration inequality for general simplex-valued samples </div>
<div class="theorem-content" markdown="1">

For any $$t>0$$,
 
$$
\mathbb{P}\left(
\|\hat{\mathbf{p}}-\mathbf{p}\|_1
\ge
\sqrt{\frac{K}{N}}+t
\right)
\le
\exp\left(-\frac{Nt^2}{2}\right).
$$

Equivalently, for any $$\delta\in(0,1)$$, with probability at least $$1-\delta$$,

$$
\|\hat{\mathbf{p}}-\mathbf{p}\|_1
\le
\sqrt{\frac{K}{N}}
+
\sqrt{\frac{2\log(1/\delta)}{N}}.
$$


</div>
</div>




<details class="proof-block">
<summary class="proof-title">Proof (Click to expand)</summary>
<div class="proof-content" markdown="1">

Define

$$
f\big(X^{(1)},\dots,X^{(N)}\big)
=
\left\|
\frac{1}{N}\sum_{n=1}^N X^{(n)}-\mathbf{p}
\right\|_1.
$$

If we replace one sample $$X^{(j)}$$ by another simplex point $$\tilde X^{(j)}$$, then by the triangle inequality,

$$
\left|
f(X^{(1)},\dots,X^{(j)},\dots,X^{(N)})
-
f(X^{(1)},\dots,\tilde X^{(j)},\dots,X^{(N)})
\right|
\le
\frac{1}{N}\|X^{(j)}-\tilde X^{(j)}\|_1.
$$

Now any two points in the simplex are at $$L_1$$ distance at most 2, so
$$
\left|f(\cdots)-f(\cdots)\right|\le \frac{2}{N}.
$$

Therefore, McDiarmid's inequality gives

$$
\mathbb{P}\left(f-\mathbb{E}f\ge t\right)
\le
\exp\left(
-\frac{2t^2}{\sum_{n=1}^N (2/N)^2}
\right)
=
\exp\left(-\frac{Nt^2}{2}\right),
$$

which proves

$$
\mathbb{P}\left(
\|\hat{\mathbf{p}}-\mathbf{p}\|_1
\ge
\mathbb{E}\|\hat{\mathbf{p}}-\mathbf{p}\|_1+t
\right)
\le
\exp\left(-\frac{Nt^2}{2}\right).
$$

It remains to bound the expectation term. By Cauchy--Schwarz,
$$
\|\mathbf{u}\|_1\le \sqrt{K}\,\|\mathbf{u}\|_2,
$$
hence

$$
\mathbb{E}\|\hat{\mathbf{p}}-\mathbf{p}\|_1
\le
\sqrt{K}\,\mathbb{E}\|\hat{\mathbf{p}}-\mathbf{p}\|_2
\le
\sqrt{K}\,\sqrt{\mathbb{E}\|\hat{\mathbf{p}}-\mathbf{p}\|_2^2}.
$$

Since the samples are independent,
$$
\mathbb{E}\|\hat{\mathbf{p}}-\mathbf{p}\|_2^2
=
\frac{1}{N}\,\mathbb{E}\|X-\mathbf{p}\|_2^2.
$$

Also, because $$\mathbf{p}=\mathbb{E}[X]$$,
$$
\mathbb{E}\|X-\mathbf{p}\|_2^2
=
\mathbb{E}\|X\|_2^2-\|\mathbf{p}\|_2^2.
$$

Finally, since $$X\in\Delta^{K-1}$$,
$$
\|X\|_2^2
=
\sum_{i=1}^K X_i^2
\le
\left(\sum_{i=1}^K X_i\right)^2
=
1.
$$
Therefore,
$$
\mathbb{E}\|X-\mathbf{p}\|_2^2
\le
\mathbb{E}\|X\|_2^2
\le
1.
$$

Combining the above inequalities yields
$$
\mathbb{E}\|\hat{\mathbf{p}}-\mathbf{p}\|_1
\le
\sqrt{K}\sqrt{\frac{1}{N}}
=
\sqrt{\frac{K}{N}}.
$$

Substituting this into the McDiarmid bound gives
$$
\mathbb{P}\left(
\|\hat{\mathbf{p}}-\mathbf{p}\|_1
\ge
\sqrt{\frac{K}{N}}+t
\right)
\le
\exp\left(-\frac{Nt^2}{2}\right).
$$
Finally, setting
$$
t=\sqrt{\frac{2\log(1/\delta)}{N}}
$$
gives the high-probability form
$$
\|\hat{\mathbf{p}}-\mathbf{p}\|_1
\le
\sqrt{\frac{K}{N}}
+
\sqrt{\frac{2\log(1/\delta)}{N}}
$$
with probability at least $$1-\delta$$.

<div class="proof-end">□</div>
</div>
</details>


---
### 3.1 The most extreme simplex case: one-hot samples

A particularly important special case is when each sample lies at a vertex of the simplex:

$$
X \in \{\mathbf{e}_1,...,\mathbf{e}_K\},
$$

where $$\mathbf{e}_i$$ is the $$i$$-th standard basis vector. 
This means that each sample is a one-hot vector, or equivalently, a hard categorical label. In this setting, 
$$\hat{\mathbf{p}}=\frac1N \sum_{n=1}^N X^{(n)}$$ is exactly the empirical class frequency, and $$\mathbf{p}=\mathbb{E}[X]$$ is the true categorical distribution.
It is precisely the classical problem of estimating a categorical distribution from finitely many samples.
And for this problem, **Weissman's inequality** directly controls the $$L_1$$ error of the empirical distribution:


<div class="theorem-block">
<div class="theorem-title">Theorem 1 Weissman's inequality for one-hot samples </div>
<div class="theorem-content" markdown="1">
If $$X^{(1)},\dots,X^{(N)}$$ are i.i.d. one-hot samples from a categorical distribution $$\mathbf{p}$$, then for any $$\varepsilon > 0$$,

$$
\mathbb{P}\left(\|\hat{\mathbf{p}}-\mathbf{p}\|_1 \geq \varepsilon \right) \leq (2^K-2)\exp\left(-\frac{N\varepsilon^2}{2}\right)
$$

</div>
</div>


---
### 3.3 Comparing the sample complexity

A convenient way to compare the bounds is to ask the following question:

>How many samples do we need in order to guarantee $$\|\hat{\mathbf{p}}-\mathbf{p}\|_1 \leq \varepsilon $$ with probability at least $$1-\delta$$?


**Hoeffding's inequality**

The sample complexity is

$$
N = \mathcal{O}\left(\frac{K^2(\log K+\log(1/\delta))}{\varepsilon^2}\right).
$$

So here the sample size scales **quadratically** in $$K$$, up to logarithmic factors.


<details class="proof-block">
<summary class="proof-title">Proof (Click to expand)</summary>
<div class="proof-content" markdown="1">
  
We require

$$
 2K\exp(-\frac{2N\varepsilon^2}{K^2}) \leq \delta
$$

Taking logarithms gives

$$
N \geq \frac{K^2}{2\varepsilon^2}\left(\log(2K)+\log\frac1\delta\right)
$$

Therefore, the resulting sample complexity is

$$
N = \mathcal{O}\left(\frac{K^2(\log K+\log(1/\delta))}{\varepsilon^2}\right).
$$

<div class="proof-end">□</div>
</div>
</details>

**Inequality for general simplex-valued samples**

The sample complexity is

$$
N = \mathcal{O}\left(\frac{K+\log(1/\delta)}{\varepsilon^2}\right).
$$

So the required number of samples again grows **linearly** in $$K$$.

<details class="proof-block">
<summary class="proof-title">Proof (Click to expand)</summary>
<div class="proof-content" markdown="1">

From the general simplex concentration bound, with probability at least $$1-\delta$$,

$$
\|\hat{\mathbf{p}}-\mathbf{p}\|_1
\le
\sqrt{\frac{K}{N}}
+
\sqrt{\frac{2\log(1/\delta)}{N}}.
$$

Therefore, to ensure
$$
\|\hat{\mathbf{p}}-\mathbf{p}\|_1 \le \varepsilon,
$$
it is enough to require
$$
\sqrt{\frac{K}{N}}
+
\sqrt{\frac{2\log(1/\delta)}{N}}
\le
\varepsilon.
$$

Factoring out $$1/\sqrt{N}$$ gives

$$
\frac{\sqrt{K}+\sqrt{2\log(1/\delta)}}{\sqrt{N}}
\le
\varepsilon,
$$

which is equivalent to

$$
N
\ge
\frac{\left(\sqrt{K}+\sqrt{2\log(1/\delta)}\right)^2}{\varepsilon^2}.
$$

Expanding the square,

$$
N
\ge
\frac{K+2\log(1/\delta)+2\sqrt{2K\log(1/\delta)}}{\varepsilon^2}.
$$
Thus,
$$
N = \mathcal{O}\left(\frac{K+\log(1/\delta)+\sqrt{K\log(1/\delta)}}{\varepsilon^2}\right).
$$

In particular, this implies
$$
N = \mathcal{O}\left(\frac{K+\log(1/\delta)}{\varepsilon^2}\right).
$$

<div class="proof-end">□</div>
</div>
</details>

**Weissman's inequality**

The sample complexity is

$$
N=\mathcal{O}\left(\frac{K+\log1/\delta}{\varepsilon^2}\right)
$$

So the required number of samples grows **linearly** in $$K$$.


<details class="proof-block">
<summary class="proof-title">Proof (Click to expand)</summary>
<div class="proof-content" markdown="1">

We require

$$
  (2^K-2)\exp\left(-\frac{N\varepsilon^2}{2}\right) \leq \delta
$$

Taking logarithms gives

$$
N \geq \frac{2}{\varepsilon^2}\left(\log(2^K-2)+\log \frac1\delta\right)
$$

Since $$\log(2^K-2)\approx K\log 2$$, this means that Weissman's inequality yields the sample complexity

$$
N=\mathcal{O}\left(\frac{K+\log\frac1\delta}{\varepsilon^2}\right)
$$

So the required number of samples grows **linearly** in $$K$$.

<div class="proof-end">□</div>
</div>
</details>

This comparison makes the difference transparent:

$$
\text{General simplex: } \mathcal{O}\left(\frac{K}{\varepsilon^2}\right),
$$
$$
\text{Weissman (one-hot): } \mathcal{O}\left(\frac{K}{\varepsilon^2}\right),
$$
$$
\text{Hoeffding: } \mathcal{O}\left(\frac{K^2\log K}{\varepsilon^2}\right).
$$

In other words, the coordinate-wise Hoeffding argument loses a full factor of $$K$$ in dimension dependence because it reduces an $$L_1$$ deviation event to $$K$$ separate coordinate events. By contrast, both Weissman's inequality and the general simplex bound achieve essentially linear dependence on $$K$$. 

#### 3.3.1 Choice of norm matters

It is also worth noting that this gap is largely a consequence of measuring error in $$L_1$$ distance. For $$L_1$$, the coordinate-wise Hoeffding argument must first reduce the event
$$
\|\hat{\mathbf{p}}-\mathbf{p}\|_1 \ge \varepsilon
$$
to the existence of at least one coordinate deviation of size $$\varepsilon/K$$, which is exactly where the extra factor of $$K^2$$ in the exponent comes from.
By contrast, if we measure the error in $$L_\infty$$ distance, then

$$
\|\hat{\mathbf{p}}-\mathbf{p}\|_\infty \ge \varepsilon
\quad\Longleftrightarrow\quad
\exists i \in [K] \text{ such that } |\hat p_i-p_i|\ge \varepsilon.
$$

In this case, a direct union bound together with coordinate-wise Hoeffding already gives

$$
\mathbb{P}\left(\|\hat{\mathbf{p}}-\mathbf{p}\|_\infty \ge \varepsilon\right)
\le
\sum_{i=1}^K
\mathbb{P}\left(|\hat p_i-p_i|\ge \varepsilon\right)
\le
2K\exp(-2N\varepsilon^2).
$$

So for $$L_\infty$$, the coordinate-wise argument is already of the correct order, with only a logarithmic dependence on $$K$$ in the corresponding sample complexity. In other words, the main advantage of Weissman's inequality (or inequality for general simplex-valued samples) is not just that it is **better** in general, but that it is much better matched to the geometry of the $$L_1$$ distance on distributions.


---
### 3.4 One-hot samples as a worst-case upper bound

To compare the general simplex samples with the categorical one-hot case, define a **hardened** sample

$$
Y^{(n)} \in \{\mathbf{e}_1,\dots,\mathbf{e}_K\}
$$

by drawing

$$
\mathbb{P}\!\left(Y^{(n)}=\mathbf{e}_i \mid X^{(n)}\right)=X_i^{(n)},
\qquad i=1,\dots,K.
$$

That is, conditional on the soft vector $$X^{(n)}$$, we sample a one-hot vector according to its coordinates.
By construction,
$$
\mathbb{E}\!\left[Y^{(n)} \mid X^{(n)}\right] = X^{(n)},
$$
and therefore
$$
\mathbb{E}[Y^{(n)}] = \mathbb{E}[X^{(n)}] = \mathbf{p}.
$$
So the hard and soft samples have the same mean, but the hard sample is more extreme since it lives on the vertices of the simplex.
The key observation is that this hardening operation can only increase dispersion under convex losses.

<div class="theorem-block">
<div class="theorem-title">Theorem 1 </div>
<div class="theorem-content" markdown="1">
  
For any convex function $$\varphi:\mathbb{R}^K \to \mathbb{R}$$,

$$
\mathbb{E}\!\left[\varphi\!\left(\hat{\mathbf{p}}_{\mathsf{soft}}-\mathbf{p}\right)\right]
\le
\mathbb{E}\!\left[\varphi\!\left(\hat{\mathbf{p}}_{\mathsf{hard}}-\mathbf{p}\right)\right],
$$

where
$$
\hat{\mathbf{p}}_{\mathsf{hard}} = \frac{1}{N}\sum_{n=1}^N Y^{(n)}.
$$

</div>
</div>

<details class="proof-block">
<summary class="proof-title">Proof (Click to expand)</summary>
<div class="proof-content" markdown="1">

Since
$$
\hat{\mathbf{p}}_{\mathrm{hard}}
=
\frac{1}{N}\sum_{n=1}^N Y^{(n)},
$$
we have

$$
\mathbb{E}\!\left[\hat{\mathbf{p}}_{\mathrm{hard}} \mid X^{(1)},\dots,X^{(N)}\right]
=
\frac{1}{N}\sum_{n=1}^N \mathbb{E}\!\left[Y^{(n)} \mid X^{(n)}\right]
=
\frac{1}{N}\sum_{n=1}^N X^{(n)}
=
\hat{\mathbf{p}}_{\mathrm{soft}}.
$$

Therefore,
$$
\hat{\mathbf{p}}_{\mathrm{soft}}-\mathbf{p}
=
\mathbb{E}\!\left[\hat{\mathbf{p}}_{\mathrm{hard}}-\mathbf{p}\mid X^{(1)},\dots,X^{(N)}\right].
$$
Applying Jensen's inequality conditionally gives

$$
\varphi\!\left(\hat{\mathbf{p}}_{\mathrm{soft}}-\mathbf{p}\right)
=
\varphi\!\left(
\mathbb{E}\!\left[\hat{\mathbf{p}}_{\mathrm{hard}}-\mathbf{p}\mid X^{(1)},\dots,X^{(N)}\right]
\right)
\le
\mathbb{E}\!\left[
\varphi\!\left(\hat{\mathbf{p}}_{\mathrm{hard}}-\mathbf{p}\right)
\middle| X^{(1)},\dots,X^{(N)}
\right].
$$

Taking expectation on both sides proves the claim.

<div class="proof-end">□</div>
</div>
</details>


In particular, taking
$$
\varphi(\mathbf{u})=\|\mathbf{u}\|_1
$$
shows that
$$
\mathbb{E}\!\left[\|\hat{\mathbf{p}}_{\mathrm{soft}}-\mathbf{p}\|_1\right]
\le
\mathbb{E}\!\left[\|\hat{\mathbf{p}}_{\mathrm{hard}}-\mathbf{p}\|_1\right].
$$
Therefore, although Weissman's inequality is stated for one-hot categorical samples, it should be viewed as a conservative worst-case reference for general simplex-valued samples.

---
## 4. Reference

1. Weissman et al. [Inequalities for the L1 deviation of the empirical distribution](https://shiftleft.com/mirrors/www.hpl.hp.com/techreports/2003/HPL-2003-97R1.pdf). _Hewlett-Packard Labs, Tech 2003._