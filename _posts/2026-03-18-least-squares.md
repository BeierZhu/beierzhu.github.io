---
layout: post
title: "Least Squares: Closed Form, QR, SVD, Gradient Descent, and Ridge Regression"
date: 2026-03-18
description: A comprehensive walkthrough of the least squares problem from five angles — closed-form solution, QR decomposition, SVD, gradient descent, and regularization via ridge regression.
tags: [math, linear-algebra, optimization]
categories: [blog]
math: true
toc:
  beginning: true
---

## Problem Setup

Given a data matrix $$\mathbf{A} \in \mathbb{R}^{m \times n}$$ (with $$m \geq n$$) and a target vector $$\mathbf{b} \in \mathbb{R}^m$$, the **least squares problem** seeks

$$
\hat{\mathbf{x}} = \arg\min_{\mathbf{x} \in \mathbb{R}^n} \|\mathbf{A}\mathbf{x} - \mathbf{b}\|_2^2.
$$

This arises ubiquitously in regression, signal processing, and scientific computing. Below we examine five perspectives on solving it.

---

## 1. Closed-Form Solution (Normal Equations)

Expanding the objective:

$$
\|\mathbf{A}\mathbf{x} - \mathbf{b}\|_2^2 = \mathbf{x}^\top \mathbf{A}^\top \mathbf{A}\, \mathbf{x} - 2\mathbf{b}^\top \mathbf{A}\mathbf{x} + \mathbf{b}^\top \mathbf{b}.
$$

Taking the gradient and setting it to zero:

$$
\nabla_{\mathbf{x}} = 2\mathbf{A}^\top \mathbf{A}\, \mathbf{x} - 2\mathbf{A}^\top \mathbf{b} = \mathbf{0},
$$

which gives the **normal equations**:

$$
\mathbf{A}^\top \mathbf{A}\, \hat{\mathbf{x}} = \mathbf{A}^\top \mathbf{b}.
$$

When $$\mathbf{A}$$ has full column rank, $$\mathbf{A}^\top \mathbf{A}$$ is symmetric positive definite and invertible:

$$
\boxed{\hat{\mathbf{x}} = (\mathbf{A}^\top \mathbf{A})^{-1} \mathbf{A}^\top \mathbf{b}.}
$$

The matrix $$\mathbf{A}^+ := (\mathbf{A}^\top \mathbf{A})^{-1} \mathbf{A}^\top$$ is the **Moore–Penrose pseudoinverse** of $$\mathbf{A}$$.

**Limitation.** Explicitly forming and inverting $$\mathbf{A}^\top \mathbf{A}$$ squares the condition number and can be numerically unstable. Better to use QR or SVD.

---

## 2. QR Decomposition

Factor $$\mathbf{A} = \mathbf{Q}\mathbf{R}$$ where $$\mathbf{Q} \in \mathbb{R}^{m \times n}$$ has orthonormal columns ($$\mathbf{Q}^\top \mathbf{Q} = \mathbf{I}_n$$) and $$\mathbf{R} \in \mathbb{R}^{n \times n}$$ is upper triangular with positive diagonal (the thin/reduced QR).

Substituting into the normal equations:

$$
(\mathbf{Q}\mathbf{R})^\top (\mathbf{Q}\mathbf{R})\,\hat{\mathbf{x}} = (\mathbf{Q}\mathbf{R})^\top \mathbf{b}
\implies \mathbf{R}^\top \\underbrace{\mathbf{Q}^\top \mathbf{Q}}_{\mathbf{I}} \mathbf{R}\,\hat{\mathbf{x}} = \mathbf{R}^\top \mathbf{Q}^\top \mathbf{b}.
$$

Since $$\mathbf{R}$$ is invertible we can left-multiply by $$(\mathbf{R}^\top)^{-1}$$:

$$
\boxed{\mathbf{R}\,\hat{\mathbf{x}} = \mathbf{Q}^\top \mathbf{b}.}
$$

This is solved cheaply by **back-substitution** in $$O(n^2)$$ once we have $$\mathbf{Q}^\top \mathbf{b}$$. The QR approach is numerically stable and is the standard algorithm in most software (LAPACK's `dgels`).

---

## 3. Singular Value Decomposition (SVD)

The full SVD of $$\mathbf{A}$$ is

$$
\mathbf{A} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top,
$$

where $$\mathbf{U} \in \mathbb{R}^{m \times m}$$, $$\mathbf{V} \in \mathbb{R}^{n \times n}$$ are orthogonal, and $$\boldsymbol{\Sigma} = \mathrm{diag}(\sigma_1, \ldots, \sigma_n)$$ with $$\sigma_1 \geq \cdots \geq \sigma_n \geq 0$$.

The least-squares solution is given directly by the **pseudoinverse**:

$$
\hat{\mathbf{x}} = \mathbf{A}^+ \mathbf{b} = \mathbf{V}\boldsymbol{\Sigma}^+ \mathbf{U}^\top \mathbf{b},
$$

where $$\boldsymbol{\Sigma}^+ = \mathrm{diag}(\sigma_1^{-1}, \ldots, \sigma_r^{-1}, 0, \ldots)$$ and $$r = \mathrm{rank}(\mathbf{A})$$.

Writing $$\mathbf{u}_i, \mathbf{v}_i$$ for the $$i$$-th columns of $$\mathbf{U}, \mathbf{V}$$:

$$
\hat{\mathbf{x}} = \sum_{i=1}^{r} \frac{\mathbf{u}_i^\top \mathbf{b}}{\sigma_i} \mathbf{v}_i.
$$

**Advantages.**
- Handles rank-deficient $$\mathbf{A}$$ gracefully (yields the minimum-norm solution).
- Reveals the numerical rank via the singular values.
- Directly quantifies sensitivity: small $$\sigma_i$$ amplify noise in $$\mathbf{b}$$.

**Condition number** $$\kappa(\mathbf{A}) = \sigma_1 / \sigma_r$$ governs the sensitivity of $$\hat{\mathbf{x}}$$ to perturbations.

---

## 4. Gradient Descent

Rather than solving analytically, we minimize $$f(\mathbf{x}) = \|\mathbf{A}\mathbf{x} - \mathbf{b}\|_2^2$$ iteratively. The gradient is

$$
\nabla f(\mathbf{x}) = 2\mathbf{A}^\top(\mathbf{A}\mathbf{x} - \mathbf{b}).
$$

**Gradient descent update** with step size $$\eta > 0$$:

$$
\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} - \eta \cdot 2\mathbf{A}^\top(\mathbf{A}\mathbf{x}^{(k)} - \mathbf{b}).
$$

### Convergence

The objective is $$\mu$$-strongly convex and $$L$$-smooth with

$$
\mu = 2\sigma_{\min}^2(\mathbf{A}), \qquad L = 2\sigma_{\max}^2(\mathbf{A}).
$$

With the optimal fixed step size $$\eta^* = 1/L$$, gradient descent converges **linearly**:

$$
f(\mathbf{x}^{(k)}) - f(\hat{\mathbf{x}}) \leq \left(1 - \frac{\mu}{L}\right)^k \bigl(f(\mathbf{x}^{(0)}) - f(\hat{\mathbf{x}})\bigr).
$$

The convergence rate depends on the **condition number** $$\kappa = L/\mu = \kappa(\mathbf{A})^2$$. Poorly conditioned problems converge very slowly — motivating preconditioning or momentum methods (e.g., conjugate gradient, Nesterov).

---

## 5. Ridge Regression (Tikhonov Regularization)

When $$\mathbf{A}$$ is ill-conditioned or $$m < n$$, we add an $$\ell_2$$ penalty:

$$
\hat{\mathbf{x}}_\lambda = \arg\min_{\mathbf{x}} \|\mathbf{A}\mathbf{x} - \mathbf{b}\|_2^2 + \lambda\|\mathbf{x}\|_2^2, \quad \lambda > 0.
$$

The (now uniquely defined) solution is

$$
\boxed{\hat{\mathbf{x}}_\lambda = (\mathbf{A}^\top \mathbf{A} + \lambda \mathbf{I})^{-1} \mathbf{A}^\top \mathbf{b}.}
$$

### SVD Perspective

Using the SVD $$\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$$:

$$
\hat{\mathbf{x}}_\lambda = \sum_{i=1}^{n} \frac{\sigma_i}{\sigma_i^2 + \lambda} (\mathbf{u}_i^\top \mathbf{b})\, \mathbf{v}_i.
$$

Compare with the unregularized SVD solution: each coefficient $$1/\sigma_i$$ is replaced by $$\sigma_i/(\sigma_i^2 + \lambda)$$. For large $$\sigma_i$$ the change is negligible; for small $$\sigma_i \ll \sqrt{\lambda}$$, the coefficient is shrunk toward zero. Ridge regression thus **damps the contribution of low-energy directions**, reducing variance at the cost of introducing bias.

### Bias–Variance Tradeoff

Assuming $$\mathbf{b} = \mathbf{A}\mathbf{x}^* + \boldsymbol{\epsilon}$$ with $$\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$$:

$$
\mathbb{E}[\hat{\mathbf{x}}_\lambda] - \mathbf{x}^* = -\lambda(\mathbf{A}^\top\mathbf{A} + \lambda\mathbf{I})^{-1}\mathbf{x}^* \quad \text{(bias)},
$$

$$
\mathrm{Var}(\hat{\mathbf{x}}_\lambda) = \sigma^2 (\mathbf{A}^\top\mathbf{A} + \lambda\mathbf{I})^{-1}\mathbf{A}^\top\mathbf{A}(\mathbf{A}^\top\mathbf{A} + \lambda\mathbf{I})^{-1}.
$$

As $$\lambda \to 0$$ the bias vanishes and variance grows; as $$\lambda \to \infty$$ both bias and variance change in opposite directions. The optimal $$\lambda$$ balances the two — typically chosen by cross-validation.

---

## Summary

| Method | Complexity | Handles rank deficiency | Key property |
|---|---|---|---|
| Normal equations | $$O(mn^2 + n^3)$$ | No | Simple; unstable if ill-conditioned |
| QR decomposition | $$O(mn^2)$$ | No | Numerically stable; standard in practice |
| SVD | $$O(mn^2)$$ | **Yes** | Most general; reveals structure |
| Gradient descent | $$O(mn)$$ per iter | No | Scalable; slow if $$\kappa$$ large |
| Ridge regression | $$O(mn^2 + n^3)$$ | **Yes** | Regularization; bias–variance tradeoff |
