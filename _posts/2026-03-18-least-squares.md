---
layout: post
title: "Least Squares: Closed Form, QR, SVD, Gradient Descent, and Ridge Regression"
date: 2026-03-18
description: A comprehensive walkthrough of the least squares problem from five angles — closed-form solution, QR decomposition, SVD, gradient descent, and regularization via ridge regression.
tags: [optimization]
categories: [blog]
math: true
toc:
  beginning: true
---

## 0. Least Squares Problem

Suppose we want to solve a linear system $$A\mathbf{x}=\mathbf{b}$$ with $$A \in \mathbb{R}^{m \times n}$$, $$ \mathbf{b} \in \mathbb{R}^m$$ and $$m \geq n$$. 
Since the system is typically overdetermined, it may not admit an exact solution. Instead, we seek a vector $$\hat{\mathbf{x}}$$ such that $$A\hat{\mathbf{x}} \approx \mathbf{b}$$.
This leads to the **least squares** (LS) problem:

\begin{equation}\label{eq:ls}
\hat{\mathbf{x}}=\arg\min_{\mathbf{x}\in\mathbb{R}^n}\lVert A\mathbf{x}-\mathbf{b}\rVert_2^2.
\end{equation}
 
Least squares arises ubiquitously in regression problems. 
In this blog, we review and compare five common approaches for solving LS problems, including the closed-form solution, QR factorization, SVD, gradient descent, and ridge regression. 
Our focus is on their numerical stability and behavior under rank deficiency.

---

## 1. Normal Equations and the Closed-Form Solution
Solving \eqref{eq:ls} leads to the **normal equations**:

\begin{equation}\label{eq:normal-equations}
{A}^\top {A}\, \hat{\mathbf{x}} = {A}^\top \mathbf{b}.
\end{equation}

When $${A}$$ has full column rank $$n$$, $${A}^\top {A}$$ is symmetric positive definite and invertible, and the solution can be written in closed form as

$$
\boxed{\hat{\mathbf{x}} = ({A}^\top {A})^{-1} {A}^\top \mathbf{b}}.
$$

The matrix $${A}^+ := ({A}^\top {A})^{-1} {A}^\top$$ is the **Moore–Penrose pseudoinverse** of $${A}$$.

<details class="proof-block">
<summary class="proof-title">Proof of \eqref{eq:normal-equations} (Click to expand)</summary>
<div class="proof-content" markdown="1">

Expanding the objective:

$$
\|{A}\mathbf{x} - \mathbf{b}\|_2^2 = \mathbf{x}^\top {A}^\top {A}\, \mathbf{x} - 2\mathbf{b}^\top {A}\mathbf{x} + \mathbf{b}^\top \mathbf{b}.
$$

Taking the gradient w.r.t $$\mathbf{x}$$ and setting it to zero:

$$
\nabla_{\mathbf{x}}\|{A}\mathbf{x} - \mathbf{b}\|_2^2 = 2{A}^\top {A}\, \mathbf{x} - 2{A}^\top \mathbf{b} = \mathbf{0},
$$

which gives the **normal equations**:

$$
{A}^\top {A}\, \hat{\mathbf{x}} = {A}^\top \mathbf{b}.
$$

<div class="proof-end">□</div>
</div>
</details>


**<span style="color:red;">Limitation.</span>** This formula is valuable analytically, but computing the solution through the normal equations is often numerically undesirable.
Even when $$A$$ has full column rank, the matrix may still be ill-conditioned, and forming $$A^\top A$$ further amplifies this issue. 
To make this precise, we briefly recall the notion of the condition number.

### 1.1 Condition Number

The main numerical issue of the normal equations is related to the **condition number**. For an invertible matrix $$A$$, the spectral condition number w.r.t $$\ell_2$$ norm is defined as

$$
\kappa_2(A):= \|A\|_2 \|A^{-1}\|_2
$$

For a symmetric positive definite matrix $$A$$, this becomes

\begin{equation}\label{eq:kappa}
\kappa_2(A)=\frac{\lambda_\mathsf{max}(A)}{\lambda_\mathsf{min}(A)},
\end{equation}

where $$\lambda_\mathsf{max}(A)$$ and $$\lambda_\mathsf{min}(A)$$ denote the largest and smallest eigen values of $$A$$.

<details class="proof-block">
<summary class="proof-title">Definition of Matrix Norm (Click to expand))</summary>
<div class="proof-content" markdown="1">
  Given any norm $$\|\cdot\|$$ on the space $$\mathbb{R}^n$$ of $$n$$-dimensional vectors with real entries, 
  the suborinate matrix norm on the $$\mathbb{R}^{n\times n}$$ matrices with real entries is defined by

  $$\|A\|=\mathrm{max}_{\mathbf{v}\in \mathbb{R}^n\backslash \{\mathbf{0}\}}\frac{\|A\mathbf{v}\|}{\|\mathbf{v}\|}$$
  
</div>
</details>

<details class="proof-block">
<summary class="proof-title">Proof of \eqref{eq:kappa} (Click to expand)</summary>
<div class="proof-content" markdown="1">

Since $$A$$ is symmetric positive definite, all its eigenvalues are real and positive, and
there exists an orthonormal basis of eigenvectors $$\mathbf{w}_1,\dots,\mathbf{w}_n$$ such that

$$
A\mathbf{w}_i=\lambda_i \mathbf{w}_i,\quad i=1,2,\dots,n.
$$

Without loss of generality, assume that
$$
0<\lambda_1\le \lambda_2\le \cdots \le \lambda_n.
$$
Now let $$\mathbf{u} \in \mathbb{R}^n$$ be arbitrary and write it as

$$
\mathbf{u}=c_1\mathbf{w}_1+\cdots+c_n\mathbf{w}_n.
$$

Then

$$
A\mathbf{u}=c_1\lambda_1 \mathbf{w}_1+\cdots+c_n\lambda_n \mathbf{w}_n.
$$

Using the orthonormality of $$\mathbf{w}_1,\dots,\mathbf{w}_n$$, we have

$$
\|Au\|_2^2
=
(c_1\lambda_1 w_1+\cdots+c_n\lambda_n w_n)^\top
(c_1\lambda_1 w_1+\cdots+c_n\lambda_n w_n)
=
c_1^2\lambda_1^2+\cdots+c_n^2\lambda_n^2.
$$

Since $$\lambda_i\le \lambda_n$$ for all $$i$$, it follows that
$$
\|A\mathbf{u}\|_2^2
\le
(c_1^2+\cdots+c_n^2)\lambda_n^2.
$$
Again by orthonormality,
$$
\|\mathbf{u}\|_2^2=c_1^2+\cdots+c_n^2.
$$
Hence
$$
\|A\mathbf{u}\|_2^2 \le \lambda_n^2 \|\mathbf{u}\|_2^2,
$$
and therefore
$$
\|A\mathbf{u}\|_2 \le \lambda_n \|\mathbf{u}\|_2.
$$
By the definition of the matrix $$\ell_2$$-norm, this implies

$$
\|A\|_2 = \lambda_n = \lambda_\mathsf{max}(A).
$$

Similarly, we have 

$$
\|A^{-1}\|_2 = \frac{1}{\lambda_1} = \frac{1}{\lambda_\mathsf{min}(A)}.
$$

By the definition of the condition number, 

$$
\kappa_2(A)=\frac{\lambda_\mathsf{max}(A)}{\lambda_\mathsf{min}(A)}.
$$

<div class="proof-end">□</div>
</div>
</details>

The condition number measures the sensitivity of the solution of a linear system to pertubations in the data. 
Consider

$$
M \mathbf{x} = \mathbf{c}
$$

If $$\mathbf{c}$$ is perturbed to $$\mathbf{c}+\delta \mathbf{c}$$ (for example, rounding errors during the calculation.), 
then the solution changes to $$\mathbf{x}+\delta \mathbf{x}$$, where 

$$
M \delta \mathbf{x} = \delta \mathbf{c}.
$$

This leads to the relative error bound

$$
\frac{\|\delta \mathbf{x}\|_2}{\|\mathbf{x}\|_2^2} \leq \kappa_2(M) \frac{\|\delta \mathbf{c}\|_2}{\|\mathbf{c}\|_2^2} 
$$

<details class="proof-block">
<summary class="proof-title">Proof (Click to expand)</summary>
<div class="proof-content" markdown="1">
Evidently, $$\mathbf{c}=M\mathbf{x}$$ and $$\delta \mathbf{x} = M^{-1}\delta \mathbf{c}$$. Further

$$\|\mathbf{c}\|\leq \|M\|\|\mathbf{x}\|$$ and $$\|\delta \mathbf{x}\|\leq \|M^{-1}\|\|\delta \mathbf{c}\|$$.

The result follows immediately by multiplying these inequalities.

<div class="proof-end">□</div>
</div>
</details>

Importantly, this is an upper bound: a large condition number does not mean that every perturbation causes a large error,
but it does mean that the system is potentially unstable in the worst case. For the normal equations, we solve

$$
{A}^\top {A}\, \hat{\mathbf{x}} = {A}^\top \mathbf{b}.
$$

Since 

$$
\kappa_2(A^\top A) = \kappa_2(A)^2,
$$

forming $$A^\top A$$ squares the condition number and may significantly worsen numerical stability. 
This is why, despite its simple closed form, the normal-equation approach is often avoided in practice.

For example, if
$$
A=
\begin{pmatrix}
\varepsilon & 0\\
0 & 1
\end{pmatrix}
$$
where $$\varepsilon\in(0,1)$$, then $$\kappa_2(A)=\varepsilon^{-1}>1$$, while
$$
\kappa_2(A^\top A)=\varepsilon^{-2}
=\varepsilon^{-1}\kappa_2(A)\gg \kappa_2(A)
$$
when $$0<\varepsilon\ll 1$$.  

---

## 2. QR Decomposition


There are various alternative techniques which avoid the direct construction of the normal matrix $$A^\top A$$, 
and so do not lead to this extreme ill-conditioning. Here, we shall describe one algorithm based on QR decomposition,
which begins by factorizing the matrix $$A$$ into an orthogonal matrix $$Q$$ and an upper triangular matrix $R$$ 


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

<object data="/assets/img/qr-demo.pdf" type="application/pdf" width="100%" height="600px">
  <p>PDF cannot be displayed. <a href="/assets/img/qr-demo.pdf">Download qr-demo.pdf</a>.</p>
</object>

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
