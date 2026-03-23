---
layout: post
title: "Least Squares: Closed Form, QR, SVD, Gradient Descent, and Ridge Regression"
date: 2026-03-18
description: "This blog gives a unified overview of five common methods for solving least-squares problems: the normal equations, QR decomposition, SVD, gradient descent, and ridge regression. We focus on the main issues that distinguish them in practice, including numerical stability, rank deficiency, minimum-norm solutions, and robustness to noise."
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
which begins by factorizing the matrix $$A$$ into an orthogonal matrix $$Q$$ and an upper triangular matrix $$R$$ (see figure below). 

<figure style="text-align: center;">
  <img src="/assets/img/qr-demo.svg" alt="QR Decomposition Demo" style="width: 30%; max-width: 100%;" />
  <figcaption>Figure: QR Decomposition of matrix A into orthogonal Q and upper triangular R.</figcaption>
</figure>

The following theorem shows that such a decomposition exists.

<div class="theorem-block">
<div class="theorem-title">Theorem 2.1 (QR factorisation) </div>
<div class="theorem-content" markdown="1">
Suppose that $$A \in \mathbb{R}^{m \times n}$$ where $$m \geq n$$. Then, $$A$$ can be written in the form

$$A = {Q}{R},$$

where $${R}$$ is an upper triangular $$n \times n$$ matrix, and $$Q$$ is an $$m \times n$$ matrix which satisfies

$${Q}^\top {Q} = I_n,$$

where $$I_n$$ is the $$n \times n$$ identity matrix. If $$\mathrm{rank}(A) = n$$, then $$R$$ is nonsingular.
</div>
</div>

<details class="proof-block">
<summary class="proof-title">Proof of Theorem 2.1 (Click to expand)</summary>
<div class="proof-content" markdown="1">
 We use induction on $$n$$, the number of columns in $$A$$. The theorem clearly holds when $$n=1$$ so that $$A$$ has only one column. 
Indeed, writing $$\mathbf{c}$$ for this column vector and assuming that $$\mathbf{c}\neq \mathbf{0}$$, the matrix $${Q}$$ has just one column, 
  the vector $$\mathbf{c}/\|\mathbf{c}\|_2$$, and $${R}$$ has a single element, $$\|\mathbf{c}\|_2$$. 

 Suppose that the theorem is true when $$n=k$$, where $$1\le k < m$$. Consider a matrix $$A$$ which has $$m$$ rows and $$k+1$$ columns, partitioned as

 $$
 A=(A_k\ \ \mathbf{a}),
 $$

 where $$\mathbf{a} \in\mathbb{R}^m$$ is a column vector and $$A_k$$ has $$k$$ columns. To obtain the desired factorisation $${Q}{R}$$ of $$A$$ we seek $${Q}=({Q}_k\ \ \mathbf{q})$$ and

 $$
 {R}=
 \begin{pmatrix}
 {R}_k & \mathbf{r} \\
 0 & {\alpha}
 \end{pmatrix}
 $$

 such that

 $$
 A=(A_k\ \ \mathbf{a})
 =
 ({Q}_k\ \ \mathbf{q})
 \begin{pmatrix}
 {R}_k & \mathbf{r} \\
 0 & {\alpha}
 \end{pmatrix}.
 $$

 Multiplying this out and requiring that $${Q}^{\top}{Q}=I_{k+1}$$, the identity matrix of order $$k+1$$, we conclude that

 $$
 \begin{aligned}
 A_k &= {Q}_k{R}_k, \\
 \mathbf{a} &= {Q}_k \mathbf{r} + \mathbf{q} {\alpha}, \\
 {Q}_k^{\top}{Q}_k &= I_k, \\
 \mathbf{q}^{\top}{Q}_k &= \mathbf{0}^{\top}, \\
 \mathbf{q}^{\top}\mathbf{q} &= \mathbf{1}.
 \end{aligned}
 $$

 These equations show that $${Q}_k{R}_k$$ is the factorisation of $$A_k$$, which exists by the inductive hypothesis, and then lead to

 $$
 \begin{aligned}
 \mathbf{r} &= {Q}_k^{\top}\mathbf{a}, \\
 \mathbf{q} &= (1/{\alpha})\bigl(\mathbf{a}-{Q}_k{Q}_k^{\top}\mathbf{a}\bigr),
 \end{aligned}
 $$

 where $${\alpha}=\|\mathbf{a}-{Q}_k{Q}_k^{\top}\mathbf{a}\|_2$$). The number $${\alpha}$$ is the constant required to ensure that the vector $$\mathbf{q}$$ is normalised.

 With these definitions of $$\mathbf{q}$$, $$\mathbf{r}$$, $$\alpha$$, $${Q}_k$$ and $${R}_k$$ we have constructed the required factors of $$A$$, showing that the theorem is true when $$n=k+1$$. Since it holds when $$n=1$$ the induction is complete.

<div class="proof-end">□</div>
</div>
</details>

With QR factorisation, solving \eqref{eq:ls} is equivalent to solve:

$$
\boxed{R\mathbf{x}=Q^\top \mathbf{b}}
$$

<details class="proof-block">
<summary class="proof-title">Proof (Click to expand)</summary>
<div class="proof-content" markdown="1">
 Replace $$A$$ with $$QR$$, we have

$$
\min_\mathbf{x}\|A\mathbf{x}-\mathbf{b}\|_2^2=\min_\mathbf{x}\|QR\mathbf{x}-\mathbf{b}\|_2^2.
$$

Decompose $\mathbf{b}$ as 

$$
\mathbf{b}=QQ^\top\mathbf{b} + (I-QQ^\top)\mathbf{b}.
$$

This leads to 

$$
\|QR\mathbf{x}-\mathbf{b}\|_2^2=\|Q(R\mathbf{x}-Q^\top\mathbf{b}) - (I-QQ^\top)\mathbf{b}\|_2^2.
$$

Let

$$
\mathbf{u}:=Q(R\mathbf{x}-Q^\top\mathbf{b}), \quad \mathbf{v}=(I-QQ^\top)\mathbf{b}.
$$

It is evident that $$\mathbf{u} \perp \mathbf{v}$$, as

$$
\mathbf{u}^\top \mathbf{v} = (R\mathbf{x}-Q^\top\mathbf{b})^\top Q^\top (I-QQ^\top)\mathbf{b}=(R\mathbf{x}-Q^\top\mathbf{b})^\top  (Q^\top- Q^\top)\mathbf{b} = \mathbf{0}
$$

Recall that $$\|\mathbf{u}-\mathbf{v}\|_2^2=\|\mathbf{u}\|_2^2+\|\mathbf{v}\|_2^2$$ if $$\mathbf{u} \perp \mathbf{v}$$.
Therefore

$$
\|QR\mathbf{x}-\mathbf{b}\|_2^2=\|R\mathbf{x}-Q^\top\mathbf{b}\|_2^2 + \| (I-QQ^\top)\mathbf{b}\|_2^2.
$$

Note that the second term on RHS is not related to $$\mathbf{x}$$. Hence, $$\mathbf{x}$$ defined as the solution of $$R\mathbf{x}=Q^\top\mathbf{b}$$, is the required least squares solution.
<div class="proof-end">□</div>
</div>
</details>

Since left multiplication by an orthogonal matrix does not change singular values,
$$R$$ has the same singular values as $$A$$. Hence, 

$$
\kappa_2(R)=\kappa_2(A)
$$

Therefore, QR factorization avoids the **squaring of the condition number** (recall that condition number for the normal equations is $$\kappa_2(A)^2$$) and is numerically more stable than the normal-equation approach.

---

## 3. Singular Value Decomposition (SVD)

When using QR factorization to solve a least-squares problem, we typically assume that 
$$A$$ has full column rank. In practice, however, $$A$$ may be rank-deficient. 
In that case, more appropriate choices include SVD or gradient descent with appropriate initialization.
We begin with the SVD of $$A$$:

$$
A=U\Sigma V^\top,
$$

where $$U\in\mathbb{R}^{m\times m}$$ and $$V\in\mathbb{R}^{n\times n}$$ are orthogonal matrices, and $$\Sigma\in\mathbb{R}^{m\times n}$$ is diagonal except possibly for trailing zero rows or columns.
Suppose that $$\operatorname{rank}(A)=r$$. Then we may write the SVD in block form as

$$
A=
[U_1\ U_2]
\begin{bmatrix}
\Sigma_1 & 0\\
0 & 0
\end{bmatrix}
\begin{bmatrix}
V_1^\top\\
V_2^\top
\end{bmatrix},
$$

where:

- $$U_1\in\mathbb{R}^{m\times r}$$ and $$V_1\in\mathbb{R}^{n\times r}$$,
- $$U_2\in\mathbb{R}^{m\times (m-r)}$$ and $$V_2\in\mathbb{R}^{n\times (n-r)}$$,
- $$\Sigma_1\in\mathbb{R}^{r\times r}$$ is diagonal with positive singular values.

This decomposition reveals the geometry of the least-squares problem:

- the columns of $$V_2$$ span the null space $$\mathrm{Null}(A)$$,
- the columns of $$V_1$$ correspond to the effective directions of $$A$$,
- the columns of $$U_1$$ span the column space $$\mathrm{Col}(A)$$.


Using the SVD of $$A$$, we obtain
$$
\|A\mathbf{x}-\mathbf{b}\|_2^2
=
\|U\Sigma V^\top \mathbf{x}-\mathbf{b}\|_2^2.
$$
Since $$U$$ is orthogonal and preserves the Euclidean norm,

$$
\|A\mathbf{x}-\mathbf{b}\|_2^2
=
\|\Sigma V^\top \mathbf{x}-U^\top \mathbf{b}\|_2^2.
$$

Now write

$$
V^\top \mathbf{x}
=
\begin{bmatrix}
V_1^\top\\
V_2^\top
\end{bmatrix}\mathbf{x}
=
\begin{bmatrix}
\mathbf{y}\\
\mathbf{z}
\end{bmatrix},
\qquad
U^\top \mathbf{b}
=
\begin{bmatrix}
U_1^\top\\
U_2^\top
\end{bmatrix}\mathbf{b}
=
\begin{bmatrix}
\mathbf{c}\\
\mathbf{d}
\end{bmatrix},
$$

where $$\mathbf{y},\mathbf{c}\in\mathbb{R}^r$$ and $$\mathbf{z}\in\mathbb{R}^{n-r}$$, $$\mathbf{d}\in\mathbb{R}^{m-r}$$.
Hence,

$$
\|A\mathbf{x}-\mathbf{b}\|_2^2
=
\left\|
\begin{bmatrix}
\Sigma_1\mathbf{y}\\
0
\end{bmatrix}
-
\begin{bmatrix}
\mathbf{c}\\
\mathbf{d}
\end{bmatrix}
\right\|_2^2
=
\|\Sigma_1\mathbf{y}-\mathbf{c}\|_2^2+\|\mathbf{d}\|_2^2.
$$

The second term is independent of $$\mathbf{x}$$, and the first term depends only on $$\mathbf{y}$$, not on $$\mathbf{z}$$. 
Since $$\Sigma_1$$ is invertible, the minimizing choice of $$\mathbf{y}$$ is
$$
\mathbf{y}=\Sigma_1^{-1}\mathbf{c}=\Sigma_1^{-1}U_1^\top \mathbf{b}.
$$
Thus every least-squares solution can be written as

$$
\mathbf{x}
=
V_1\Sigma_1^{-1}U_1^\top \mathbf{b}+V_2\mathbf{z},
\qquad
\mathbf{z}\in\mathbb{R}^{n-r}.
$$

Equivalently, if we define
$$
\mathbf{x}^\dagger:=V_1\Sigma_1^{-1}U_1^\top \mathbf{b},
$$
then all least-squares solutions are of the form

$$
\mathbf{x}=\mathbf{x}^\dagger+\mathbf{z},
\qquad
\mathbf{z}\in\mathrm{Null}(A).
$$


The minimum-norm least-squares solution is

$$
\boxed{
\mathbf{x}^\dagger
=
V_1\Sigma_1^{-1}U_1^\top \mathbf{b}.
}
$$



Therefore, SVD is often the preferred tool when $$A$$ is singular or nearly singular.

---

## 4. Gradient Descent and Implicit Bias

In the rank-deficient setting, SVD provides a natural and explicit way to characterize all least-squares solutions and to identify the minimum-norm one. Interestingly, gradient descent offers another effective approach: although it does not explicitly impose a minimum-norm criterion, **with zero initialization it implicitly converges to the same solution**.

Consider the least-squares objective

$$
f(\mathbf{x})=\frac{1}{2}\|A\mathbf{x}-\mathbf{b}\|_2^2.
$$

Gradient descent updates take the form

$$
\mathbf{x}_{t+1}
=
\mathbf{x}_t-\eta \nabla f(\mathbf{x}_t)
=
\mathbf{x}_t-\eta A^\top(A\mathbf{x}_t-\mathbf{b}),
$$

where $$\eta>0$$ is the step size.

When the least-squares solution is not unique, gradient descent exhibits an important implicit bias:  if initialized at
$$
\mathbf{x}_0=0,
$$
then, under a suitable choice of step size, it converges to the same minimum-norm least-squares solution derived in the previous section, namely

$$
\mathbf{x}^\dagger=A^+\mathbf{b}.
$$

The key reason is that every gradient update lies in the range of $$A^\top$$. Indeed,

$$
\mathbf{x}_{t+1}-\mathbf{x}_t
=
-\eta A^\top(A\mathbf{x}_t-\mathbf{b})
\in \mathrm{range}(A^\top).
$$

Since $$\mathbf{x}_0=0\in \mathrm{range}(A^\top)$$, it follows by induction that
$$
\mathbf{x}_t\in \mathrm{range}(A^\top)
\qquad \text{for all } t\ge 0.
$$

On the other hand, from the SVD analysis in the previous section, all least-squares solutions can be written as

$$
\mathbf{x}
=
\mathbf{x}^\dagger+\mathbf{z},
\qquad
\mathbf{z}\in \mathrm{Null}(A),
$$

where $$\mathbf{x}^\dagger=A^+\mathbf{b}$$ is the minimum-norm least-squares solution.

Now recall the fundamental orthogonal decomposition
$$
\mathbb{R}^n
=
\mathrm{range}(A^\top)\oplus \mathrm{Null}(A).
$$
Therefore, among all least-squares solutions, the unique one that lies entirely in $$\mathrm{range}(A^\top)$$ is precisely $$\mathbf{x}^\dagger$$. Since gradient descent initialized at zero never leaves $$\mathrm{range}(A^\top)$$, if it converges to a least-squares solution, that limit must be
$$
\mathbf{x}^\dagger=A^+\mathbf{b}.
$$

This shows that zero initialization induces an implicit regularization effect: **although the objective itself does not explicitly prefer one least-squares solution over another, gradient descent selects the minimum-norm one through its optimization trajectory**.

### 4.1 Why Initialization Matters

The conclusion above depends crucially on the initialization.  If gradient descent starts from a nonzero initial point $$\mathbf{x}_0$$, then its component in $$\mathrm{Null}(A)$$ is preserved throughout the iterations. Indeed, if

$$
\mathbf{x}_0=\mathbf{x}_0^{\parallel}+\mathbf{x}_0^{\perp},
\qquad
\mathbf{x}_0^{\parallel}\in \mathrm{range}(A^\top),
\quad
\mathbf{x}_0^{\perp}\in \mathrm{Null}(A),
$$

then

$$
A\mathbf{x}_0^{\perp}=0,
\qquad
A^\top A\mathbf{x}_0^{\perp}=0.
$$

Hence the null-space component is never changed by gradient descent. As a result, the limit point is generally not the minimum-norm solution, but rather the minimum-norm solution plus the initial null-space component.


---


## 5. Ridge Regression

Besides SVD and gradient descent, another common approach for handling ill-conditioned or rank-deficient least-squares problems is **ridge regression**. The basic idea is to replace the original least-squares problem with the regularized problem

$$
\hat{\mathbf{x}}_\lambda
=
\arg\min_{\mathbf{x}}
\left(
\|A\mathbf{x}-\mathbf{b}\|_2^2
+
\lambda \|\mathbf{x}\|_2^2
\right),
$$

where $$\lambda>0$$ is the regularization parameter. The additional term $$\lambda\|\mathbf{x}\|_2^2$$ discourages solutions with large norm. 
The solution can be written in closed form as

$$
\boxed{
\hat{\mathbf{x}}_\lambda
=
(A^\top A+\lambda I)^{-1}A^\top \mathbf{b}.
}
$$

<details class="proof-block">
<summary class="proof-title">Proof (Click to expand)</summary>
<div class="proof-content" markdown="1">
  
 Taking the gradient of the ridge objective and setting it to zero, we obtain
 
 $$
 2A^\top(A\mathbf{x}-\mathbf{b})+2\lambda \mathbf{x}=0,
 $$

 which gives
 
 $$
 (A^\top A+\lambda I)\mathbf{x}=A^\top \mathbf{b}.
 $$
 
 Since $$\lambda>0$$, the matrix $$A^\top A+\lambda I$$ is symmetric positive definite and hence invertible. Therefore, the ridge solution admits the closed-form expression
 
 $$
 \boxed{
 \hat{\mathbf{x}}_\lambda
 =
 (A^\top A+\lambda I)^{-1}A^\top \mathbf{b}.
 }
 $$
 
This formula shows why ridge regression is useful in the rank-deficient setting: even if $$A^\top A$$ is singular, adding $$\lambda I$$ shifts the eigenvalues away from zero and makes the system invertible.

<div class="proof-end">□</div>
</div>
</details>


### 5.1 Connection to the Minimum-Norm Solution

It is also useful to compare ridge regression with the minimum-norm solution from a constrained-optimization perspective.
Recall that the minimum-norm solution can be defined as

$$
\min_{\mathbf{x}\in\mathbb{R}^n}\|\mathbf{x}\|_2^2
\qquad
\text{subject to}
\qquad
A\mathbf{x}=\mathbf{b}.
$$

Using a Lagrange multiplier $$\boldsymbol{\lambda}$$, we define

$$
\mathcal{L}(\mathbf{x},\boldsymbol{\lambda})
=
\|\mathbf{x}\|_2^2+\boldsymbol{\lambda}^\top(A\mathbf{x}-\mathbf{b}).
$$

Thus, the minimum-norm solution is obtained by minimizing the norm under a hard equality constraint. By contrast, ridge regression does not enforce the constraint $$A\mathbf{x}=\mathbf{b}$$ exactly. Instead, it solves

$$
\mathcal{L}_\lambda(\mathbf{x})
=
\|A\mathbf{x}-\mathbf{b}\|_2^2+\lambda\|\mathbf{x}\|_2^2
,
$$

which replaces the hard constraint by a soft penalty. In this sense, ridge regression introduces an explicit trade-off between data fitting and solution norm, whereas the minimum-norm formulation minimizes the norm among exact solutions.


Rank deficiency alone is not the main reason to use ridge regression, since the minimum-norm solution already gives a natural canonical solution in that case. The real advantage of ridge appears in noisy or nearly singular settings. In exact or nearly noiseless settings, the minimum-norm solution is often the most faithful canonical solution to the original least-squares problem. In contrast, when $$A$$ or $$\mathbf{b}$$ is noisy, ridge regression is usually more robust, since it explicitly suppresses unstable small-singular-value directions by introducing a norm penalty. Thus, the minimum-norm solution is more natural from an analytical viewpoint, whereas ridge regression is often preferable from a practical viewpoint.


---

## Summary

| Method | Key property |
|---|---|
| Normal equations | Closed form, but may be numerically unstable |
| QR decomposition | Stable solver for the original LS problem |
| SVD| Explicitly selects the minimum-norm solution |
| Gradient descent | Implicitly selects the minimum-norm solution |
| Ridge regression  | Explicit regularization for robustness; but not faithful to original objective|
