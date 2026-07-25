---
layout: note
title: "Gradient Descent, Smoothness, and the Descent Lemma"
source_title: "MIT6.7960-01.3-Gradient Descent, Smoothness, and the Descent Lemma"
course: "MIT 6.7960"
sequence: "01.3"
source_context: "The Gradient, Steepest Descent, and the Guarantees of Gradient Descent"
permalink: "/notes/mit6-7960-01-3-gradient-descent-smoothness-and-the-descent-lemma/"
tags:
  - "math/optimization"
  - "topic/gradient-descent"
  - "topic/smoothness"
---
The negative gradient is an infinitesimal descent direction, but an algorithm must choose a finite displacement. This note derives the gradient-descent update, states exactly what differentiability already guarantees, introduces L-smoothness as quantitative control of gradient drift, and proves the Descent Lemma by integration. The material remains in one note because each stage resolves the limitation of the previous one: direction alone does not choose a distance, local descent does not provide a uniform threshold, and smoothness becomes useful only after it is converted into a finite-step upper model.

## 1.6 The gradient-descent update

§1.5 identified a unit direction; an algorithm needs an actual displacement. Gradient descent uses the entire gradient and a scale parameter $\eta>0$.

> [!definition] Gradient descent
> Starting from $x_0\in\mathbb R^n$, the constant-step gradient-descent iteration is
>
> $$
> \boxed{\ x_{k+1}=x_k-\eta\nabla f(x_k)\ }.
> $$
>
> In practice the learning rate may vary with $k$; the chapter uses a constant $\eta$ unless stated otherwise.

Write

$$
g_k:=\nabla f(x_k),
\qquad
s_k:=x_{k+1}-x_k=-\eta g_k.
$$

The objects inherited from §1.5 are related but not identical:

| Object | Formula | Question answered |
|---|---|---|
| Unit steepest-descent direction | $-g_k/\|g_k\|_2$ | Which ray? |
| Gradient-descent step | $s_k=-\eta g_k$ | Which displacement? |
| Distance travelled | $\|s_k\|_2=\eta\|g_k\|_2$ | How far? |
| Learning rate | $\eta$ | By what factor is the gradient scaled? |

Thus $\eta$ is not the distance travelled. The distance is proportional to the current gradient norm and becomes small whenever that norm is small.

### What differentiability already guarantees

Fix $x$ and write $g=\nabla f(x)\neq0$. Substituting the displacement $-\eta g$ into the differentiability expansion of §1.2 gives

$$
f(x-\eta g)
=f(x)-\eta\|g\|_2^2+r(\eta),
\qquad
r(\eta)=o(\eta)
\quad(\eta\downarrow0).
$$

The negative first-order term is enough to prove genuine local descent, not merely predict it.

> [!abstract] Proof — a sufficiently small step decreases $f$
> Since $r(\eta)/\eta\to0$ and $\|g\|_2^2>0$, there exists a point-dependent threshold $\bar\eta(x)>0$ such that
>
> $$
> |r(\eta)|\le\frac12\eta\|g\|_2^2
> \qquad\text{whenever}\qquad
> 0<\eta<\bar\eta(x).
> $$
>
> Consequently,
>
> $$
> f(x-\eta g)
> \le f(x)-\frac12\eta\|g\|_2^2
> <f(x).
> $$
>
> Therefore every nonstationary point admits some positive gradient step that decreases the objective. $\blacksquare$

> [!important] A descent direction does not make every finite step descend
> Differentiability proves that a sufficiently small step works at a fixed point, but it does not reveal the value of $\bar\eta(x)$ or provide one step size that is safe at every iterate. A large step can leave the region in which the first-order model is accurate and increase the objective.

> [!example]- A one-dimensional quadratic shows overshoot
> Let
>
> $$
> f(x)=\frac{\lambda}{2}x^2,
> \qquad
> \lambda>0.
> $$
>
> Since $f'(x)=\lambda x$, gradient descent gives
>
> $$
> x_{k+1}=(1-\eta\lambda)x_k,
> \qquad
> f(x_{k+1})=(1-\eta\lambda)^2f(x_k).
> $$
>
> For $x_k\neq0$, strict descent occurs exactly when
>
> $$
> |1-\eta\lambda|<1
> \quad\Longleftrightarrow\quad
> \boxed{\ 0<\eta<\frac{2}{\lambda}\ }.
> $$
>
> | Learning rate | Behavior |
> |---|---|
> | $0<\eta<1/\lambda$ | Approaches the minimizer from the same side |
> | $\eta=1/\lambda$ | Reaches the minimizer in one step |
> | $1/\lambda<\eta<2/\lambda$ | Overshoots but contracts |
> | $\eta=2/\lambda$ | Oscillates with unchanged objective value |
> | $\eta>2/\lambda$ | Oscillates and diverges |
>
> The admissible step size is controlled by curvature: larger $\lambda$ requires smaller $\eta$.

### Why the full gradient step appears

The unit vector $-g/\|g\|_2$ selects a ray but not a distance. The linear model alone cannot choose one:

$$
f(x)+\langle g,s\rangle
\longrightarrow -\infty
\qquad\text{along}\qquad
s=-tg,\quad t\to\infty.
$$

The failure is expected—the linear approximation is only local. A quadratic movement penalty makes step selection well posed:

$$
\boxed{
m_\eta(s)
=f(x)+\langle g,s\rangle
+\frac{1}{2\eta}\|s\|_2^2
}.
$$

> [!warning] The quadratic term is a chosen penalty, not the Hessian of $f$
> The second-order Taylor model contains $\tfrac12s^\top\nabla^2f(x)s$. Here the term $\tfrac1{2\eta}s^\top Is$ is introduced deliberately to penalize large moves; it does not assert that $\nabla^2f(x)=I/\eta$.

> [!abstract] Derivation — minimize linear gain plus movement cost
> The two terms depending on $s$ can be completed to a square:
>
> $$
> \begin{aligned}
> \langle g,s\rangle+\frac{1}{2\eta}\|s\|_2^2
> &=\frac{1}{2\eta}
> \left(\|s+\eta g\|_2^2-\eta^2\|g\|_2^2\right)\\
> &=\frac{1}{2\eta}\|s+\eta g\|_2^2
> -\frac{\eta}{2}\|g\|_2^2.
> \end{aligned}
> $$
>
> The last term is constant in $s$, and the squared term is minimized uniquely when
>
> $$
> s^\star=-\eta g.
> $$
>
> Equivalently,
>
> $$
> \nabla_sm_\eta(s)=g+\frac1\eta s=0,
> \qquad
> \nabla_s^2m_\eta(s)=\frac1\eta I\succ0.
> $$
>
> Hence $x_{k+1}=x_k+s^\star=x_k-\eta\nabla f(x_k)$ is the unique minimizer of this regularized first-order model.

> [!tip] Read $\eta$ as inverse penalty strength
> The coefficient on $\|s\|_2^2$ is $1/(2\eta)$. A smaller $\eta$ imposes a stronger movement penalty and produces a shorter step; a larger $\eta$ weakens the penalty and permits a longer step.

> [!note]- Hard radius versus soft quadratic penalty
> A hard constraint and a soft penalty answer related but different step-selection problems:
>
> | Formulation | Euclidean solution |
> |---|---|
> | $\min\langle g,s\rangle$ subject to $\|s\|_2\le\Delta$ | $-\Delta g/\|g\|_2$ |
> | $\min\langle g,s\rangle+\tfrac1{2\eta}\|s\|_2^2$ | $-\eta g$ |
>
> The trust-region radius is the explicit constraint $\Delta$. The penalty parameter $\eta$ is related to that viewpoint, but it is not literally a radius.

> [!warning] Stationarity stops the algorithm, not the analysis
> If $\nabla f(x_k)=0$, then $x_{k+1}=x_k$. A stationary point may be a minimum, maximum, saddle point, or a point whose behavior appears only at higher order; the update alone does not distinguish them.

Differentiability has now given a **local, qualitative** fact: some sufficiently small step decreases $f$ at each nonstationary point. What it has not given is an explicit, iterate-independent choice of $\eta$. The next assumption—$L$-smoothness—controls how rapidly the gradient can change and turns “sufficiently small” into a quantitative bound.

---

## 1.7 $L$-smoothness — a Lipschitz condition on the gradient

§1.6 proved that differentiability gives a point-dependent statement: some sufficiently small gradient step decreases $f$, but the threshold is unknown. To make the threshold quantitative, we need a uniform bound on how rapidly the gradient can change.

> [!definition] Lipschitz continuity
> Let $g:D\to\mathbb R^m$, where $D\subseteq\mathbb R^n$. The map $g$ is **$L$-Lipschitz on $D$** if some $L\ge0$ satisfies
>
> $$
> \boxed{
> \|g(a)-g(b)\|_2
> \le L\|a-b\|_2
> \qquad\text{for every }a,b\in D
> }.
> $$
>
> The output cannot change faster than $L$ times the input. In one dimension this bounds every secant slope:
>
> $$
> \frac{|g(a)-g(b)|}{|a-b|}\le L
> \qquad(a\neq b).
> $$

> [!definition] $L$-smoothness
> A differentiable function $f:D\to\mathbb R$ is **$L$-smooth on $D$** when its gradient is $L$-Lipschitz:
>
> $$
> \boxed{
> \|\nabla f(x)-\nabla f(y)\|_2
> \le L\|x-y\|_2
> \qquad\text{for every }x,y\in D
> }.
> $$
>
> Thus $L$ is a speed limit on the change of the gradient vector—both its magnitude and its direction.

The two commonly confused Lipschitz conditions control different objects:

| Property | Inequality | What it controls |
|---|---|---|
| $f$ is $M$-Lipschitz | $\lvert f(x)-f(y)\rvert\le M\|x-y\|_2$ | Change in function value; for differentiable $f$, this corresponds to a bound on $\|\nabla f\|_2$ |
| $f$ is $L$-smooth | $\|\nabla f(x)-\nabla f(y)\|_2\le L\|x-y\|_2$ | Change in gradient; in the $C^2$ case, this corresponds to a curvature bound |

> [!important] Global and regional smoothness are different claims
> The quantifier “for every $x,y\in D$” matters. A function may fail to be $L$-smooth on all of $\mathbb R^n$ while being smooth on every bounded region. Optimization arguments must name a region that contains the iterates, or assume a global bound.

> [!note]- Global smoothness is an analysis model, not an automatic fact
> Whether a useful global $L$ exists depends strongly on the objective and parameterization.
>
> | Objective or model | Typical smoothness status |
> |---|---|
> | Least squares $f(w)=\|Xw-y\|_2^2/(2n)$ | Globally smooth with $L=\|X\|_2^2/n$ |
> | Linear logistic regression | Globally smooth with $L\le\|X\|_2^2/(4n)$ under the same averaging convention |
> | $f(x)=x^4$ | Not globally smooth; on $[-R,R]$, $L=12R^2$ is valid |
> | ReLU network | Not classically smooth at activation boundaries; only piecewise differentiable/smooth |
> | Deep network with smooth activations | Often smooth on bounded parameter regions, but a useful global Hessian bound may still fail because products of weights can make curvature unbounded |
>
> Consequently, practical analysis often uses a regional constant or a step-dependent $L_k$ rather than one global value. Backtracking line search tests a local quadratic upper bound and reduces the step until it holds, avoiding the need to know the smallest global $L$ in advance.
>
> Two algorithmic devices should not be mistaken for assumptions on the objective. **Gradient clipping** bounds the update used by the algorithm but does not make the original gradient Lipschitz. Adding $\lambda\|w\|_2^2/2$ contributes $\lambda I$ to the Hessian; if the original objective is $L$-smooth the sum is $(L+\lambda)$-smooth, but this term alone does not cap an otherwise unbounded Hessian.
>
> For stochastic gradients, even a smooth full objective does not imply that every noisy SGD step decreases the true loss. Stochastic analyses usually replace monotone per-step descent with statements in expectation and add assumptions on gradient noise.

The constant is not unique: if $L$ works, every larger constant works. The smallest valid value is

$$
L_{\min}
=\sup_{x\neq y}
\frac{\|\nabla f(x)-\nabla f(y)\|_2}{\|x-y\|_2},
$$

when the supremum is finite. Using an upper bound larger than $L_{\min}$ is valid but produces more conservative step-size guarantees.

> [!example]- Three boundary-setting examples
> **Quadratic.** For $f(x)=\tfrac\lambda2x^2$ with $\lambda>0$,
>
> $$
> |f'(x)-f'(y)|
> =\lambda|x-y|,
> $$
>
> so the smallest global smoothness constant is $L=\lambda$.
>
> **Smooth only on bounded regions.** For $f(x)=x^4$, $f''(x)=12x^2$ is unbounded, so no finite global $L$ exists. On $[-R,R]$, however, $|f''(x)|\le12R^2$, so $L=12R^2$ is valid.
>
> **Lipschitz but not smooth.** The function $f(x)=|x|$ is $1$-Lipschitz, but it is not differentiable at $0$ and therefore is not $L$-smooth under this definition.

> [!note] The degenerate case $L=0$
> If $L=0$, then $\nabla f(x)=\nabla f(y)$ for every $x,y\in D$. On a convex domain the gradient is a constant vector $c$, so
>
> $$
> f(x)=\langle c,x\rangle+d.
> $$
>
> The function may have nonzero slope, but it has no curvature. Every later expression containing $1/L$ or $2/L$ therefore assumes $L>0$.

![small-and-large-smoothness-constants]({{ '/assets/notes/gradient-descent/small-and-large-smoothness-constants.jpg' | relative_url }})
*Geometric preview: §1.8 will prove that the tangent-plane error is at most $\tfrac L2\|y-x\|_2^2$. For the same displacement and error tolerance, a smaller $L$ permits a wider trustworthy neighborhood; the figure illustrates this bound rather than defining $L$.*

### The Hessian view (twice-differentiable $f$)

The Hessian is the Jacobian of the gradient:

$$
H(x):=\nabla^2f(x),
\qquad
[H(x)]_{ij}
=\frac{\partial^2f}{\partial x_i\partial x_j}(x).
$$

When $f\in C^2$, mixed partials commute and $H(x)$ is symmetric. Its spectral norm is the largest factor by which it can amplify a direction:

$$
\|H(x)\|_2
:=\max_{\|u\|_2=1}\|H(x)u\|_2.
$$

> [!important] Theorem — smoothness as a Hessian bound
> Let $D\subseteq\mathbb R^n$ be open and convex, and let $f\in C^2(D)$. Then
>
> $$
> \boxed{
> f\text{ is }L\text{-smooth on }D
> \quad\Longleftrightarrow\quad
> \|\nabla^2f(x)\|_2\le L
> \text{ for every }x\in D
> }.
> $$

> [!abstract]- Proof of the Hessian characterization
> For $x,y\in D$, convexity keeps the segment $x+t(y-x)$ inside $D$. The fundamental theorem of calculus applied to the gradient gives
>
> $$
> \nabla f(y)-\nabla f(x)
> =\int_0^1\nabla^2f(x+t(y-x))(y-x)\,dt.
> $$
>
> If $\|\nabla^2f(z)\|_2\le L$ throughout $D$, then
>
> $$
> \begin{aligned}
> \|\nabla f(y)-\nabla f(x)\|_2
> &\le\int_0^1
> \|\nabla^2f(x+t(y-x))\|_2\|y-x\|_2\,dt\\
> &\le L\|y-x\|_2.
> \end{aligned}
> $$
>
> Conversely, suppose $\nabla f$ is $L$-Lipschitz. For every unit $u$,
>
> $$
> \|\nabla^2f(x)u\|_2
> =\lim_{t\to0}
> \frac{\|\nabla f(x+tu)-\nabla f(x)\|_2}{|t|}
> \le L.
> $$
>
> Taking the maximum over unit $u$ gives $\|\nabla^2f(x)\|_2\le L$. $\blacksquare$

For a unit direction $u$, the one-dimensional slice

$$
\varphi_u(t):=f(x+tu)
$$

satisfies

$$
\boxed{
\varphi_u''(0)
=u^\top\nabla^2f(x)u
}.
$$

Thus the Hessian quadratic form is the second directional derivative used in optimization: positive values bend the slice upward, negative values bend it downward, and zero means no second-order bending along that direction.

![hessian-directional-curvature]({{ '/assets/notes/gradient-descent/hessian-directional-curvature.png' | relative_url }})
*For $f(x_1,x_2)=\tfrac12(4x_1^2+x_2^2)$ at the common base point $x_0=(0,0)$, the $e_1$ slice has second derivative $4$ and the $e_2$ slice has second derivative $1$. The common base point makes the directional comparison exact.*

> [!note]- Hessian geometry — eigenvalues and directional curvature
> Because the symmetric Hessian has an orthonormal eigenbasis, write
>
> $$
> Hv_k=\lambda_kv_k,
> \qquad
> u=\sum_kc_kv_k,
> \qquad
> \sum_kc_k^2=1.
> $$
>
> Then every unit-direction curvature is
>
> $$
> u^\top Hu
> =\sum_k\lambda_kc_k^2.
> $$
>
> This is a weighted average of the eigenvalues, so
>
> $$
> \lambda_{\min}
> \le u^\top Hu
> \le\lambda_{\max}.
> $$
>
> The extreme values occur at the corresponding eigenvectors. In optimization language, eigenvectors are the principal axes of the local quadratic model and eigenvalues are the second directional derivatives along those axes. They should not be confused with the differential-geometric principal curvatures of the graph away from a critical point.

> [!note]- Quadratic level-set geometry
> In Hessian eigen-coordinates $\delta=\sum_ka_kv_k$,
>
> $$
> \frac12\delta^\top H\delta
> =\frac12\sum_k\lambda_ka_k^2.
> $$
>
> The eigenvalue signs determine the local quadratic geometry:
>
> | Hessian spectrum | Quadratic geometry |
> |---|---|
> | All eigenvalues positive | Bowl; positive level sets are ellipsoids |
> | Some eigenvalues zero, none negative | Flat directions; positive level sets extend as cylinders |
> | Positive and negative eigenvalues | Saddle; level sets are hyperbolic |
> | All eigenvalues negative | Dome; negative level sets are ellipsoids |
>
> In the positive-definite case and for $c>0$,
>
> $$
> \frac12\sum_k\lambda_ka_k^2=c
> \quad\Longrightarrow\quad
> r_k=\sqrt{\frac{2c}{\lambda_k}}.
> $$
>
> Hence a large eigenvalue gives a short, sharp axis and a small positive eigenvalue gives a long, gentle axis.
>
> ![hessian-eigenvalues-and-elliptic-level-sets]({{ '/assets/notes/gradient-descent/hessian-eigenvalues-and-elliptic-level-sets.jpg' | relative_url }})
> *For a positive-definite quadratic model at its critical point, eigenvectors are the ellipse axes and the semi-axis lengths scale as $1/\sqrt{\lambda_k}$.*

> [!note]- Second-order test at a critical point
> The following classification applies only when $\nabla f(x)=0$:
>
> | Hessian at $x$ | Conclusion |
> |---|---|
> | $\nabla^2f(x)\succ0$ | Strict local minimum |
> | $\nabla^2f(x)\prec0$ | Strict local maximum |
> | $\nabla^2f(x)$ indefinite | Saddle point |
> | $\nabla^2f(x)$ semidefinite with a zero eigenvalue | Inconclusive |
>
> Semidefinite cases require higher-order information: $x^4$, $-x^4$, and $x^3$ all have second derivative zero at the origin but yield a minimum, maximum, and neither, respectively.
>
> ![hessian-sign-classification]({{ '/assets/notes/gradient-descent/hessian-sign-classification.png' | relative_url }})
> *Eigenvalue signs classify nondegenerate critical points; a zero eigenvalue makes the second-order test inconclusive.*

### Smoothness controls magnitude, not sign

For a symmetric Hessian,

$$
\|\nabla^2f(x)\|_2
=\max_i|\lambda_i(\nabla^2f(x))|.
$$

Hence the Hessian form of $L$-smoothness is

$$
\boxed{
-LI\preceq\nabla^2f(x)\preceq LI
},
$$

which allows every eigenvalue to lie anywhere in $[-L,L]$. It permits negative curvature and therefore does not imply convexity.

| Property | Hessian condition in the $C^2$ case | What it controls |
|---|---|---|
| $L$-smooth | $-LI\preceq H(x)\preceq LI$ | Magnitude of upward and downward curvature |
| Convex | $H(x)\succeq0$ | Sign of curvature |
| Convex and $L$-smooth | $0\preceq H(x)\preceq LI$ | Nonnegative curvature bounded above by $L$ |

For example, $f(x)=-\tfrac12x^2$ is concave but $1$-smooth: its second derivative is $-1$, whose magnitude is bounded by $1$.

> [!summary] What $L$ buys
> Differentiability gave an unquantified local remainder. $L$-smoothness replaces it with a uniform rate limit on the gradient. In §1.8 this will yield
>
> $$
> \left|
> f(y)-f(x)-\langle\nabla f(x),y-x\rangle
> \right|
> \le\frac L2\|y-x\|_2^2,
> $$
>
> turning “choose $\eta$ sufficiently small” into an explicit step-size calculation.

---

## 1.8 The Descent Lemma

Differentiability gave a pointwise little-$o$ remainder. $L$-smoothness makes the linearization error uniformly quadratic and computable.

> [!important] Theorem — Descent Lemma and two-sided linearization error
> Let $D\subseteq\mathbb R^n$ be open and convex, and let $f:D\to\mathbb R$ be differentiable and $L$-smooth on $D$. Then for every $x,y\in D$,
>
> $$
> \boxed{
> \left|
> f(y)-f(x)-\langle\nabla f(x),y-x\rangle
> \right|
> \le\frac L2\|y-x\|_2^2
> }.
> $$
>
> In particular, the one-sided upper bound called the **Descent Lemma** is
>
> $$
> \boxed{
> f(y)
> \le f(x)+\langle\nabla f(x),y-x\rangle
> +\frac L2\|y-x\|_2^2
> }.
> $$

The convexity required here is a property of the **domain** $D$, not of the function $f$: it ensures that the segment from $x$ to $y$ remains inside the region where the smoothness assumption holds. More generally, the proof only needs that segment and the $L$-smoothness bound along it.

> [!note]- A local constant is enough for one step
> For an update from $x_k$ to $x_{k+1}$, the same proof works with a step-dependent $L_k$ whenever
>
> $$
> \|\nabla f(z)-\nabla f(x_k)\|_2
> \le L_k\|z-x_k\|_2
> $$
>
> along the segment joining the two iterates. The resulting upper model uses $L_k/2$. Backtracking line search operationalizes this idea by shrinking a trial step until a suitable local upper-bound inequality is satisfied.

### Proof by integrating gradient drift

Set

$$
d:=y-x,
\qquad
R_x(d):=f(x+d)-f(x)-\langle\nabla f(x),d\rangle.
$$

The quantity $R_x(d)$ is the error made by the tangent model at $x$.

> [!abstract] Proof
> **1. Restrict $f$ to the segment.** Define
>
> $$
> \varphi(t):=f(x+td),
> \qquad 0\le t\le1.
> $$
>
> The chain rule and the fundamental theorem of calculus give the exact identity
>
> $$
> f(y)-f(x)
> =\int_0^1
> \langle\nabla f(x+td),d\rangle\,dt.
> $$
>
> **2. Subtract the tangent prediction.** Since $\langle\nabla f(x),d\rangle$ is constant in $t$,
>
> $$
> R_x(d)
> =\int_0^1
> \langle\nabla f(x+td)-\nabla f(x),d\rangle\,dt.
> $$
>
> Thus the tangent error is the accumulated drift of the gradient along the segment.
>
> **3. Bound the drift.** Cauchy–Schwarz and $L$-smoothness imply
>
> $$
> \begin{aligned}
> \big|
> \langle\nabla f(x+td)-\nabla f(x),d\rangle
> \big|
> &\le
> \|\nabla f(x+td)-\nabla f(x)\|_2\|d\|_2\\
> &\le Lt\|d\|_2^2.
> \end{aligned}
> $$
>
> **4. Integrate.** Therefore
>
> $$
> |R_x(d)|
> \le\int_0^1Lt\|d\|_2^2\,dt
> =\frac L2\|d\|_2^2,
> $$
>
> because $\int_0^1t\,dt=1/2$. Substituting $d=y-x$ proves the two-sided bound and hence the Descent Lemma. $\blacksquare$

> [!tip] Where the factor $1/2$ comes from
> At position $x+td$, the point is $t\|d\|_2$ away from $x$, so the gradient-drift bound grows linearly as $Lt\|d\|_2$. Its average over $0\le t\le1$ is $L/2$:
>
> $$
> \int_0^1Lt\,dt=\frac L2.
> $$

### Geometric upper model

For fixed $x$, define

$$
Q_x(y)
:=f(x)+\langle\nabla f(x),y-x\rangle
+\frac L2\|y-x\|_2^2.
$$

The lemma says $f(y)\le Q_x(y)$ wherever the assumptions hold. Moreover,

$$
Q_x(x)=f(x),
\qquad
\nabla_yQ_x(x)=\nabla f(x),
$$

so the quadratic upper model and $f$ have the same value and gradient at the base point.

![descent-lemma-upper-model]({{ '/assets/notes/gradient-descent/descent-lemma-upper-model.jpg' | relative_url }})
*The blue function lies below its orange $L$-quadratic upper model, and the two are tangent at the base point. The picture is one-dimensional, but the bound holds in $\mathbb R^n$ and does not require $f$ to be convex.*

> [!warning] Three quadratic expressions with different meanings
>
> | Expression | Source | Status |
> |---|---|---|
> | $f(x)+\langle g,s\rangle+\tfrac12s^\top\nabla^2f(x)s$ | Second-order Taylor expansion | Local approximation using the actual Hessian |
> | $f(x)+\langle g,s\rangle+\tfrac1{2\eta}\|s\|_2^2$ | §1.6 step-selection model | Chosen movement penalty |
> | $f(x)+\langle g,s\rangle+\tfrac L2\|s\|_2^2$ | Descent Lemma | Guaranteed upper bound from $L$-smoothness |

The Descent Lemma does **not** assume convexity. Smoothness limits curvature magnitude; negative curvature only pushes $f$ farther below the upper model. Applying the same argument to $-f$ gives the matching lower model, which is why the absolute-error form above holds.

> [!example]- Tightness of the constant $L/2$
> For
>
> $$
> f(z)=\frac L2\|z\|_2^2,
> \qquad
> \nabla f(z)=Lz,
> $$
>
> the gradient is exactly $L$-Lipschitz and
>
> $$
> f(y)
> =f(x)+\langle\nabla f(x),y-x\rangle
> +\frac L2\|y-x\|_2^2
> $$
>
> for every $x,y$. Hence no smaller uniform quadratic coefficient can work for every $L$-smooth function.

> [!important] Spend the bound on a gradient step
> Let $g=\nabla f(x)$ and set $y=x-\eta g$. The upper bound immediately yields
>
> $$
> \boxed{
> f(x-\eta g)
> \le
> f(x)-\eta\left(1-\frac{L\eta}{2}\right)\|g\|_2^2
> }.
> $$
>
> The vague remainder from §1.6 has become the explicit term $\tfrac{L\eta^2}{2}\|g\|_2^2$. Section §1.9 interprets this inequality as a step-size window and a per-step decrease guarantee.

---

## Connections

- The local geometry that selects the negative-gradient direction is proved in [Differentiability, Directional Derivatives, and Steepest Descent]({{ '/notes/mit6-7960-01-2-differentiability-directional-derivatives-and-steepest-descent/' | relative_url }}).
- Substituting the finite update into the Descent Lemma and accumulating the decrease gives [Stability, Stationarity, and Global Progress]({{ '/notes/mit6-7960-01-4-stability-stationarity-and-global-progress/' | relative_url }}).
- The deterministic gradient step analyzed here becomes a noisy batch update in [Stochastic Gradient Steps and Trainable Loss Geometry]({{ '/notes/mit6-7960-02-2-stochastic-gradient-steps-and-trainable-loss-geometry/' | relative_url }}).
