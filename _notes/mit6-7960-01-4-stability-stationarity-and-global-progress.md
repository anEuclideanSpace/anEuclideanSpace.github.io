---
layout: note
title: "Stability, Stationarity, and Global Progress"
source_title: "MIT6.7960-01.4-Stability, Stationarity, and Global Progress"
course: "MIT 6.7960"
sequence: "01.4"
source_context: "The Gradient, Steepest Descent, and the Guarantees of Gradient Descent"
permalink: "/notes/mit6-7960-01-4-stability-stationarity-and-global-progress/"
tags:
  - "math/optimization"
  - "topic/stability"
  - "topic/convergence"
---
The Descent Lemma becomes an optimization theorem only after a concrete step is substituted into it and the resulting decreases are accumulated. This note derives the per-step stability window, turns repeated descent into a finite stationarity budget, and then adds the Polyak-Łojasiewicz condition to show when stationarity certifies global objective progress. The sequence is logically cumulative: the stability coefficient funds the telescoping argument, the telescoping argument yields small gradients, and the additional structural condition converts those gradients into a geometric function-gap rate.

## 1.9 The per-step guarantee and the stability window

The Descent Lemma becomes operational only after choosing a move. For the gradient step, write $g:=\nabla f(x)$ and substitute $y=x-\eta g$:

$$
\begin{aligned}
f(x-\eta g)
&\le f(x)+\langle g,-\eta g\rangle+\frac L2\|-\eta g\|_2^2\\
&=f(x)-\eta\|g\|_2^2+\frac{L\eta^2}{2}\|g\|_2^2.
\end{aligned}
$$

> [!important] Per-step guarantee
> For an $L$-smooth objective and the full-gradient update $x_{k+1}=x_k-\eta\nabla f(x_k)$,
>
> $$
> \boxed{
> f(x_{k+1})
> \le
> f(x_k)-\underbrace{\eta\left(1-\frac{L\eta}{2}\right)}_{c(\eta)}
> \|\nabla f(x_k)\|_2^2
> }.
> $$
>
> Unlike the $o(\eta)$ statement in §1.6, this bound gives an explicit coefficient valid uniformly wherever the same smoothness constant $L$ applies.

### Read the sign of the coefficient

The inequality itself holds for every $\eta$; what changes is the sign of

$$
c(\eta)=\eta\left(1-\frac{L\eta}{2}\right).
$$

| Step size | Sign of $c(\eta)$ | What the theorem certifies when $g\neq0$ |
|---|---:|---|
| $0<\eta<2/L$ | $c(\eta)>0$ | Strict decrease |
| $\eta=2/L$ | $c(\eta)=0$ | Non-increase only |
| $\eta>2/L$ | $c(\eta)<0$ | No descent guarantee |

$$
\boxed{\text{strict descent is guaranteed for }0<\eta<\frac2L.}
$$

> [!warning] Loss of a guarantee is not guaranteed failure
> When $\eta>2/L$, the upper bound stops proving descent; it does **not** say that every such step must increase every function. The quadratic below shows why no larger uniform safe window is possible.

> [!example]- Tight quadratic: the entire window in one recurrence
> Let
>
> $$
> f(x)=\frac L2x^2,
> \qquad
> \nabla f(x)=Lx.
> $$
>
> Gradient descent is exactly
>
> $$
> x_{k+1}=(1-\eta L)x_k.
> $$
>
> | Step size | Multiplier $1-\eta L$ | Behavior |
> |---|---:|---|
> | $0<\eta<1/L$ | $(0,1)$ | Contracts without crossing the minimum |
> | $\eta=1/L$ | $0$ | Reaches the minimum in one step |
> | $1/L<\eta<2/L$ | $(-1,0)$ | Crosses the minimum but contracts |
> | $\eta=2/L$ | $-1$ | Oscillates with unchanged loss |
> | $\eta>2/L$ | $\lvert 1-\eta L\rvert>1$ | Oscillates and diverges |
>
> Thus the boundary $2/L$ is tight for the class of all $L$-smooth functions.

### Why $1/L$ appears so often

The guaranteed drop is $c(\eta)\|g\|_2^2$. Maximizing its concave coefficient gives

$$
c'(\eta)=1-L\eta=0,
\qquad
c''(\eta)=-L<0,
$$

so

$$
\eta^\star=\frac1L,
\qquad
c\!\left(\frac1L\right)=\frac1{2L}.
$$

> [!summary] Exact meaning of “optimal”
> The choice $\eta=1/L$ maximizes the **worst-case single-step decrease certified by $L$-smoothness alone**:
>
> $$
> \boxed{
> f(x_{k+1})\le f(x_k)-\frac1{2L}\|\nabla f(x_k)\|_2^2
> }.
> $$
>
> It is not a universal best learning rate for a particular objective or training run.

A slightly more conservative form is often easier to reuse. For every $0<\eta\le1/L$,

$$
1-\frac{L\eta}{2}\ge\frac12
\quad\Longrightarrow\quad
\boxed{
f(x_{k+1})\le f(x_k)-\frac\eta2\|\nabla f(x_k)\|_2^2
}.
$$

> [!note]- Preview: additional structure can change the best constant step
> The number $1/L$ uses only an upper curvature bound. If a later analysis also knows a positive lower curvature bound $\mu$, then on a symmetric positive-definite quadratic the constant step that minimizes the worst spectral contraction is $2/(L+\mu)$. This is a different optimization criterion under stronger assumptions; §1.11–§1.13 introduce the missing ideas.

![gradient-descent-stability-window]({{ '/assets/notes/gradient-descent/gradient-descent-stability-window.png' | relative_url }})
*The coefficient $c(\eta)$ controls the guaranteed drop, not the actual loss. Its sign gives the strict-descent window, and its maximum occurs at $\eta=1/L$.*

### Geometric meaning: minimize the guaranteed upper model

Along the descent ray, the Descent-Lemma upper bound is a parabola in $\eta$:

$$
q(\eta)
=f(x)-\eta\|g\|_2^2+\frac{L\eta^2}{2}\|g\|_2^2,
\qquad
f(x-\eta g)\le q(\eta).
$$

It starts at $q(0)=f(x)$, reaches its minimum at $1/L$, and returns to the starting height at $2/L$. Hence the step $1/L$ is best understood as **walking to the bottom of the guaranteed upper parabola**—not necessarily to the minimum of the actual loss along that ray.

![guaranteed-upper-model-step]({{ '/assets/notes/gradient-descent/guaranteed-upper-model-step.png' | relative_url }})
*The orange curve is the guaranteed upper model $q(\eta)$, not the actual blue loss. The step $1/L$ minimizes this upper bound; the actual best step may differ.*

> [!note]- Real training: deterministic window versus stochastic updates
> The bound above is an **exact full-gradient statement**. Mini-batch training uses an estimate $\hat g_k$ rather than the true gradient $g_k=\nabla f(x_k)$. Smoothness then gives
>
> $$
> f(x_k-\eta\hat g_k)
> \le
> f(x_k)-\eta\langle g_k,\hat g_k\rangle
> +\frac{L\eta^2}{2}\|\hat g_k\|_2^2.
> $$
>
> A noisy direction need not align with $g_k$, so an individual mini-batch step need not lower the full training objective. If the estimator is conditionally unbiased and
>
> $$
> \mathbb E[\hat g_k\mid x_k]=g_k,
> \qquad
> V_k:=\mathbb E[\|\hat g_k-g_k\|_2^2\mid x_k],
> $$
>
> then
>
> $$
> \boxed{
> \mathbb E[f(x_{k+1})\mid x_k]
> \le
> f(x_k)
> -\eta\left(1-\frac{L\eta}{2}\right)\|g_k\|_2^2
> +\frac{L\eta^2}{2}V_k
> }.
> $$
>
> The first term is deterministic progress; the last is the cost of gradient noise. Near a stationary point $\|g_k\|$ is small, so the noise term can dominate unless the learning rate or variance is reduced. This is why stochastic theory usually proves decrease **in expectation** or an average stationarity bound, rather than monotone loss on every step.
>
> | Practical mechanism | What it changes | What it does **not** prove by itself |
> |---|---|---|
> | Backtracking line search | Tests a local sufficient-decrease condition and shrinks the trial step | A globally optimal learning rate |
> | Warmup and decay schedules | Control early instability and later noise | A schedule derived from one fixed global $L$ |
> | Larger mini-batches | Usually reduce $V_k$ | Deterministic full-gradient descent |
> | Momentum | Adds state and changes the stability region | Validity of the plain-GD window $0<\eta<2/L$ for the new dynamics |
> | Adam/RMSProp | Rescale coordinates using gradient statistics | Removal of base-learning-rate tuning |
> | Gradient clipping | Caps the applied update | Smoothness of the original objective |
> | Normalization/preconditioning | Change effective curvature and conditioning | A universal guarantee independent of the model and data |

> [!warning] Training descent is not validation improvement
> This section concerns the objective whose gradient is used—usually empirical training loss. Even monotone training loss would not imply monotone validation loss or better generalization; those are statistical questions, not consequences of $L$-smoothness.

> [!warning] Trap — guaranteed descent is weaker than convergence to a good point
> With $\eta=1/L$, each full-gradient step lowers $f$ by at least $\tfrac{1}{2L}\|\nabla f(x_k)\|_2^2$. But nothing here assumes a minimizer exists, that $f$ is convex, or even that $f$ is bounded below. Descent is a local progress statement; convergence requires an accounting argument and at least a finite lower bound. That is the task of §1.10.

---

## 1.10 From descent to stationarity: the finite-budget argument

Section §1.9 gave a local statement about one update. This section adds those statements across time. If the objective has only a finite amount of height available to lose, then the squared gradients—which pay for every decrease—cannot remain large forever.

> [!important] Assumptions and what is deliberately absent
> Let $f$ be $L$-smooth on a region containing the full-gradient iterates
>
> $$
> x_{k+1}=x_k-\eta\nabla f(x_k),
> \qquad
> 0<\eta\le\frac1L.
> $$
>
> Assume only that $f$ has a finite lower bound
>
> $$
> f^\star:=\inf_x f(x)>-\infty.
> $$
>
> No convexity is assumed, and $f^\star$ need not be attained by any point $x^\star$.

Write

$$
f_k:=f(x_k),
\qquad
g_k:=\nabla f(x_k),
\qquad
\Delta_0:=f_0-f^\star.
$$

### Step 1 — every gradient spends descent budget

For $0<\eta\le1/L$, §1.9 gives

$$
f_{k+1}\le f_k-\frac{\eta}{2}\|g_k\|_2^2,
$$

or equivalently,

$$
\boxed{
\frac{\eta}{2}\|g_k\|_2^2
\le
f_k-f_{k+1}
}.
$$

The left side is the decrease forced by the current gradient. The entire run can spend at most $\Delta_0$: once the objective reaches its lower bound, no height remains.

### Step 2 — telescope all the drops

Sum the one-step inequality over $k=0,\ldots,K-1$:

$$
\frac{\eta}{2}\sum_{k=0}^{K-1}\|g_k\|_2^2
\le
\sum_{k=0}^{K-1}(f_k-f_{k+1}).
$$

The right side telescopes:

$$
(f_0-f_1)+(f_1-f_2)+\cdots+(f_{K-1}-f_K)=f_0-f_K.
$$

Since $f_K\ge f^\star$,

$$
\boxed{
\sum_{k=0}^{K-1}\|g_k\|_2^2
\le
\frac{2(f_0-f^\star)}{\eta}
=\frac{2\Delta_0}{\eta}
}.
$$

The right side is independent of $K$. This fixed bound on a growing sum of nonnegative terms is the load-bearing fact.

### Consequence A — a finite-horizon best-iterate rate

The smallest of $K$ nonnegative numbers cannot exceed their average:

$$
\boxed{
\min_{0\le k<K}\|g_k\|_2^2
\le
\frac1K\sum_{k=0}^{K-1}\|g_k\|_2^2
\le
\frac{2\Delta_0}{\eta K}
}.
$$

For the standard choice $\eta=1/L$,

$$
\boxed{
\min_{0\le k<K}\|\nabla f(x_k)\|_2^2
\le
\frac{2L\big(f(x_0)-f^\star\big)}{K}
}.
$$

| Quantity controlled | Worst-case finite-horizon rate |
|---|---:|
| Best squared gradient, $\min_{k<K}\|g_k\|_2^2$ | $O(1/K)$ |
| Best gradient norm, $\min_{k<K}\|g_k\|_2$ | $O(1/\sqrt K)$ |

Thus reaching an $\varepsilon$-stationary point—some iterate with $\|g_k\|_2\le\varepsilon$—is guaranteed once

$$
K\ge\frac{2\Delta_0}{\eta\varepsilon^2}.
$$

At $\eta=1/L$, this is $K\ge2L\Delta_0/\varepsilon^2$. The rate is for the **best iterate seen**, not necessarily the last iterate: loss values are monotone here, but gradient norms need not be.

### Consequence B — the whole gradient sequence tends to zero

Because the partial sums above are increasing and uniformly bounded,

$$
\sum_{k=0}^{\infty}\|g_k\|_2^2<\infty.
$$

A convergent series of nonnegative terms must have terms tending to zero; otherwise infinitely many terms would remain above some fixed positive threshold and the sum would diverge. Hence

$$
\boxed{\|\nabla f(x_k)\|_2\longrightarrow0.}
$$

This asymptotic conclusion concerns every iterate, but it supplies no $O(1/\sqrt k)$ rate for the last iterate. That rate belongs to the finite-horizon minimum above.

![descent-budget-and-stationarity]({{ '/assets/notes/gradient-descent/descent-budget-and-stationarity.png' | relative_url }})
*Left: every step spends part of the finite budget $f_0-f^\star$, although the monotone loss may plateau above $f^\star$. Top right: the $O(1/\sqrt K)$ guarantee is for the best gradient norm among the first $K$ iterates. Bottom right: the entire deterministic gradient sequence tends to zero, possibly irregularly, with no last-iterate rate proved here.*

### What the theorem does—and does not—make converge

Since $f_k$ is non-increasing and bounded below, it converges to some value $\bar f\ge f^\star$. Also,

$$
\sum_{k=0}^{\infty}\|x_{k+1}-x_k\|_2^2
=
\eta^2\sum_{k=0}^{\infty}\|g_k\|_2^2
<\infty.
$$

But square-summable steps need not have finite total path length: $\sum 1/k^2<\infty$ while $\sum1/k=\infty$. Therefore shrinking steps do not by themselves prove that the positions $x_k$ settle at one point.

| Statement | Proved under the current assumptions? |
|---|:---:|
| $f_{k+1}\le f_k$ | Yes |
| $f_k\to\bar f$ for some $\bar f\ge f^\star$ | Yes |
| $\min_{k<K}\|g_k\|_2=O(1/\sqrt K)$ | Yes |
| $\|g_k\|_2\to0$ for the whole sequence | Yes |
| $x_k\to x_\infty$ | Not in general |
| $\bar f=f^\star$ | Not in general |
| A stationary point is a local or global minimum | Not in general |

If $x_k$ does converge to a finite limit $x_\infty$, continuity of the gradient then gives $\nabla f(x_\infty)=0$. The missing step is proving that such a limit exists.

> [!example]- A globally smooth function whose iterates escape while its gradient vanishes
> Let
>
> $$
> f(x)=\arctan x,
> \qquad
> f'(x)=\frac1{1+x^2}.
> $$
>
> The function is bounded below by $-\pi/2$, its second derivative is globally bounded, and it has no finite stationary point. Gradient descent moves left at every step:
>
> $$
> x_{k+1}=x_k-\frac{\eta}{1+x_k^2}.
> $$
>
> The sequence cannot converge to a finite number, because the step would then approach a fixed negative value; instead $x_k\to-\infty$, while $f'(x_k)\to0$ and $f(x_k)\to-\pi/2$. Thus “gradient tends to zero” is strictly weaker than “iterates converge to a stationary point.”

> [!note]- Real training: mini-batch noise leaves a stationarity floor
> Let $\hat g_k$ be conditionally unbiased and have bounded conditional variance:
>
> $$
> \mathbb E[\hat g_k\mid x_k]=g_k,
> \qquad
> \mathbb E[\|\hat g_k-g_k\|_2^2\mid x_k]\le\sigma^2.
> $$
>
> Combining the stochastic bound from §1.9 with $\eta\le1/L$, then telescoping in expectation, gives
>
> $$
> \boxed{
> \frac1K\sum_{k=0}^{K-1}\mathbb E\|g_k\|_2^2
> \le
> \frac{2\Delta_0}{\eta K}
> +L\eta\sigma^2
> }.
> $$
>
> | Term | Meaning |
> |---|---|
> | $2\Delta_0/(\eta K)$ | Optimization term; vanishes as training continues |
> | $L\eta\sigma^2$ | Noise floor under a constant learning rate |
>
> A smaller learning rate lowers the floor but slows the first term; a larger batch usually reduces $\sigma^2$; learning-rate decay trades early progress for later precision. Individual mini-batch steps need not lower the full training loss, so practice often keeps the last checkpoint or the validation-best checkpoint rather than the theoretical minimum-full-gradient iterate. Momentum and adaptive optimizers alter the dynamics and require their own analysis.

---

## 1.11 (Stretch) The PL condition: when stationarity certifies global progress

Section §1.10 drives the gradient toward zero, but smoothness alone does not say whether a small gradient occurs near the global optimum or on a bad plateau. The **Polyak–Łojasiewicz (PL) condition** supplies exactly that missing link.

Define the function-value suboptimality

$$
\delta(x):=f(x)-f^\star,
\qquad
\delta_k:=f(x_k)-f^\star,
\qquad
f^\star:=\inf_x f(x).
$$

> [!important] Polyak–Łojasiewicz inequality
> A differentiable function satisfies the PL condition with constant $\mu>0$ on a region $D$ if
>
> $$
> \boxed{
> \frac12\|\nabla f(x)\|_2^2
> \ge
> \mu\big(f(x)-f^\star\big)
> \qquad\text{for every }x\in D
> }.
> $$
>
> Equivalently,
>
> $$
> f(x)-f^\star
> \le
> \frac1{2\mu}\|\nabla f(x)\|_2^2.
> $$
>
> A large optimality gap must produce a correspondingly large gradient. The condition rules out **uniformly small gradients far above the optimum**.

Two consequences are immediate:

$$
\nabla f(x)=0
\quad\Longrightarrow\quad
f(x)=f^\star,
$$

and

$$
\|\nabla f(x)\|_2\le\varepsilon
\quad\Longrightarrow\quad
f(x)-f^\star\le\frac{\varepsilon^2}{2\mu}.
$$

Thus every stationary point in a PL region is globally optimal. The global minimizer need not be unique; PL forbids **suboptimal** stationary points, not multiple equivalent optima.

![polyak-lojasiewicz-gradient-dominance]({{ '/assets/notes/gradient-descent/polyak-lojasiewicz-gradient-dominance.png' | relative_url }})
*Qualitative consequence, not a complete characterization: PL forbids a high-loss region whose gradient is too small relative to its optimality gap. Merely having a nonzero slope away from the minimum is weaker than the uniform quantitative inequality with one constant $\mu>0$.*

### PL is not another name for convexity

| Condition | What it controls | Can the function be non-convex? | Are suboptimal stationary points excluded? |
|---|---|:---:|:---:|
| Convexity | The function lies above every tangent plane | No | Yes |
| $\mu$-strong convexity | Convexity plus a uniform quadratic lower curvature | No | Yes; the minimizer is unique |
| PL | Gradient size versus function-value gap | Yes | Yes; minimizers may be non-unique |

The logical relationships are

$$
\boxed{\text{strong convexity}\Longrightarrow\text{PL}},
\qquad
\text{PL}\not\Longrightarrow\text{convexity},
\qquad
\text{convexity}\not\Longrightarrow\text{PL}.
$$

So PL is weaker than **strong** convexity and compatible with non-convexity; it is not a weaker form of ordinary convexity.

> [!example]- Convex does not imply PL: a flat minimum
> For $f(x)=x^4$,
>
> $$
> f^\star=0,
> \qquad
> f'(x)=4x^3.
> $$
>
> PL would require $8x^6\ge\mu x^4$, hence $8x^2\ge\mu$ for every nonzero $x$. This fails as $x\to0$ for every fixed $\mu>0$. The function is convex and is $L$-smooth on every bounded interval, but its minimum is too flat for PL.

> [!example]- PL does not imply convexity: a curved set of global minima
> Let
>
> $$
> f(x,y)=\frac12(y-\sin x)^2,
> \qquad
> f^\star=0.
> $$
>
> With $r:=y-\sin x$,
>
> $$
> \nabla f=(-r\cos x,r),
> \qquad
> \frac12\|\nabla f\|_2^2
> =\frac12r^2(1+\cos^2x)
> \ge f.
> $$
>
> Hence PL holds with $\mu=1$, while the Hessian is indefinite at some points, so the function is non-convex. Every point on $y=\sin x$ is a global minimizer. For the gradient-descent theorem below, $L$-smoothness must additionally hold on the bounded region traversed by the iterates.

> [!abstract]- Why strong convexity implies PL
> Strong convexity gives, for $g=\nabla f(x)$,
>
> $$
> f(y)\ge f(x)+\langle g,y-x\rangle+\frac{\mu}{2}\|y-x\|_2^2.
> $$
>
> The quadratic lower model is minimized at $y=x-g/\mu$, with value $f(x)-\|g\|_2^2/(2\mu)$. Taking the infimum over $y$ therefore gives
>
> $$
> f^\star\ge f(x)-\frac1{2\mu}\|g\|_2^2,
> $$
>
> which rearranges to the PL inequality.

### Combine PL with the descent guarantee

Assume the iterates stay in a region where $f$ is both $L$-smooth and $\mu$-PL, and take $\eta=1/L$. Section §1.9 gives

$$
\delta_{k+1}
\le
\delta_k-\frac1{2L}\|\nabla f(x_k)\|_2^2.
$$

PL converts the gradient term into the same quantity $\delta_k$:

$$
\frac1{2L}\|\nabla f(x_k)\|_2^2
\ge
\frac{\mu}{L}\delta_k.
$$

Therefore

$$
\boxed{
\delta_{k+1}
\le
\left(1-\frac{\mu}{L}\right)\delta_k
}.
$$

Unrolling the same contraction $K$ times gives

$$
\boxed{
f(x_K)-f^\star
\le
\left(1-\frac{\mu}{L}\right)^K
\big(f(x_0)-f^\star\big)
}.
$$

This is **linear (geometric) convergence**: “linear” means that every iteration removes a fixed fraction of the remaining error, not that the error decreases along a straight line. On a semilog plot, the geometric sequence becomes a straight line.

### The gradient–gap sandwich and the condition number

Smoothness gives the reverse comparison. Apply one step $x-\nabla f(x)/L$ in the Descent Lemma and use $f^\star\le f(x-\nabla f(x)/L)$:

$$
\frac12\|\nabla f(x)\|_2^2
\le
L\big(f(x)-f^\star\big).
$$

Together with PL,

$$
\boxed{
\mu\big(f(x)-f^\star\big)
\le
\frac12\|\nabla f(x)\|_2^2
\le
L\big(f(x)-f^\star\big)
}.
$$

At any suboptimal point this implies $\mu\le L$, so the contraction factor lies in $[0,1)$. It also makes gradient norm and function-value gap equivalent up to constants inside the PL region.

Define the condition number

$$
\kappa:=\frac{L}{\mu}.
$$

Then the rate is

$$
\delta_K
\le
\left(1-\frac1\kappa\right)^K\delta_0
\le
e^{-K/\kappa}\delta_0.
$$

To guarantee $\delta_K\le\varepsilon$, it is enough to take

$$
\boxed{
K\ge
\kappa\log\frac{\delta_0}{\varepsilon}
}.
$$

A large $\kappa$ means that the upper curvature scale $L$ is large relative to the gradient-dominance scale $\mu$, so each guaranteed contraction is weak. Section §1.12 makes this directional on a quadratic valley.

| Section | Quantity controlled | Guarantee |
|---|---|---|
| §1.10, smooth non-convex | $\min_{k<K}\|\nabla f(x_k)\|_2$ | $O(1/\sqrt K)$ stationarity |
| Smooth convex, without PL | $f(x_K)-f^\star$ | $O(1/K)$ under the usual minimizer assumptions |
| §1.11, smooth + PL | $f(x_K)-f^\star$ | Geometric, $(1-1/\kappa)^K$ |

Rates should be compared only when they measure the same object. The first row is not a function-gap rate; PL is what connects that stationarity measure to global suboptimality.

![function-gap-convergence-rates]({{ '/assets/notes/gradient-descent/function-gap-convergence-rates.png' | relative_url }})
*Every curve measures normalized function-value suboptimality. PL gives a straight line on the semilog plot; a larger condition number $\kappa=L/\mu$ makes that line flatter. The dashed $O(1/k)$ curve is an ordinary smooth-convex function-gap reference, not the non-convex gradient-norm bound from §1.10.*

> [!note]- PL also repairs the missing iterate-convergence step from §1.10
> Let $q:=1-\mu/L$. The geometric function-gap bound and the smoothness side of the sandwich give
>
> $$
> \|\nabla f(x_k)\|_2
> \le
> \sqrt{2L\delta_0}\,q^{k/2}.
> $$
>
> Hence, for $\eta=1/L$,
>
> $$
> \sum_{k=0}^{\infty}\|x_{k+1}-x_k\|_2
> =
> \eta\sum_{k=0}^{\infty}\|\nabla f(x_k)\|_2
> <\infty.
> $$
>
> The path now has finite total length, so $x_k$ is Cauchy and converges in $\mathbb R^n$ to a global minimizer. The limit need not be unique across initializations: PL permits a set of global minimizers, and neural-network parameter symmetries can represent the same predictor in many ways.

> [!note]- Real training: stochastic PL gives geometric decay to a noise floor
> Suppose the mini-batch estimator is conditionally unbiased and has bounded variance:
>
> $$
> \mathbb E[\hat g_k\mid x_k]=g_k,
> \qquad
> \mathbb E[\|\hat g_k-g_k\|_2^2\mid x_k]\le\sigma^2.
> $$
>
> For $\eta\le1/L$, the stochastic descent inequality and PL give
>
> $$
> \mathbb E[\delta_{k+1}\mid x_k]
> \le
> (1-\mu\eta)\delta_k
> +\frac{L\eta^2}{2}\sigma^2.
> $$
>
> Therefore
>
> $$
> \boxed{
> \mathbb E[\delta_K]
> \le
> (1-\mu\eta)^K\delta_0
> +\frac{L\eta\sigma^2}{2\mu}
> }.
> $$
>
> A constant learning rate produces fast initial contraction but a nonzero noise floor. Reducing $\eta$, increasing the batch size, decaying the learning rate, or using variance reduction can improve late-stage precision. Momentum and adaptive optimizers require different recurrences.

> [!warning] PL in deep learning is a conditional model, not a default fact
> Global PL is strong. Analyses of sufficiently wide or overparameterized networks often prove PL-like behavior only near initialization or on a sublevel set, under conditions on width, initialization scale, data separation, and the tangent-kernel spectrum. The guarantee applies only while the iterates remain in that region. Even then it explains optimization of the empirical training objective—not validation performance, generalization, or a uniquely meaningful parameter vector.

---

## Connections

- The update rule and quadratic upper bound used by these convergence arguments are derived in [Gradient Descent, Smoothness, and the Descent Lemma]({{ '/notes/mit6-7960-01-3-gradient-descent-smoothness-and-the-descent-lemma/' | relative_url }}).
- The roles of curvature, condition number, and step size become explicit in [Conditioning and Practical Gradient Descent]({{ '/notes/mit6-7960-01-5-conditioning-and-practical-gradient-descent/' | relative_url }}).
