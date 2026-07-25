---
layout: note
title: "Gaussian Smoothing and Evolution Strategies"
source_title: "MIT6.7960-02.3-Gaussian Smoothing and Evolution Strategies"
course: "MIT 6.7960"
sequence: "02.3"
source_context: "How to Train a Neural Net — Backpropagation and Differentiable Programming"
permalink: "/notes/mit6-7960-02-3-gaussian-smoothing-and-evolution-strategies/"
tags:
  - "math/probability"
  - "math/optimization"
  - "topic/gaussian-smoothing"
  - "topic/evolution-strategies"
---
Derivative-free optimization replaces computation-graph derivatives with information extracted from nearby function evaluations. This note builds that method from its probability foundations: it differentiates Gaussian densities with respect to their means, derives the log-derivative and Gaussian-smoothing identities, constructs the finite-sample evolution-strategy estimator, and then analyzes smoothing resolution, baselines, antithetic sampling, dimensional cost, and environmental noise. The material remains unified because every practical estimator choice traces back to the exact smoothed-objective identity established at the beginning.

## 2.5 Gaussian smoothing and derivative-free search

Cross-entropy repairs a discrete task metric by replacing it with a differentiable surrogate. A different problem arises when a scalar objective can be evaluated but its derivative cannot be obtained reliably. Evolution Strategies (ES) address that setting by perturbing the parameters, comparing function values, and estimating a direction from those comparisons.

The central identity used later is

$$
\boxed{
\nabla_\theta
\mathcal N(u;\theta,\sigma^2I)
=
\frac{u-\theta}{\sigma^2}
\mathcal N(u;\theta,\sigma^2I)
}.
$$

This formula requires very little probability theory. Once the Gaussian density is written down, it follows from the ordinary chain rule. The derivation below starts in one dimension so that the roles of the symbols and the source of the sign remain visible.

### What is being differentiated?

The one-dimensional Gaussian density is

$$
\boxed{
\mathcal N(u;\theta,\sigma^2)
=
\frac{1}{\sqrt{2\pi}\sigma}
\exp\left(
-\frac{(u-\theta)^2}{2\sigma^2}
\right)
}.
$$

| Symbol | Role in this derivative |
|---|---|
| $u$ | the fixed location at which the density is evaluated |
| $\theta$ | the center, or mean, of the Gaussian; this is the variable being differentiated |
| $\sigma$ | the fixed standard deviation controlling the width of the bell curve |
| $\sigma^2$ | the variance |

The question is

$$
\frac{\partial}{\partial\theta}
\mathcal N(u;\theta,\sigma^2).
$$

In words:

> Hold the observation location $u$ and the width $\sigma$ fixed. Move the center $\theta$ of the bell curve and ask how the density at that fixed $u$ changes.

This distinction matters. We are not moving the observation $u$ along a fixed curve; we are moving the curve underneath a fixed observation.

> [!note] Density is not point probability
> For a continuous random variable, the probability of exactly one point is zero. $\mathcal N(u;\theta,\sigma^2)$ is a **density**: probabilities are obtained by integrating it over intervals. The derivative here measures how the height of that density at the fixed location $u$ changes as its center moves.

### One-dimensional derivation by the chain rule

Because $\sigma$ is fixed, the normalizing factor

$$
C
=
\frac{1}{\sqrt{2\pi}\sigma}
$$

is constant with respect to $\theta$. Define the exponent

$$
h(\theta)
=
-\frac{(u-\theta)^2}{2\sigma^2}.
$$

Then the density is simply

$$
\mathcal N(u;\theta,\sigma^2)
=
C e^{h(\theta)}.
$$

The exponential chain rule gives

$$
\frac{d}{d\theta}e^{h(\theta)}
=
e^{h(\theta)}h'(\theta).
$$

Therefore,

$$
\frac{\partial}{\partial\theta}
\mathcal N(u;\theta,\sigma^2)
=
C e^{h(\theta)}h'(\theta).
$$

Since

$$
C e^{h(\theta)}
=
\mathcal N(u;\theta,\sigma^2),
$$

the remaining task is to calculate $h'(\theta)$.

Start with the inner square:

$$
\frac{d}{d\theta}(u-\theta)^2
=
2(u-\theta)
\frac{d}{d\theta}(u-\theta).
$$

The location $u$ is held fixed, so

$$
\frac{d}{d\theta}(u-\theta)
=
-1.
$$

Hence

$$
\frac{d}{d\theta}(u-\theta)^2
=
-2(u-\theta).
$$

Substitution into the exponent gives

$$
\begin{aligned}
h'(\theta)
&=
-\frac{1}{2\sigma^2}
\frac{d}{d\theta}(u-\theta)^2\\
&=
-\frac{1}{2\sigma^2}
\left[-2(u-\theta)\right]\\
&=
\frac{u-\theta}{\sigma^2}.
\end{aligned}
$$

The positive sign comes from two negatives:

1. the Gaussian exponent contains a leading minus sign;
2. differentiating $u-\theta$ with respect to $\theta$ produces another minus sign.

The negatives cancel. Returning to the outer exponential,

$$
\begin{aligned}
\frac{\partial}{\partial\theta}
\mathcal N(u;\theta,\sigma^2)
&=
C e^{h(\theta)}
\frac{u-\theta}{\sigma^2}\\
&=
\boxed{
\frac{u-\theta}{\sigma^2}
\mathcal N(u;\theta,\sigma^2)
}.
\end{aligned}
$$

The type of every factor is worth checking:

| Quantity | Type in one dimension |
|---|---|
| $u-\theta$ | scalar displacement |
| $1/\sigma^2$ | scalar inverse variance |
| $\mathcal N(u;\theta,\sigma^2)$ | positive scalar density |
| $\partial_\theta\mathcal N$ | scalar slope with respect to the mean |

### The sign has a direct geometric meaning

Suppose

$$
u=2,
\qquad
\theta=0.
$$

Then

$$
u-\theta=2>0,
$$

so

$$
\frac{\partial}{\partial\theta}
\mathcal N(u;\theta,\sigma^2)
>0.
$$

Increasing $\theta$ moves the Gaussian center to the right, toward $u=2$. The fixed observation becomes closer to the center, so its density increases.

If instead

$$
u=-2,
\qquad
\theta=0,
$$

then

$$
u-\theta=-2<0.
$$

Increasing $\theta$ now moves the center to the right and farther away from $u=-2$, so the density at the fixed observation decreases. The derivative is negative.

Thus

$$
\boxed{
\frac{u-\theta}{\sigma^2}
\text{ points in the direction that moves the mean toward }u.
}
$$

The inverse variance controls sensitivity. For the same displacement $u-\theta$, a narrow Gaussian has small $\sigma$ and changes sharply when its center moves; a wide Gaussian has large $\sigma$ and changes more gently.

![gaussian-density-mean-gradient]({{ '/assets/notes/neural-network-training/gaussian-density-mean-gradient.png' | relative_url }})

### Differentiating with respect to $u$ gives the opposite sign

A common error is to forget which symbol is the differentiation variable. With respect to the mean,

$$
\boxed{
\nabla_\theta
\mathcal N(u;\theta,\sigma^2)
=
\frac{u-\theta}{\sigma^2}
\mathcal N(u;\theta,\sigma^2)
}.
$$

With respect to the observation location,

$$
\boxed{
\nabla_u
\mathcal N(u;\theta,\sigma^2)
=
-\frac{u-\theta}{\sigma^2}
\mathcal N(u;\theta,\sigma^2)
}.
$$

Therefore,

$$
\boxed{
\nabla_\theta\mathcal N
=
-\nabla_u\mathcal N
}.
$$

The geometry explains the sign difference:

| Operation | Effect when $u>\theta$ | Derivative sign |
|---|---|---:|
| increase $\theta$ | moves the center toward fixed $u$ | positive |
| increase $u$ | moves the observation away from fixed $\theta$ | negative |

### Extension to $d$ dimensions

Let

$$
u,\theta\in\mathbb R^d.
$$

For an isotropic Gaussian with covariance $\sigma^2I$,

$$
\boxed{
\mathcal N(u;\theta,\sigma^2I)
=
\frac{1}{(2\pi)^{d/2}\sigma^d}
\exp\left(
-\frac{\lVert u-\theta\rVert_2^2}{2\sigma^2}
\right)
}.
$$

The covariance $\sigma^2I$ means that every coordinate has the same variance $\sigma^2$ and the Gaussian has spherical level sets. The squared Euclidean distance expands as

$$
\lVert u-\theta\rVert_2^2
=
\sum_{j=1}^{d}(u_j-\theta_j)^2.
$$

Define the exponent

$$
h(\theta)
=
-\frac1{2\sigma^2}
\sum_{j=1}^{d}(u_j-\theta_j)^2.
$$

The gradient stacks one partial derivative for each component of $\theta$:

$$
\nabla_\theta h(\theta)
=
\begin{bmatrix}
\dfrac{\partial h}{\partial\theta_1}\\[4pt]
\vdots\\[4pt]
\dfrac{\partial h}{\partial\theta_d}
\end{bmatrix}.
$$

For a particular coordinate $k$, only one term of the sum depends on $\theta_k$:

$$
\begin{aligned}
\frac{\partial h}{\partial\theta_k}
&=
-\frac1{2\sigma^2}
\frac{\partial}{\partial\theta_k}
(u_k-\theta_k)^2\\
&=
-\frac1{2\sigma^2}
\left[-2(u_k-\theta_k)\right]\\
&=
\frac{u_k-\theta_k}{\sigma^2}.
\end{aligned}
$$

Stacking every coordinate gives

$$
\nabla_\theta h(\theta)
=
\begin{bmatrix}
\dfrac{u_1-\theta_1}{\sigma^2}\\[4pt]
\vdots\\[4pt]
\dfrac{u_d-\theta_d}{\sigma^2}
\end{bmatrix}
=
\frac{u-\theta}{\sigma^2}.
$$

Applying the exponential chain rule again,

$$
\begin{aligned}
\nabla_\theta
\mathcal N(u;\theta,\sigma^2I)
&=
\mathcal N(u;\theta,\sigma^2I)
\nabla_\theta h(\theta)\\
&=
\boxed{
\frac{u-\theta}{\sigma^2}
\mathcal N(u;\theta,\sigma^2I)
}.
\end{aligned}
$$

Now the types are:

| Quantity | Type in $d$ dimensions |
|---|---|
| $u-\theta$ | vector in $\mathbb R^d$ |
| $1/\sigma^2$ | scalar |
| $\mathcal N(u;\theta,\sigma^2I)$ | positive scalar density |
| $\nabla_\theta\mathcal N$ | vector in $\mathbb R^d$ |

The right-hand side is therefore a scalar multiple of the displacement vector $u-\theta$. The gradient with respect to the mean points from the current mean toward the observation.

### Two-dimensional sanity check

Take

$$
\theta
=
\begin{bmatrix}
0\\
0
\end{bmatrix},
\qquad
u
=
\begin{bmatrix}
2\\
-1
\end{bmatrix},
\qquad
\sigma=1.
$$

Then

$$
\frac{u-\theta}{\sigma^2}
=
\begin{bmatrix}
2\\
-1
\end{bmatrix}.
$$

Therefore,

$$
\nabla_\theta
\mathcal N(u;\theta,I)
=
\mathcal N(u;\theta,I)
\begin{bmatrix}
2\\
-1
\end{bmatrix}.
$$

Because the density is positive, it changes the magnitude but not the direction. The mean-gradient points exactly from

$$
\theta=(0,0)
$$

toward

$$
u=(2,-1).
$$

This is the multidimensional version of shifting the one-dimensional bell curve toward a fixed observation.

### The log-derivative trick

The same calculation can be shortened by differentiating the logarithm of the density first. In one dimension,

$$
\log\mathcal N(u;\theta,\sigma^2)
=
-\log(\sqrt{2\pi}\sigma)
-
\frac{(u-\theta)^2}{2\sigma^2}.
$$

The first term is constant with respect to $\theta$, so in one dimension

$$
\frac{\partial}{\partial\theta}
\log\mathcal N(u;\theta,\sigma^2)
=
\frac{u-\theta}{\sigma^2}.
$$

The same coordinatewise calculation in $d$ dimensions gives

$$
\boxed{
\nabla_\theta
\log\mathcal N(u;\theta,\sigma^2I)
=
\frac{u-\theta}{\sigma^2}
}.
$$

For any positive differentiable function $f(\theta)$,

$$
\nabla_\theta\log f(\theta)
=
\frac{\nabla_\theta f(\theta)}{f(\theta)}.
$$

Multiplying by $f(\theta)$ gives the general identity

$$
\boxed{
\nabla_\theta f(\theta)
=
f(\theta)
\nabla_\theta\log f(\theta)
}.
$$

Applying it to the Gaussian density,

$$
\begin{aligned}
\nabla_\theta
\mathcal N(u;\theta,\sigma^2I)
&=
\mathcal N(u;\theta,\sigma^2I)
\nabla_\theta
\log\mathcal N(u;\theta,\sigma^2I)\\
&=
\mathcal N(u;\theta,\sigma^2I)
\frac{u-\theta}{\sigma^2}.
\end{aligned}
$$

This is called the **log-derivative trick** or **score-function identity**. The vector

$$
\nabla_\theta\log p_\theta(u)
$$

is called the score with respect to the parameter $\theta$. The same device reappears in Evolution Strategies, policy gradients, and maximum-likelihood estimation.

> [!important] Why take a logarithm?
> The logarithm turns a product-like density into a sum and removes the outer exponential from the derivative. The original density is recovered afterward through
> $$
> \nabla f=f\nabla\log f.
> $$
> This is an algebraic convenience, not a change to the underlying probability model.

### Why this identity produces the ES perturbation vector

Define the Gaussian-smoothed objective

$$
\boxed{
J_\sigma(\theta)
=
\int_{\mathbb R^d}
J(u)
\mathcal N(u;\theta,\sigma^2I)
\,du
}.
$$

This integral is a weighted average of objective values near $\theta$:

- $u$ ranges over possible perturbed parameter vectors;
- $J(u)$ is the objective at that perturbed vector;
- $\mathcal N(u;\theta,\sigma^2I)$ assigns more weight to points near the center $\theta$;
- $\sigma$ controls how wide a neighborhood is averaged.

Equivalently, if

$$
\varepsilon\sim\mathcal N(0,I),
$$

then

$$
u=\theta+\sigma\varepsilon
$$

has distribution $\mathcal N(\theta,\sigma^2I)$, so

$$
J_\sigma(\theta)
=
\mathbb E_\varepsilon
\left[
J(\theta+\sigma\varepsilon)
\right].
$$

In the integral representation, $J(u)$ contains no explicit $\theta$. Under the regularity conditions that permit differentiating under the integral sign,

$$
\nabla_\theta J_\sigma(\theta)
=
\int
J(u)
\nabla_\theta
\mathcal N(u;\theta,\sigma^2I)
\,du.
$$

Insert the Gaussian mean-derivative identity:

$$
\nabla_\theta J_\sigma(\theta)
=
\int
J(u)
\frac{u-\theta}{\sigma^2}
\mathcal N(u;\theta,\sigma^2I)
\,du.
$$

Now substitute

$$
u=\theta+\sigma\varepsilon.
$$

Then

$$
u-\theta
=
\sigma\varepsilon,
$$

and therefore

$$
\frac{u-\theta}{\sigma^2}
=
\frac{\varepsilon}{\sigma}.
$$

The integral becomes an expectation over standard Gaussian perturbations:

$$
\boxed{
\nabla J_\sigma(\theta)
=
\frac1\sigma
\mathbb E_{\varepsilon\sim\mathcal N(0,I)}
\left[
J(\theta+\sigma\varepsilon)
\varepsilon
\right]
}.
$$

The perturbation vector $\varepsilon$ in the ES estimator is therefore not an informal rule added by hand. It comes directly from differentiating the Gaussian density with respect to its center:

$$
\boxed{
\frac{u-\theta}{\sigma^2}
\xrightarrow{\;u=\theta+\sigma\varepsilon\;}
\frac{\varepsilon}{\sigma}.
}
$$

> [!summary] The complete derivation chain
> $$
> \mathcal N(u;\theta,\sigma^2I)
> =
> C\exp\left(
> -\frac{\lVert u-\theta\rVert^2}{2\sigma^2}
> \right)
> $$
> $$
> \Downarrow
> $$
> $$
> \nabla_\theta
> \left(
> -\frac{\lVert u-\theta\rVert^2}{2\sigma^2}
> \right)
> =
> \frac{u-\theta}{\sigma^2}
> $$
> $$
> \Downarrow
> $$
> $$
> \nabla_\theta\mathcal N
> =
> \mathcal N\frac{u-\theta}{\sigma^2}
> $$
> $$
> \Downarrow
> $$
> $$
> \nabla J_\sigma(\theta)
> =
> \frac1\sigma
> \mathbb E
> \left[
> J(\theta+\sigma\varepsilon)\varepsilon
> \right].
> $$

The probability theory supplies the Gaussian density and the interpretation of its integral. The derivative itself uses only the square rule, the exponential chain rule, and the definition of a multivariable gradient.

### From a zeroth-order oracle to a direction

Suppose the goal is

$$
\min_{\theta\in\mathbb R^d}J(\theta),
$$

but the only available operation is

$$
\theta
\longmapsto
J(\theta).
$$

This is a **zeroth-order oracle**: it returns a scalar function value but no derivative. The computation behind $J$ might contain a simulator, discrete decisions, an external program, a non-differentiable metric, or operations without backward rules.

The problem is therefore not “how do we use a gradient?” but

$$
\boxed{
\text{Can comparisons of nearby function values reveal a useful search direction?}
}
$$

#### Start with two probes in one dimension

For a scalar parameter, evaluate both sides of the current point:

$$
J(\theta+\sigma),
\qquad
J(\theta-\sigma).
$$

If the objective is being minimized and

$$
J(\theta+\sigma)
<
J(\theta-\sigma),
$$

the right side appears better at scale $\sigma$. The central finite difference is

$$
\boxed{
\frac{J(\theta+\sigma)-J(\theta-\sigma)}{2\sigma}
}.
$$

When $J$ is sufficiently smooth and $\sigma$ is small,

$$
\frac{J(\theta+\sigma)-J(\theta-\sigma)}{2\sigma}
=
J'(\theta)+O(\sigma^2).
$$

Thus two function evaluations can approximate a one-dimensional derivative without differentiating the internal computation.

> [!example]- A quadratic probe gives the exact derivative
> Let
> $$
> J(\theta)=\frac12(\theta-3)^2,
> \qquad
> \theta=0,
> \qquad
> \sigma=0.5.
> $$
> The two probes are
> $$
> J(0.5)=\frac12(0.5-3)^2=3.125,
> $$
> $$
> J(-0.5)=\frac12(-0.5-3)^2=6.125.
> $$
> Therefore,
> $$
> \frac{J(0.5)-J(-0.5)}{2(0.5)}
> =
> \frac{3.125-6.125}{1}
> =-3.
> $$
> The analytic derivative is
> $$
> J'(\theta)=\theta-3,
> \qquad
> J'(0)=-3.
> $$
> The negative sign says that increasing $\theta$ lowers the objective. Under gradient descent,
> $$
> \theta_{k+1}=\theta_k-\eta(-3)=\theta_k+3\eta,
> $$
> so the parameter moves toward the minimizer at $3$. Central differences are exact for this quadratic because the third and higher derivatives vanish.

### Why coordinatewise finite differences do not scale

For $\theta\in\mathbb R^d$, a central difference for coordinate $j$ is

$$
\frac{
J(\theta+\sigma e_j)
-
J(\theta-\sigma e_j)
}{2\sigma},
$$

where $e_j$ is the $j$th standard basis vector. Computing all $d$ coordinates requires approximately

$$
2d
$$

complete objective evaluations. For a model with billions of parameters, that is infeasible.

Evolution Strategies avoid probing every coordinate separately. One random perturbation changes all coordinates at once:

$$
\varepsilon
\sim
\mathcal N(0,I),
\qquad
\theta_{\mathrm{probe}}
=
\theta+\sigma\varepsilon.
$$

Here

- $\varepsilon\in\mathbb R^d$ supplies a random direction and random length;
- $\sigma>0$ sets the perturbation scale;
- $J(\theta+\sigma\varepsilon)$ reports how that complete parameter perturbation performed.

Rather than paying two evaluations for every coordinate, ES draws $M$ random directions,

$$
\varepsilon_1,\ldots,\varepsilon_M
\overset{\mathrm{iid}}{\sim}
\mathcal N(0,I),
$$

and evaluates

$$
s_i
=
J(\theta+\sigma\varepsilon_i).
$$

### The basic finite-sample ES estimator

The Gaussian identity derived above gives

$$
\nabla J_\sigma(\theta)
=
\frac1\sigma
\mathbb E
\left[
J(\theta+\sigma\varepsilon)\varepsilon
\right].
$$

Replacing the expectation by a Monte Carlo average produces

$$
\boxed{
\widehat g_{\mathrm{ES}}
=
\frac{1}{\sigma M}
\sum_{i=1}^{M}
J(\theta+\sigma\varepsilon_i)
\varepsilon_i
}.
$$

For minimization,

$$
\boxed{
\theta_{k+1}
=
\theta_k
-
\eta\widehat g_{\mathrm{ES}}
}.
$$

For reward maximization, the sign changes:

$$
\theta_{k+1}
=
\theta_k
+
\eta\widehat g_R.
$$

Always identify whether the source uses a loss to minimize or a reward to maximize before interpreting the update sign.

The lecture compresses this estimator and a discontinuous example into one slide:

![evolution-strategies-estimator-source]({{ '/assets/notes/neural-network-training/evolution-strategies-estimator-source.png' | relative_url }})

*Source: MIT 6.7960 Fall 2024, Lecture 2, slide 21.*

> [!warning] Typographical correction to the source slide
> The update on the slide should contain an equality sign:
> $$
> \theta^{k+1}
> =
> \theta^k
> -
> \eta\frac1{\sigma M}
> \sum_{i=1}^{M}s_i\varepsilon_i.
> $$
> The slide's verbal phrase “move toward perturbations that achieved lower loss” is an intuition. The precise object estimated by the displayed formula is the gradient of the Gaussian-smoothed objective $J_\sigma$.

This estimator is unbiased for the smoothed gradient:

$$
\mathbb E[\widehat g_{\mathrm{ES}}]
=
\nabla J_\sigma(\theta),
$$

provided the samples are independent and the required expectations exist. It is generally **not** an unbiased estimator of $\nabla J(\theta)$.

### Why “score times direction” leaves a meaningful vector

Consider an uphill direction $v$: perturbations with a large component along $v$ tend to produce larger loss. When a sampled $\varepsilon_i$ points roughly along $v$, the product

$$
J(\theta+\sigma\varepsilon_i)\varepsilon_i
$$

therefore gives that direction a larger positive weight. Perturbations pointing roughly along $-v$ tend to receive smaller loss and hence a smaller weight in the opposite direction.

After averaging many symmetric random directions, unrelated components cancel in expectation while correlations between direction and score remain. The estimator records

$$
\boxed{
\text{which parameter perturbations systematically co-occur with larger objective values.}
}
$$

For minimization, stepping along $-\widehat g_{\mathrm{ES}}$ moves toward perturbations that tend to achieve smaller loss.

This covariance-like intuition is useful, but the Gaussian smoothing derivation is the mathematical justification. The estimator is not merely a heuristic weighted vote.

### ES differentiates a smoothed objective, not a discontinuity

The smoothed objective is

$$
J_\sigma(\theta)
=
\mathbb E_\varepsilon
\left[
J(\theta+\sigma\varepsilon)
\right].
$$

It evaluates the average performance of a Gaussian cloud of parameter vectors around $\theta$. This changes the optimization problem:

$$
\boxed{
\text{ES does not create a classical derivative of }J;
\quad
\text{it computes a gradient of }J_\sigma.
}
$$

#### A discontinuous step becomes a Gaussian CDF

Consider the minimization objective

$$
J(\theta)
=
\begin{cases}
0,&\theta<0,\\
1,&\theta\ge0.
\end{cases}
$$

The original derivative is zero at every $\theta\ne0$ and nonexistent at zero. Plain local differentiation therefore cannot report the lower region from a point on the positive half-line.

Gaussian smoothing gives

$$
\begin{aligned}
J_\sigma(\theta)
&=
\mathbb E[J(\theta+\sigma\varepsilon)]\\
&=
\Pr(\theta+\sigma\varepsilon\ge0)\\
&=
\Pr\left(\varepsilon\ge-\frac\theta\sigma\right)\\
&=
\Phi\left(\frac\theta\sigma\right),
\end{aligned}
$$

where $\Phi$ is the standard normal cumulative distribution function. Its derivative is

$$
\boxed{
J_\sigma'(\theta)
=
\frac1\sigma
\phi\left(\frac\theta\sigma\right)
>0,
}
$$

where $\phi$ is the standard normal density. Gradient descent on $J_\sigma$ therefore moves left, toward the lower region of the original step objective.

The mechanism is not that the pointwise derivative discovered a jump. It is that some random probes cross the jump, and their different function values change the local average.

> [!warning] Smoothing is still local at scale $\sigma$
> If $\theta\gg\sigma$, a Gaussian perturbation almost never crosses the boundary. Then
> $$
> \frac1\sigma
> \phi\left(\frac\theta\sigma\right)
> \approx0,
> $$
> so the smoothed signal is also tiny. ES replaces infinitesimal local information with neighborhood information at scale $\sigma$; it does not provide a global-search guarantee.

### $\sigma$ is a search resolution

The smoothing scale determines which geometric features survive in $J_\sigma$.

![gaussian-smoothing-resolution-tradeoff]({{ '/assets/notes/neural-network-training/gaussian-smoothing-resolution-tradeoff.png' | relative_url }})

| Choice | What the probes see | Benefit | Cost |
|---|---|---|---|
| small $\sigma$ | a narrow neighborhood | preserves fine detail and keeps $J_\sigma$ close to $J$ | may not cross discrete boundaries; finite differences are small; noise is amplified by $1/\sigma$ |
| large $\sigma$ | a broad neighborhood | crosses wider plateaus and reveals large-scale trends | changes the objective more; may erase narrow but good solutions |

The tradeoff is

$$
\boxed{
\text{small }\sigma:
\text{high resolution, short exploration range}
}
$$

versus

$$
\boxed{
\text{large }\sigma:
\text{low resolution, broad exploration range}.
}
$$

For a smooth $J$, $J_\sigma$ approaches $J$ as $\sigma\to0$ under suitable conditions. But the Monte Carlo estimator becomes harder to use when score noise or numerical error dominates the small perturbation effect. In practice, $\sigma$ is an optimization hyperparameter, not a harmless approximation constant.

### Constant offsets create variance but no true direction

Suppose a constant is added to the objective:

$$
\widetilde J(\theta)
=
J(\theta)+C.
$$

The true and smoothed gradients do not change:

$$
\nabla\widetilde J_\sigma(\theta)
=
\nabla J_\sigma(\theta).
$$

But one raw estimator term becomes

$$
\frac1\sigma
\left[
J(\theta+\sigma\varepsilon)+C
\right]
\varepsilon,
$$

which contains

$$
\frac C\sigma\varepsilon.
$$

Its expectation is zero because

$$
\mathbb E[\varepsilon]=0,
$$

but a finite sample average does not cancel it exactly. A large objective offset can therefore create large estimator variance despite containing no gradient information.

#### Subtracting a baseline

For any baseline $b$ independent of the current perturbation,

$$
\mathbb E[b\varepsilon]=0.
$$

Hence

$$
\nabla J_\sigma(\theta)
=
\frac1\sigma
\mathbb E
\left[
\left(J(\theta+\sigma\varepsilon)-b\right)
\varepsilon
\right].
$$

A baseline estimator is therefore

$$
\boxed{
\widehat g_b
=
\frac1{M\sigma}
\sum_{i=1}^{M}
\left[
J(\theta+\sigma\varepsilon_i)-b
\right]
\varepsilon_i.
}
$$

Possible baselines include $J(\theta)$, a running mean from previous iterations, or an independently estimated expected score.

> [!warning] A batch mean is not literally independent
> If $b$ is the mean of the same $M$ scores used in the estimator, it is correlated with each $\varepsilon_i$. The naive centered estimator is shrunk by a finite-sample factor rather than remaining exactly unbiased. This can be corrected with a leave-one-out baseline or a scale factor; in practice the induced scale change is often absorbed by the learning rate. The exact expectation claim applies directly only when the baseline is independent of the current perturbation.

### Antithetic sampling compares both ends of one random axis

For each $\varepsilon_i$, evaluate both

$$
\theta+\sigma\varepsilon_i
$$

and

$$
\theta-\sigma\varepsilon_i.
$$

The paired estimator is

$$
\boxed{
\widehat g_{\mathrm{anti}}
=
\frac1{2M\sigma}
\sum_{i=1}^{M}
\left[
J(\theta+\sigma\varepsilon_i)
-
J(\theta-\sigma\varepsilon_i)
\right]
\varepsilon_i.
}
$$

This is the high-dimensional random-direction analogue of a central difference. It asks:

> Along this one random axis, which end produces the smaller loss?

Any constant offset cancels exactly:

$$
\left[J_++C\right]
-
\left[J_-+C\right]
=
J_+-J_-.
$$

Symmetry of the Gaussian also shows that

$$
\mathbb E[\widehat g_{\mathrm{anti}}]
=
\nabla J_\sigma(\theta).
$$

The cost is $2M$ function evaluations rather than $M$, but the difference often carries a much cleaner directional signal.

> [!example]- Relation to a directional derivative when $J$ is smooth
> Taylor expansion gives
> $$
> J(\theta+\sigma\varepsilon)
> =
> J(\theta)
> +
> \sigma\nabla J(\theta)^\top\varepsilon
> +O(\sigma^2),
> $$
> $$
> J(\theta-\sigma\varepsilon)
> =
> J(\theta)
> -
> \sigma\nabla J(\theta)^\top\varepsilon
> +O(\sigma^2).
> $$
> Subtracting cancels the shared value and even-order terms:
> $$
> \frac{J(\theta+\sigma\varepsilon)-J(\theta-\sigma\varepsilon)}{2\sigma}
> \approx
> \nabla J(\theta)^\top\varepsilon.
> $$
> Multiplication by $\varepsilon$ produces
> $$
> (\nabla J(\theta)^\top\varepsilon)\varepsilon.
> $$
> Since a standard Gaussian satisfies
> $$
> \mathbb E[\varepsilon\varepsilon^\top]=I,
> $$
> averaging gives
> $$
> \mathbb E
> \left[
> (\nabla J^\top\varepsilon)\varepsilon
> \right]
> =
> \nabla J.
> $$
> For finite $\sigma$, the exact target remains $\nabla J_\sigma$; the Taylor calculation explains the small-$\sigma$ connection to $\nabla J$ when the original objective is smooth.

### ES and mini-batch SGD randomize different objects

Both methods are stochastic, but their randomness enters at different levels.

| Method | What is sampled? | Requires backpropagation? | Expected target |
|---|---|---:|---|
| mini-batch SGD | data examples | yes | gradient of the empirical/population objective |
| ES | parameter perturbations | no | gradient of a Gaussian-smoothed objective |
| stochastic ES | both data/environment outcomes and parameter perturbations | no | smoothed gradient plus score noise |

For a mini-batch $B$,

$$
g_B(\theta)
=
\frac1{|B|}
\sum_{i\in B}
\nabla_\theta\ell_i(\theta)
$$

still uses derivatives computed by backpropagation. ES instead observes complete scalar scores such as

$$
J(\theta+\sigma\varepsilon_i)
$$

and never differentiates their internal computation.

Thus

$$
\boxed{
\text{SGD is stochastic in data space and first-order;}
\quad
\text{ES is stochastic in parameter space and zeroth-order.}
}
$$

### What ES gains and what it discards

ES requires only the interface

$$
\theta
\longmapsto
\text{one scalar score}.
$$

That makes it applicable when

- the system is a black-box simulator;
- intermediate operations are discrete or non-differentiable;
- part of the computation is an external program;
- the available objective is an episodic reward or evaluation metric;
- backward rules are unavailable or unreliable.

The evaluations for distinct perturbations are independent and can be distributed across many workers. This is an important systems advantage.

The same minimal interface is also the main statistical disadvantage. Backpropagation uses every local derivative in the computation graph to produce the entire parameter gradient. ES throws away that internal structure and retains one scalar score per complete run.

| Property | Backpropagation | Evolution Strategies |
|---|---|---|
| internal requirement | differentiable graph with backward rules | runnable scalar objective |
| information extracted from a run | local sensitivities throughout the graph | one final score |
| target | gradient of the executed objective | gradient of a perturbed/smoothed objective |
| function-evaluation efficiency | usually high | often low |
| black-box and discrete systems | difficult | possible |
| parallel perturbation evaluation | not the main mechanism | natural |

> [!important] “Derivative-free” does not mean “cheap”
> The number of queries in the formula does not explicitly equal $d$, but estimator variance usually worsens with dimension. Many random directions may be required before irrelevant components cancel.

> [!example]- Exact dimension effect for a linear objective
> Let
> $$
> J(\theta)=a^\top\theta.
> $$
> An antithetic single-direction estimate reduces to
> $$
> \widehat g=(a^\top\varepsilon)\varepsilon,
> \qquad
> \mathbb E[\widehat g]=a.
> $$
> For $\varepsilon\sim\mathcal N(0,I)$,
> $$
> \mathbb E\lVert\widehat g-a\rVert^2
> =
> (d+1)\lVert a\rVert^2.
> $$
> Averaging $M$ independent directions divides this variance by $M$:
> $$
> \mathbb E\lVert\overline g-a\rVert^2
> =
> \frac{d+1}{M}\lVert a\rVert^2.
> $$
> Even in this ideal linear example, maintaining comparable relative accuracy as $d$ grows requires the number of directions to grow with dimension.

If ordinary neural-network training supports stable backpropagation, replacing it with ES is usually less sample-efficient. ES is valuable because it broadens the class of optimizable systems, not because it dominates reverse-mode differentiation.

### Real evaluations may contain another source of randomness

Suppose the observed score is

$$
\widetilde J(\theta)
=
J(\theta)+\xi,
$$

where $\xi$ comes from a mini-batch, data augmentation, a stochastic environment, random initial state, or measurement noise. A paired difference becomes

$$
\widetilde J(\theta+\sigma\varepsilon)
-
\widetilde J(\theta-\sigma\varepsilon)
=
J_+-J_-
+
\xi_+-\xi_-.
$$

If the two probes use unrelated randomness, $\xi_+-\xi_-$ may overwhelm the parameter effect.

A common variance-reduction method is to make the positive and negative probes share as much randomness as possible:

$$
\boxed{
\text{same mini-batch, same random seed, same environment initial state.}
}
$$

This is the **common-random-numbers** principle. It does not remove all noise, but it makes the comparison more likely to isolate the consequence of changing the parameters.

### One ES iteration, with responsibilities separated

For minimization, an antithetic ES iteration can be written as:

1. sample $\varepsilon_i\overset{\mathrm{iid}}{\sim}\mathcal N(0,I)$ for $i=1,\ldots,M$;
2. evaluate
   $$
   J_i^+=J(\theta_k+\sigma\varepsilon_i),
   \qquad
   J_i^-=J(\theta_k-\sigma\varepsilon_i);
   $$
3. estimate
   $$
   \widehat g_k
   =
   \frac1{2M\sigma}
   \sum_{i=1}^{M}(J_i^+-J_i^-)\varepsilon_i;
   $$
4. update
   $$
   \theta_{k+1}
   =
   \theta_k-\eta\widehat g_k.
   $$

The roles are distinct:

| Component | Responsibility |
|---|---|
| objective $J$ | assigns a scalar score to a complete run |
| perturbation distribution | defines the neighborhood and smoothed objective |
| Monte Carlo estimator | converts function-value comparisons into a noisy direction |
| optimizer step $\eta$ | decides how far to move along that direction |

This is analogous to the earlier loss/backprop/optimizer separation, except ES replaces backpropagation with a zeroth-order direction estimator.

### Durable summary

The defining smoothed objective and gradient identity are

$$
\boxed{
J_\sigma(\theta)
=
\mathbb E[J(\theta+\sigma\varepsilon)],
\qquad
\nabla J_\sigma(\theta)
=
\frac1\sigma
\mathbb E[J(\theta+\sigma\varepsilon)\varepsilon].
}
$$

The most useful finite-sample estimator is often the antithetic form

$$
\boxed{
\widehat g_{\mathrm{anti}}
=
\frac1{2M\sigma}
\sum_{i=1}^{M}
\left[
J(\theta+\sigma\varepsilon_i)
-
J(\theta-\sigma\varepsilon_i)
\right]
\varepsilon_i.
}
$$

The conceptual chain is

$$
\boxed{
\text{function values only}
\to
\text{random parameter probes}
\to
\text{Gaussian-smoothed objective}
\to
\text{Monte Carlo direction estimate}
\to
\text{parameter update}.
}
$$

> [!summary] The central boundary
> ES does not recover information for free. It replaces inaccessible local derivatives with many noisy function evaluations and optimizes a scale-dependent smoothed objective. Its strength is generality and parallelism; its cost is variance, function evaluations, and loss of computation-graph structure.

The next intervention addresses a different failure mode. Gradient clipping assumes a gradient is already available and asks how to prevent an excessively large estimate from producing a destructive update.

## Connections

- The gradient-signal problems that motivate this derivative-free alternative are classified in [Stochastic Gradient Steps and Trainable Loss Geometry]({{ '/notes/mit6-7960-02-2-stochastic-gradient-steps-and-trainable-loss-geometry/' | relative_url }}).
