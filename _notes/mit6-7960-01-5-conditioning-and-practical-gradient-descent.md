---
layout: note
title: "Conditioning and Practical Gradient Descent"
source_title: "MIT6.7960-01.5-Conditioning and Practical Gradient Descent"
course: "MIT 6.7960"
sequence: "01.5"
source_context: "The Gradient, Steepest Descent, and the Guarantees of Gradient Descent"
permalink: "/notes/mit6-7960-01-5-conditioning-and-practical-gradient-descent/"
tags:
  - "math/optimization"
  - "deep-learning/practice"
  - "topic/conditioning"
---
A quadratic with unequal curvatures exposes what the abstract constants mean along an actual trajectory. This note solves that conditioned dynamics direction by direction, records the most important traps about stability and metric-dependent steepest descent, and then maps the deterministic theory to practical neural-network training. Keeping these parts together makes the transition explicit: conditioning shows why one scalar learning rate can serve different directions poorly, the trap sheet prevents overclaiming what the theory proves, and the practice section identifies which mechanisms survive when gradients are stochastic and the model is only piecewise smooth.

## 1.12 Worked example: conditioning in one picture

This quadratic makes the constants \\(\mu\\), \\(L\\), and \\(\kappa\\) visible one direction at a time:

\\[
f(x)
=\frac12\big(x_1^2+10x_2^2\big)
=\frac12x^\top
\underbrace{\begin{pmatrix}1&0\\\\ 0&10\end{pmatrix}}_{A}
x.
\\]

> [!summary] Geometry and constants
>
> \\[
> \nabla f(x)=Ax=(x_1,10x_2),
> \qquad
> \nabla^2f=A.
> \\]
>
> | Direction | Eigenvalue / curvature | Geometry |
> |---|---:|---|
> | \\(e_1\\) | \\(\lambda_1=1\\) | Flat direction; long ellipse axis |
> | \\(e_2\\) | \\(\lambda_2=10\\) | Steep direction; short ellipse axis |
>
> Hence
>
> \\[
> \mu=1,
> \qquad
> L=10,
> \qquad
> \kappa=\frac{L}{\mu}=10.
> \\]
>
> Here \\(\mu\\) is simultaneously the smallest Hessian eigenvalue, the strong-convexity constant, and an admissible PL constant. That coincidence is special to this positive-definite quadratic.

### The ellipse and the spectrum encode the same anisotropy

The level set \\(f(x)=c\\) satisfies

\\[
\frac{x_1^2}{2c}
+
\frac{x_2^2}{2c/10}
=1.
\\]

Its semi-axes are \\(\sqrt{2c}\\) and \\(\sqrt{2c/10}\\), so

\\[
\boxed{
\frac{\text{long axis}}{\text{short axis}}
=\sqrt{\frac{L}{\mu}}
=\sqrt\kappa
=\sqrt{10}
}.
\\]

The curvature ratio is \\(\kappa\\); the geometric axis ratio is \\(\sqrt\kappa\\). A large condition number produces a long, narrow valley.

### Diagonalize the dynamics: one scalar recurrence per direction

For a general step size \\(\eta\\),

\\[
x_{k+1}
=x_k-\eta Ax_k
=
\begin{pmatrix}
1-\eta&0\\
0&1-10\eta
\end{pmatrix}x_k.
\\]

Therefore

\\[
\boxed{
x_{1,k}=(1-\eta)^k x_{1,0},
\qquad
x_{2,k}=(1-10\eta)^k x_{2,0}
}.
\\]

For a quadratic with eigenvalue \\(\lambda_i\\), the corresponding coordinate multiplier is

\\[
r_i(\eta):=1-\eta\lambda_i.
\\]

| Multiplier | Directional behavior |
|---:|---|
| \\(0<r_i<1\\) | Contracts without crossing the minimum |
| \\(r_i=0\\) | Eliminated in one step |
| \\(-1<r_i<0\\) | Crosses the minimum and contracts: zig-zag |
| \\(r_i=-1\\) | Equal-amplitude oscillation |
| \\(\lvert r_i\rvert>1\\) | Diverges |

Both coordinates converge exactly when

\\[
|1-\eta\lambda_i|<1
\quad\text{for every }i,
\\]

which reduces to

\\[
\boxed{0<\eta<\frac2L=0.2.}
\\]

The largest curvature sets the global stability limit even though the flat coordinate alone could tolerate a much larger step.

### Main run: \\(\eta=1/L=0.1\\)

The two multipliers are

\\[
r_{\mathrm{flat}}=1-\eta\mu=0.9,
\qquad
r_{\mathrm{steep}}=1-\eta L=0.
\\]

Thus

\\[
x_{1,k}=0.9^k x_{1,0},
\qquad
x_{2,k}=0\quad(k\ge1).
\\]

The steep coordinate disappears in one step; all long-run progress is controlled by the flat coordinate. For example, from \\(x_0=(1,1)\\),

\\[
f(x_0)=5.5,
\qquad
x_1=(0.9,0),
\qquad
f(x_1)=0.405.
\\]

The first step removes the expensive steep-direction error, after which the iteration crawls along the valley floor.

> [!important] Parameter, gradient, and loss rates are different objects
> For \\(k\ge1\\),
>
> \\[
> \|x_k-x^\star\|_2
> =0.9^k|x_{1,0}|,
> \qquad
> \|\nabla f(x_k)\|_2
> =0.9^k|x_{1,0}|,
> \\]
>
> while
>
> \\[
> f(x_k)-f^\star
> =\frac12(0.81)^k x_{1,0}^2.
> \\]
>
> | Quantity | Exact tail rate |
> |---|---:|
> | Parameter error | \\(0.9^k\\) |
> | Gradient norm | \\(0.9^k\\) |
> | Function-value error | \\(0.81^k=(0.9^2)^k\\) |
>
> The PL theorem of §1.11 guarantees only
>
> \\[
> f(x_k)-f^\star
> \le
> 0.9^k\big(f(x_0)-f^\star\big),
> \\]
>
> a correct but loose upper bound. A guarantee need not equal the exact trajectory.

### Move near the boundary: \\(\eta=1.9/L=0.19\\)

Now

\\[
r_{\mathrm{flat}}=1-0.19=0.81,
\qquad
r_{\mathrm{steep}}=1-1.9=-0.9.
\\]

The steep coordinate alternates sign, creating zig-zag across the narrow valley. Moreover,

\\[
|r_{\mathrm{steep}}|=0.9
>
|r_{\mathrm{flat}}|=0.81,
\\]

so **the steep direction is now the slower mode**. The largest curvature always limits stability, but the flat direction is not the bottleneck for every step size.

![conditioning-step-size-comparison]({{ '/assets/notes/gradient-descent/conditioning-step-size-comparison.png' | relative_url }})
*The landscape is identical in both panels. At \\(\eta=1/L\\), the steep mode is eliminated and the flat mode controls the tail. Near \\(2/L\\), the steep mode changes sign, zig-zags, and approaches instability; it then contracts more slowly than the flat mode.*

> [!tip] The bottleneck is the multiplier closest to the unit circle
> For this quadratic,
>
> \\[
> \rho(\eta)
> :=\max\{|1-\eta\mu|,\ |1-\eta L|\}.
> \\]
>
> The slowest mode is whichever endpoint multiplier has the largest magnitude—not automatically the direction with the smallest eigenvalue.

### Best fixed step for the positive-definite quadratic

The worst directional multiplier is minimized by balancing the two endpoints:

\\[
1-\eta\mu
=
-(1-\eta L).
\\]

Hence

\\[
\boxed{
\eta_{\mathrm{opt}}
=\frac2{L+\mu}
},
\\]

and the optimal worst-mode contraction is

\\[
\boxed{
\rho_{\mathrm{opt}}
=\frac{L-\mu}{L+\mu}
=\frac{\kappa-1}{\kappa+1}
}.
\\]

Here,

\\[
\eta_{\mathrm{opt}}=\frac2{11},
\qquad
r_{\mathrm{flat}}=\frac9{11},
\qquad
r_{\mathrm{steep}}=-\frac9{11}.
\\]

The flat direction contracts without changing sign; the steep direction zig-zags with the same magnitude. This is the spectral optimum previewed in §1.9. It is specific to a positive-definite quadratic and the worst-direction multiplier—not a universal best learning rate.

### What conditioning really means

With the conservative step \\(\eta=1/L\\), the flat-mode multiplier is

\\[
1-\frac{\mu}{L}
=1-\frac1\kappa.
\\]

When \\(\kappa\\) is large, this number is close to \\(1\\): the steepest direction dictates a small stable step, while that same step barely moves the flattest direction. More generally, a single scalar learning rate must serve the entire curvature spectrum, and the mode whose multiplier lies closest to \\(\pm1\\) controls the tail.

> [!note]- Ideal preconditioning rounds the valley
> For \\(f(x)=\tfrac12x^\top Ax\\), the preconditioned update
>
> \\[
> x_{k+1}
> =x_k-\eta A^{-1}\nabla f(x_k)
> =(1-\eta)x_k
> \\]
>
> gives every eigendirection the same multiplier. Equivalently, the coordinate change \\(z=A^{1/2}x\\) turns the ellipses into circles. With the exact inverse and \\(\eta=1\\), this quadratic reaches the minimizer in one step—the Newton step. Practical preconditioners approximate this rescaling without forming a full Hessian inverse.

> [!note]- Real training: conditioning is local, changing, and noisy
> A neural-network Hessian changes along the trajectory and can have negative or near-zero eigenvalues. Consequently, \\(L\\), \\(\mu\\), and \\(\kappa\\) are usually local or effective quantities; for a general PL function, \\(\mu\\) is a gradient-dominance constant, not necessarily the smallest Hessian eigenvalue.
>
> | Mechanism | Connection to this quadratic | Important boundary |
> |---|---|---|
> | Momentum | Can accelerate slow modes and reduce valley zig-zag | Changes the stability polynomial; plain-GD multipliers no longer apply |
> | Adam/RMSProp | Dynamic diagonal rescaling resembles rough preconditioning | Does not equal the inverse Hessian or guarantee \\(\kappa=1\\) |
> | Whitening | Directly rescales correlated feature directions | Applies to a chosen representation, not every network curvature direction |
> | Batch/layer normalization | Can reshape effective optimization geometry | Is data- and architecture-dependent, not literal Hessian whitening |
> | Learning-rate decay | Reduces oscillation and stochastic noise late in training | Does not repair poor conditioning by itself |
>
> Overparameterized models often have many nearly flat parameter directions, so a global positive Hessian lower bound may fail even when the training loss behaves PL-like on the visited region. Optimization conditioning also says nothing by itself about validation performance or generalization.

---

## 1.13 Trap sheet

The chapter's clean formulas are easy to over-read. This sheet records the exact replacement for each tempting shortcut.

> [!summary] Six traps worth remembering
>
> | Tempting statement | Accurate replacement | Review |
> |---|---|---:|
> | “If GD does not converge, it diverges to infinity.” | At the stability boundary it may remain bounded and oscillate with constant amplitude. | §1.9, §1.12 |
> | “All partial derivatives exist, so the function is differentiable.” | Partials test coordinate lines; differentiability requires one uniform linear model in every direction. | §1.1–§1.3 |
> | “The negative gradient is intrinsically the steepest direction.” | It is steepest only relative to the chosen Euclidean metric. | §1.5 |
> | “Any negative subgradient is a steepest descent direction.” | For a convex non-smooth function, steepest descent is governed by the minimum-norm subgradient—not an arbitrary one. | below |
> | “A ReLU network only needs convex subgradient theory.” | The parameter loss is generally non-convex and non-smooth; convex subdifferentials are insufficient. | below |
> | “When a theorem gives no guarantee, the algorithm must fail.” | Silence of a theorem is not a theorem of failure; a counterexample is needed to prove a boundary is tight. | §1.9 |

### Stability: convergence, bounded oscillation, and divergence

For the positive-definite quadratic

\\[
f(x)=\frac12x^\top Ax,
\qquad
x_{k+1}=(I-\eta A)x_k,
\\]

the coordinate along eigenvalue \\(\lambda_i\\) obeys

\\[
[x_k]_i=(1-\eta\lambda_i)^k[x_0]_i.
\\]

> [!important] Three regimes at the largest eigenvalue \\(L\\)
>
> | Step size | Top-mode multiplier | Worst-case behavior |
> |---|---:|---|
> | \\(0<\eta<2/L\\) | \\(\lvert1-\eta L\rvert<1\\) | Converges for every initial point |
> | \\(\eta=2/L\\) | \\(-1\\) | Bounded equal-amplitude oscillation if the top-mode component is nonzero |
> | \\(\eta>2/L\\) | \\(\lvert1-\eta L\rvert>1\\) | Unbounded divergence if the top-mode component is nonzero |

The phrase **for every initial point** matters. A specially aligned \\(x_0\\) with zero component in every unstable eigendirection can tolerate a larger step. The threshold \\(2/L\\) is the uniform worst-case boundary, not a claim about every individual trajectory.

### “Steepest” always names a metric

In the Euclidean norm,

\\[
\operatorname*{arg\,min}_{\|u\|_2=1}\langle\nabla f(x),u\rangle
=-\frac{\nabla f(x)}{\|\nabla f(x)\|_2}.
\\]

Under the quadratic norm \\(\|u\|_A=\sqrt{u^\top Au}\\), the corresponding unit direction is

\\[
\boxed{
u_A^\star
=-\frac{A^{-1}\nabla f(x)}
{\sqrt{\nabla f(x)^\top A^{-1}\nabla f(x)}}
}.
\\]

The unnormalized ray \\(-A^{-1}\nabla f(x)\\) is the seed of preconditioning, Newton's method, and natural-gradient geometry. Reparameterizing a model can therefore change an ordinary Euclidean gradient step even when the represented predictor is unchanged.

> [!note]- Convex non-smooth boundary: subgradients
> For a convex function, \\(g\\) is a subgradient at \\(x\\) when
>
> \\[
> f(y)\ge f(x)+\langle g,y-x\rangle
> \qquad\text{for every }y,
> \\]
>
> and \\(\partial f(x)\\) denotes the set of all such \\(g\\). At a smooth point, \\(\partial f(x)=\{\nabla f(x)\}\\); at a kink it may contain many vectors.
>
> For example,
>
> \\[
> \partial|x|
> =
> \begin{cases}
> \{-1\},&x<0,\\
> [-1,1],&x=0,\\
> \{1\},&x>0.
> \end{cases}
> \\]
>
> At the minimizer \\(x=0\\), choosing the valid subgradient \\(g=1\\) and moving along \\(-g\\) would increase \\(|x|\\). The element nearest the origin,
>
> \\[
> g_{\min}
> \in\operatorname*{arg\,min}_{g\in\partial f(x)}\|g\|_2,
> \\]
>
> governs Euclidean steepest descent; here \\(g_{\min}=0\\). A subgradient method also need not decrease the objective on every step and usually uses a diminishing step schedule. The smooth Descent Lemma cannot be recovered by simply replacing \\(\nabla f\\) with an arbitrary subgradient.

> [!note]- ReLU boundary: piecewise smooth, non-convex, and non-smooth
> The scalar ReLU is convex and non-differentiable at zero, but a deep network's loss as a function of its parameters is generally both non-convex and non-smooth. The convex subdifferential above therefore does not describe the full training objective; non-convex generalized derivatives such as the Clarke gradient are needed for a rigorous treatment.
>
> At a ReLU kink, no classical derivative exists for automatic differentiation to recover. A framework applies a chosen convention—commonly a derivative value of zero at the origin—and backpropagates through that executed graph. Away from activation-boundary crossings the network is piecewise smooth, so this chapter remains a useful baseline; it is not a literal smoothness theorem across every kink.

---

## 1.14 From gradient descent to practical deep learning

The mathematical baseline in this chapter is exact, deterministic, full-batch gradient descent on a smooth objective. Real training changes the gradient source, update dynamics, step schedule, regularity, and success criterion.

> [!summary] Theory-to-practice map
>
> | Idealized object | Practical counterpart | What changes |
> |---|---|---|
> | Exact \\(\nabla F(\theta)\\) | Backpropagated mini-batch gradient | Adds sampling and model randomness |
> | Full-batch GD | SGD, momentum, Adam | Updates become stochastic and stateful |
> | Fixed \\(\eta\\) with known \\(L\\) | Search, warmup, decay, schedules | Curvature is unknown and changes along the path |
> | Globally smooth objective | ReLU and other piecewise-smooth models | Classical derivatives fail at kinks |
> | Training-objective progress | Validation and test performance | Optimization does not imply generalization |

### Backpropagation computes; the optimizer updates

For a batch loss \\(F_B(\theta)\\), reverse-mode automatic differentiation computes

\\[
\nabla_\theta F_B(\theta).
\\]

The optimizer then decides how to use that vector.

| Component | Question answered |
|---|---|
| Forward pass | What is the current prediction and loss? |
| Backpropagation | What is the gradient of that scalar loss with respect to every parameter? |
| Optimizer | How should the parameters change given current and past gradients? |
| Learning-rate schedule | How should the overall update scale change over time? |

> [!important] Reverse mode avoids a factor of the parameter dimension
> For \\(F:\mathbb R^n\to\mathbb R\\), one reverse sweep shares chain-rule computations across all \\(n\\) parameters. Its work is typically a small constant multiple of the executed forward computation—the same asymptotic order, not one forward pass per parameter.
>
> Centered finite differences would require approximately
>
> \\[
> \frac{F(\theta+\varepsilon e_i)-F(\theta-\varepsilon e_i)}
> {2\varepsilon}
> \qquad(i=1,\ldots,n),
> \\]
>
> or about \\(2n\\) function evaluations. That is why finite differences are a debugging check on small cases, not a training method for a large network.

Reverse mode is not free or size-independent: larger computation graphs require more forward and backward work. The backward pass also needs intermediate activations from the forward pass. **Gradient checkpointing** stores only selected activations and recomputes the rest during backward:

\\[
\boxed{\text{less activation memory}\quad\Longleftrightarrow\quad\text{more computation}.}
\\]

### Full-batch GD becomes mini-batch stochastic optimization

For an empirical objective

\\[
F(\theta)=\frac1m\sum_{i=1}^m\ell_i(\theta),
\\]

the exact gradient is

\\[
\nabla F(\theta)
=\frac1m\sum_{i=1}^m\nabla\ell_i(\theta).
\\]

A random mini-batch \\(B_k\\) supplies the estimator

\\[
g_{B_k}(\theta)
=\frac1{|B_k|}\sum_{i\in B_k}\nabla\ell_i(\theta).
\\]

Under uniform sampling and the usual conditioning assumptions,

\\[
\mathbb E[g_{B_k}(\theta)\mid\theta]
=\nabla F(\theta),
\\]

but an individual estimate is noisy. Consequently, the deterministic monotone-descent statement from §1.9 does not carry over step by step; §§1.9–§1.11 showed the corresponding expectation bounds and noise floors.

> [!warning] Consecutive logged batch losses may not evaluate the same function
> A log may compare
>
> \\[
> F_{B_k}(\theta_k)
> \qquad\text{with}\qquad
> F_{B_{k+1}}(\theta_{k+1}),
> \\]
>
> where both the parameters and the sampled data changed. Their difference is not the full-objective decrease \\(F(\theta_k)-F(\theta_{k+1})\\). A noisy batch-loss curve can therefore coexist with a healthy long-run optimization trend. Fixed evaluation data, smoothed metrics, epoch aggregates, and validation checkpoints answer different questions.

> [!note]- What batch size changes
>
> | Smaller batch | Larger batch |
> |---|---|
> | Cheaper individual update | More work per update |
> | Higher gradient variance | Lower variance; closer to full gradient |
> | Noisier loss trace | More stable loss trace |
> | Often needs more conservative late-stage steps | Can often support a larger stable step |
>
> Under ideal independent sampling, variance often scales approximately as \\(1/|B|\\), but correlation, augmentation, sampling strategy, and model state can alter that rule. Gradient accumulation combines several micro-batches before one optimizer update, trading more computation for the effect of a larger batch when activation memory is limited.

### The learning rate is tuned because \\(L\\) is unknown and non-stationary

The theoretical numbers

\\[
\eta=\frac1L,
\qquad
0<\eta<\frac2L,
\\]

are structural principles, not usually observable hyperparameters. A global \\(L\\) is rarely known for a network, and the local effective curvature changes along the trajectory.

| Practice | Role | Not a theorem that… |
|---|---|---|
| Learning-rate search | Finds a usable scale for the model, batch, and optimizer | The selected value equals \\(1/L\\) |
| Warmup | Stabilizes early updates while activations and optimizer state settle | One fixed global \\(L\\) derived the schedule |
| Decay / cosine schedule | Lowers oscillation and stochastic noise late in training | Smaller is always better from the first step |
| Backtracking line search | Tests local decrease by additional evaluations | It is always economical under noisy distributed training |
| Gradient clipping | Caps the applied update | The original objective became smooth |

Warmup and decay serve different phases: a larger middle-stage step makes rapid progress, while a smaller late-stage step reduces the noise floor and permits finer convergence. In deterministic optimization, line search can discover a local safe step; in large stochastic training, repeated trial evaluations and optimizer state make preset schedules more common.

> [!note]- Momentum and Adam change the dynamical system
> Momentum introduces state, for example
>
> \\[
> v_{k+1}=\beta v_k+g_k,
> \qquad
> \theta_{k+1}=\theta_k-\eta v_{k+1}.
> \\]
>
> Adam additionally rescales coordinates using first- and second-moment estimates. This resembles dynamic diagonal preconditioning, but the second moment is not the Hessian inverse, and the base learning rate still matters.
>
> | Method | Potential benefit | Analysis boundary |
> |---|---|---|
> | Momentum | Accumulates progress in persistent directions; can reduce valley zig-zag | Plain-GD stability multipliers no longer apply |
> | Adam/RMSProp | Adapts to coordinate-scale differences | Does not automatically solve conditioning or guarantee convergence under every setting |
> | Whitening / preconditioning | More directly reshapes selected curvature directions | Exact Hessian geometry is rarely available |
>
> These methods extend the gradient-descent baseline; they are not counterexamples to it.

### Smooth theory remains a baseline for piecewise-smooth networks

ReLU networks are differentiable away from activation-boundary crossings, and backpropagation uses a framework convention at the kinks. The \\(L\\)-smooth theorems in this chapter therefore describe an idealized local or smoothed model of training. They remain valuable because they isolate the mechanisms—descent, stability, noise, and conditioning—that more specialized non-smooth or stochastic analyses must replace.

### Optimization success is not generalization

Everything proved above concerns the empirical training objective \\(F(\theta)\\). The population risk is a different object:

\\[
R(\theta)
=\mathbb E_{(a,b)\sim\mathcal D}
\big[\ell(h_\theta(a),b)\big].
\\]

> [!warning] The final boundary
>
> \\[
> F(\theta_k)\downarrow
> \quad\not\Longrightarrow\quad
> R(\theta_k)\downarrow.
> \\]
>
> A model can drive training loss near zero and still generalize poorly. Validation performance depends on data coverage, inductive bias, regularization, augmentation, early stopping, optimizer implicit bias, and distribution shift. Optimization theory explains whether the chosen training objective is being solved; statistical learning theory asks whether solving it transfers to unseen data.

### The chapter in one table

| Added structure or practical change | What becomes possible |
|---|---|
| Differentiability | One linear model represents every first-order directional change |
| Euclidean metric | The negative gradient is the steepest unit direction |
| \\(L\\)-smoothness | Explicit per-step descent and a stability window |
| A finite lower bound | A finite descent budget forces gradients toward zero |
| PL | Stationarity certifies global function-value progress at a geometric rate |
| Quadratic spectrum | Conditioning, zig-zag, and preconditioning become directionally explicit |
| Mini-batch noise | Guarantees move from every step to expectation and acquire a noise floor |
| Real networks and evaluation | Smooth optimization becomes a baseline, and training progress separates from generalization |

\\[
\boxed{
\text{local derivative}
\;\longrightarrow\;
\text{safe update}
\;\longrightarrow\;
\text{stationarity}
\;\longrightarrow\;
\text{global progress under extra structure}
}
\\]

## Connections

- The stability and global-progress theorems interpreted by the conditioned example are developed in [Stability, Stationarity, and Global Progress]({{ '/notes/mit6-7960-01-4-stability-stationarity-and-global-progress/' | relative_url }}).
- The practical theory-to-training map is instantiated by the stochastic updates and loss signals in [Stochastic Gradient Steps and Trainable Loss Geometry]({{ '/notes/mit6-7960-02-2-stochastic-gradient-steps-and-trainable-loss-geometry/' | relative_url }}).
