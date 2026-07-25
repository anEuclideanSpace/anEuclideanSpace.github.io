---
layout: note
title: "Stochastic Gradient Steps and Trainable Loss Geometry"
source_title: "MIT6.7960-02.2-Stochastic Gradient Steps and Trainable Loss Geometry"
course: "MIT 6.7960"
sequence: "02.2"
source_context: "How to Train a Neural Net — Backpropagation and Differentiable Programming"
permalink: "/notes/mit6-7960-02-2-stochastic-gradient-steps-and-trainable-loss-geometry/"
tags:
  - "deep-learning/foundations"
  - "topic/stochastic-gradient-descent"
  - "topic/loss-geometry"
  - "topic/cross-entropy"
---
A mini-batch gradient becomes an optimization method only when it is inserted into a complete parameter-update cycle, and that cycle is useful only when the chosen loss supplies an informative derivative. This note therefore follows one stochastic-gradient step in full, derives its one-step conditional expectation, classifies loss geometries that make gradients misleading or unusable, and develops cross-entropy as a trainable classification signal through probabilities, likelihood, softmax, and the output gradient p minus y. The progression is intentional: stochastic dynamics explains how a signal is used, loss geometry explains how that signal can fail, and cross-entropy provides a concrete example of designing a better signal.

### From a batch gradient to stochastic gradient descent

The intended training objective remains the full empirical mean

$$
J_{\mathcal D}(\theta)
=
\frac1N
\sum_{i=1}^{N}
\ell_i(\theta),
$$

with gradient

$$
\nabla J_{\mathcal D}(\theta)
=
\frac1N
\sum_{i=1}^{N}
\nabla\ell_i(\theta).
$$

Full-batch gradient descent waits for all $N$ per-example gradients before moving:

$$
\theta_{k+1}
=
\theta_k-
\eta\nabla J_{\mathcal D}(\theta_k).
$$

This is the deterministic update studied in [Gradient Descent, Smoothness, and the Descent Lemma]({{ '/notes/mit6-7960-01-3-gradient-descent-smoothness-and-the-descent-lemma/' | relative_url }}). Its direction represents every training example at the current parameter vector, but one parameter update costs one complete pass over the dataset.

> [!note] The memory bottleneck, stated precisely
> A full gradient does not logically require loading the entire dataset into accelerator memory at once: smaller chunks can be processed and their gradients accumulated. The unavoidable cost is that the optimizer still waits until all $N$ examples have been processed before applying that full-gradient update. Mini-batches reduce the amount of data processed **per parameter update** and keep activation memory within a workable range.

Stochastic gradient descent instead selects a random batch $B_k$ and moves immediately using its average gradient:

$$
\boxed{
\theta_{k+1}
=
\theta_k-
\eta g_{B_k}(\theta_k),
\qquad
g_{B_k}(\theta_k)
=
\frac1b
\sum_{i\in B_k}
\nabla\ell_i(\theta_k).
}
$$

One actual training step is therefore

$$
\boxed{
B_k
\longrightarrow
\text{forward pass}
\longrightarrow
J_{B_k}(\theta_k)
\longrightarrow
\text{backpropagation}
\longrightarrow
g_{B_k}(\theta_k)
\longrightarrow
\theta_{k+1}.
}
$$

| Stage | What is computed? | What is held fixed? |
|---|---|---|
| Choose $B_k$ | the examples used by this step | current parameters $\theta_k$ |
| Forward pass | predictions and per-example losses | $\theta_k$ and $B_k$ |
| Batch reduction | $J_{B_k}=b^{-1}\sum_{i\in B_k}\ell_i$ | same $\theta_k$ |
| Backward pass | $g_{B_k}=\nabla J_{B_k}$ | same computation graph values |
| Update | $\theta_{k+1}=\theta_k-\eta g_{B_k}$ | the chosen gradient |

After the update, the temporary activations for $B_k$ can be released. The next batch is evaluated at the **new** parameter vector. Thus one epoch produces gradients such as

$$
g_{B_1}(\theta_0),
\qquad
g_{B_2}(\theta_1),
\qquad
g_{B_3}(\theta_2),
\quad\ldots
$$

rather than several pieces of one fixed gradient $\nabla J_{\mathcal D}(\theta_0)$. Even if the batches partition the dataset, their gradients cannot generally be averaged afterward to recover the full gradient at one common parameter vector, because the parameter vector moved between batches.

#### A complete one-parameter example

Consider the scalar model

$$
f_\theta(x)=\theta
$$

and two targets

$$
y_1=-3,
\qquad
y_2=1.
$$

Use the squared per-example loss

$$
\ell_i(\theta)
=
\frac12(\theta-y_i)^2.
$$

The two examples define

$$
\ell_1(\theta)
=
\frac12(\theta+3)^2,
\qquad
\ell_2(\theta)
=
\frac12(\theta-1)^2.
$$

Example 1 is minimized at $\theta=-3$, whereas example 2 is minimized at $\theta=1$. The full objective must compromise:

$$
J_{\mathcal D}(\theta)
=
\frac12\big(\ell_1(\theta)+\ell_2(\theta)\big)
=
\frac14\left[(\theta+3)^2+(\theta-1)^2\right].
$$

Its gradient is

$$
\begin{aligned}
\nabla J_{\mathcal D}(\theta)
&=
\frac12\left[(\theta+3)+(\theta-1)\right]\\
&=
\theta+1,
\end{aligned}
$$

so the full-data minimizer is

$$
\theta^*=-1.
$$

Start from $\theta_0=0$ with $\eta=0.1$. The full gradient is

$$
\nabla J_{\mathcal D}(0)=1,
$$

and the deterministic full-batch step is

$$
\theta_1^{\mathrm{full}}
=
0-0.1(1)
=
-0.1.
$$

With batch size $b=1$, two stochastic outcomes are possible:

| Selected example | Gradient at $\theta_0=0$ | SGD update | New parameter |
|---|---:|---:|---:|
| $y_1=-3$ | $g_1(0)=0-(-3)=3$ | $0-0.1(3)$ | $-0.3$ |
| $y_2=1$ | $g_2(0)=0-1=-1$ | $0-0.1(-1)$ | $+0.1$ |

![one-stochastic-gradient-step]({{ '/assets/notes/neural-network-training/one-stochastic-gradient-step.png' | relative_url }})

The second stochastic outcome moves to the right even though the full-data minimizer lies to the left. This is not an implementation error. That step correctly descends the selected sample loss:

$$
\ell_2(0)=0.500,
\qquad
\ell_2(0.1)=0.405,
$$

but the same move increases the full objective:

$$
J_{\mathcal D}(0)=2.500,
\qquad
J_{\mathcal D}(0.1)=2.605.
$$

Therefore,

$$
\boxed{
J_{B_k}(\theta_{k+1})<J_{B_k}(\theta_k)
\centernot\implies
J_{\mathcal D}(\theta_{k+1})<J_{\mathcal D}(\theta_k).
}
$$

> [!important] What one SGD step optimizes
> The update is calculated from the current batch objective, not by directly inspecting the full objective. A batch may disagree with the full dataset, so an individual step can increase full-dataset loss even while it decreases the selected batch loss.

#### Why the one-step average still matches full-batch descent

If the two examples are sampled uniformly, their gradients at $\theta_0=0$ average to

$$
\mathbb E[g_B(0)]
=
\frac12(3)+\frac12(-1)
=
1
=
\nabla J_{\mathcal D}(0).
$$

The two possible new parameters also average to the full-batch result:

$$
\mathbb E[\theta_1\mid\theta_0=0]
=
\frac12(-0.3)+\frac12(0.1)
=
-0.1.
$$

In general, under fresh uniform batch sampling,

$$
\boxed{
\mathbb E
\left[
\theta_{k+1}
\mid
\theta_k
\right]
=
\theta_k-
\eta\nabla J_{\mathcal D}(\theta_k).
}
$$

The conditioning has an operational meaning:

1. freeze the current location $\theta_k$;
2. imagine repeating the next random batch draw many times;
3. average only those possible next-step outcomes.

This is a **local, one-step** statement. It does not yet say that the average of complete SGD trajectories equals the full-batch GD trajectory. After the first random update, different runs occupy different parameter vectors, and nonlinear gradients evaluate differently at those different locations.

#### Terminology

Historically, “stochastic gradient descent” often meant one example per update, $b=1$, while “mini-batch gradient descent” meant $1<b<N$. In modern deep-learning practice, **SGD usually refers to the same update with a mini-batch**, unless $b=1$ is stated explicitly.

The durable distinction is

$$
\boxed{
\begin{aligned}
\text{full-batch GD:}&\quad
\text{expensive deterministic opinion from all examples},\\
\text{mini-batch SGD:}&\quad
\text{cheaper noisy opinion from one random subset}.
\end{aligned}
}
$$

SGD does not replace the intended full-data objective. It changes how frequently and how accurately that objective is queried during training.

### Durable summary

$$
\boxed{
\text{data and task}
\to
\text{loss}
\to
J_{\mathcal D}(\theta)
\to
\text{local derivative information}
\to
\text{parameter search}
}.
$$

---

## 2.3 When a gradient stops being useful

The existence of a derivative is not the same as the usefulness of that derivative. The source lecture places several geometries side by side:

![gradient-pathologies-source]({{ '/assets/notes/neural-network-training/gradient-pathologies-source.png' | relative_url }})

*Source: MIT 6.7960 Fall 2024, Lecture 2, slide 14. The highlighted cases are the lecture's difficult examples.*

Three separate questions must be asked:

| Level | Question | What a “yes” actually means |
|---|---|---|
| mathematical | Does a classical derivative exist? | there is a unique local linear model |
| software | Does autodiff return a value? | the executed primitives have backward rules or conventions |
| optimization | Is the returned signal useful? | its direction and scale support stable progress |

Therefore,

$$
\boxed{
\text{non-differentiable}
\not\Rightarrow
\text{hard to optimize},
\qquad
\text{differentiable}
\not\Rightarrow
\text{easy to optimize}
}.
$$

The figure below pairs each objective with the derivative visible to a first-order optimizer:

![loss-shapes-and-gradient-signals]({{ '/assets/notes/neural-network-training/loss-shapes-and-gradient-signals.png' | relative_url }})

### A well-scaled baseline

![ideal-gradient-source]({{ '/assets/notes/neural-network-training/ideal-gradient-source.png' | relative_url }})

*Source: MIT 6.7960 Fall 2024, Lecture 2, slide 15.*

The one-dimensional quadratic

$$
J(\theta)
=
\frac12(\theta-\theta^*)^2
$$

has derivative

$$
J'(\theta)=\theta-\theta^*.
$$

Gradient descent gives

$$
\theta_{k+1}
=
\theta_k-\eta(\theta_k-\theta^*).
$$

Writing $e_k=\theta_k-\theta^*$,

$$
e_{k+1}=(1-\eta)e_k.
$$

Thus $0<\eta<2$ contracts the error. The signal is strong far away, has the correct sign, and goes to zero only as the optimum is reached.

> [!warning] Two corrections to the source slide
> 1. **Convex does not imply a unique minimizer.** A constant function is convex and every point minimizes it. Uniqueness requires extra structure such as strict or strong convexity.
> 2. In several dimensions, $-\nabla J(\theta)$ need not point exactly along the straight line to a minimizer. For a differentiable convex objective,
>    $$
>    \nabla J(\theta)^\top(\theta-\theta^*)
>    \ge
>    J(\theta)-J(\theta^*)
>    \ge0,
>    $$
>    so the negative gradient has a component toward a minimizer, but anisotropic curvature can still create zig-zag trajectories. See [Conditioning and Practical Gradient Descent]({{ '/notes/mit6-7960-01-5-conditioning-and-practical-gradient-descent/#worked-example-conditioning-in-one-picture' | relative_url }}).

> [!important] Zero near a solution is not vanishing-gradient pathology
> At an unconstrained differentiable minimum, the gradient should be zero. A pathology occurs when the signal becomes negligible **before** a useful region has been reached, or becomes exactly zero in a poor region.

### An isolated kink: no unique derivative, yet useful information

Consider

$$
J(\theta)=\lvert\theta\rvert.
$$

For nonzero $\theta$,

$$
J'(\theta)
=
\begin{cases}
-1,&\theta<0,\\
1,&\theta>0.
\end{cases}
$$

At zero,

$$
J'_-(0)=-1,
\qquad
J'_+(0)=1,
$$

so the classical derivative does not exist. Nevertheless, every nonzero point receives a perfectly informative sign: move right from the left half-line and left from the right half-line.

For this convex function,

$$
\partial J(0)=[-1,1],
$$

and $0\in\partial J(0)$ is consistent with zero being a minimizer.

> [!important] What an autodiff value means at a kink
> Automatic differentiation applies registered backward rules to the executed primitives. A framework may choose one conventional value at a non-differentiable point. Receiving a numeric tensor therefore does **not** prove that the classical derivative exists; it proves only that the executed graph supplied a backward rule. This is the same boundary discussed in [Conditioning and Practical Gradient Descent]({{ '/notes/mit6-7960-01-5-conditioning-and-practical-gradient-descent/#trap-sheet' | relative_url }}).

An isolated kink is often tolerable because the surrounding regions still carry useful directions. A piecewise-linear objective may be hard for other reasons—poor local minima, flat pieces, or discontinuities—not merely because corners exist.

### A discontinuity: local slopes cannot report the jump

Consider the step objective

$$
J(\theta)
=
\begin{cases}
0,&\theta<0,\\
1,&\theta\ge0.
\end{cases}
$$

For every $\theta\ne0$,

$$
J'(\theta)=0,
$$

while at zero

$$
\lim_{\theta\to0^-}J(\theta)=0,
\qquad
\lim_{\theta\to0^+}J(\theta)=1.
$$

Starting from $\theta_0>0$, plain gradient descent obtains

$$
\theta_{k+1}
=
\theta_k-\eta\cdot0
=
\theta_k
$$

and never discovers the lower region on the other side of the boundary.

> [!warning] One-sided derivatives do not repair discontinuity
> Slopes on the two branches may guide motion within each branch, but no single local linear model predicts the jump across the boundary. Whether software can execute a backward rule is separate from whether that rule reveals the better region. The source slide's phrase “not a problem for PyTorch” should be read only as an implementation claim for a chosen graph—not as an optimization guarantee.

This explains why accuracy is normally an evaluation metric rather than the training objective. It is assembled from discrete decisions,

$$
\operatorname{accuracy}(\theta)
=
\frac1N\sum_{i=1}^N
\mathbf 1
\left[
\operatorname*{arg\,max}_c f_\theta(x_i)_c=y_i
\right],
$$

so it remains unchanged across large parameter regions and jumps only when a decision boundary is crossed. Cross-entropy supplies a continuous surrogate signal before the predicted class changes.

### Vanishing and zero gradients are different failure modes

#### Vanishing: correct direction, negligible magnitude

Let

$$
J_\varepsilon(\theta)
=
\frac\varepsilon2(\theta-\theta^*)^2,
\qquad
0<\varepsilon\ll1.
$$

Then

$$
J_\varepsilon'(\theta)
=
\varepsilon(\theta-\theta^*)
$$

and

$$
\theta_{k+1}-\theta^*
=
(1-\eta\varepsilon)(\theta_k-\theta^*).
$$

For $\varepsilon=10^{-6}$ and $\eta=1$,

$$
(1-10^{-6})^{1000}
\approx
e^{-0.001}
\approx
0.999,
$$

so one thousand steps remove almost none of the error.

With a mini-batch estimator

$$
g_k=\nabla J(\theta_k)+\xi_k,
$$

the direction becomes noise-dominated when

$$
\|\nabla J(\theta_k)\|
\ll
\sqrt{\mathbb E\|\xi_k\|^2}.
$$

The gradient may point correctly in expectation while its signal-to-noise ratio is too poor for reliable progress.

#### Zero: complete silence

If

$$
\nabla J(\theta_k)=0,
$$

then for every learning rate

$$
\theta_{k+1}=\theta_k.
$$

This does not prove that $\theta_k$ is a good solution. A zero-gradient point may be a local minimum, local maximum, saddle point, or part of a flat plateau.

| Signal | Mathematical state | First-order behavior |
|---|---|---|
| small but nonzero gradient | $0<\|\nabla J\|\ll1$ | moves slowly; noise may dominate |
| zero gradient | $\nabla J=0$ | plain gradient descent does not move |

> [!note]- Two mechanisms called “vanishing gradient”
> This section studies a flat objective region in which $\nabla J$ is already small. A deep network can also make gradients vanish during backward propagation because many Jacobians are multiplied:
> $$
> \frac{\partial J}{\partial x_0}
> =
> \frac{\partial J}{\partial x_L}
> \frac{\partial x_L}{\partial x_{L-1}}
> \cdots
> \frac{\partial x_1}{\partial x_0}.
> $$
> The symptom is similar, but the mechanism belongs to the later backpropagation sections and will be derived there rather than assumed here.

### Exploding gradient: approaching the solution enlarges the update

Let

$$
J(\theta)=\lvert\theta\rvert^p,
\qquad
0<p<1.
$$

For nonzero $\theta$,

$$
J'(\theta)
=
p\,\operatorname{sign}(\theta)
\lvert\theta\rvert^{p-1}.
$$

Since $p-1<0$,

$$
\lvert J'(\theta)\rvert
\to\infty
\qquad
\text{as}\qquad
\theta\to0.
$$

For $p=\tfrac12$,

$$
J(\theta)=\sqrt{\lvert\theta\rvert},
\qquad
J'(\theta)
=
\frac{\operatorname{sign}(\theta)}
{2\sqrt{\lvert\theta\rvert}}.
$$

The update becomes

$$
\theta_{k+1}
=
\theta_k
-
\eta
\frac{\operatorname{sign}(\theta_k)}
{2\sqrt{\lvert\theta_k\rvert}}.
$$

Near the minimizer, the step length

$$
\frac\eta{2\sqrt{\lvert\theta_k\rvert}}
$$

grows rather than shrinks, producing overshoot and instability.

> [!note]- A second exploding-gradient mechanism appears in deep graphs
> Products of Jacobians with persistent gain greater than one can make the backward signal grow approximately exponentially with depth. That is a different cause from the singular scalar example above, but it produces the same practical symptoms: huge norms, unstable updates, overflow, or NaNs. §2.5 introduces clipping as a guardrail; the later backpropagation section explains the Jacobian product itself.

### Multiple local minima: the gradient knows only the current basin

A point $\theta_{\mathrm{loc}}$ is a local minimizer if some $r>0$ satisfies

$$
J(\theta_{\mathrm{loc}})
\le
J(\theta)
\qquad
\text{whenever}
\qquad
\|\theta-\theta_{\mathrm{loc}}\|<r.
$$

A global minimizer must satisfy the inequality for every $\theta$. Consequently, a local minimum may obey

$$
J(\theta_{\mathrm{loc}})
>
J(\theta_{\mathrm{global}}).
$$

Once a descent method settles into a local basin, reaching a better basin may require first moving uphill—information and behavior that a purely local descent rule does not provide. Initialization therefore affects which basin of attraction is reached.

> [!warning] The one-dimensional “bad local minimum” picture is not the whole deep-learning story
> High-dimensional networks also contain saddle points, nearly flat directions, parameter symmetries, and connected low-loss regions. Different parameter vectors may represent the same function, and many minima may have comparable training loss. In overparameterized training, poor conditioning or flat/noisy directions can matter more than an isolated bad local minimum.

The practical target is usually not a unique global parameter vector, but a region that is stable to optimize, achieves sufficiently low training loss, and generalizes well.

### Failure modes as a communication channel

The gradient is a local message from the objective to the optimizer:

$$
\nabla J(\theta)
=
\text{“how infinitesimal parameter changes alter the loss.”}
$$

| Geometry | Message received by the optimizer |
|---|---|
| well-scaled smooth region | clear, stable local direction |
| isolated kink | no unique message at one point; informative signs around it |
| discontinuity | local slopes cannot report the jump |
| vanishing gradient | direction may be correct, but the message is too weak |
| zero-gradient plateau | complete silence |
| exploding gradient | the message is so large that the update becomes unstable |
| poor local minimum | accurate information about the current basin, none about a better distant basin |

> [!summary] What a trainable loss should provide
> A useful loss is not merely numerically meaningful. Across the region reached by training, it should provide task-aligned low values, informative directions before a good solution is reached, manageable gradient magnitudes, and sufficiently predictable local change. Everywhere differentiability is helpful but not mandatory; an autodiff return value is necessary for the chosen pipeline but not sufficient for useful optimization.

The simplest important intervention is to replace a discrete task metric with a differentiable surrogate. The next section expands the prototype example—cross-entropy—before returning to the broader intervention toolbox.

---

## 2.4 Cross-entropy: turning classification into a trainable signal

Cross-entropy is the standard example of a **surrogate objective**: the final task asks whether the predicted class is correct, but training needs a scalar that records partial progress before the predicted class changes.

For ordinary single-label classification, its operational meaning is:

$$
\boxed{
\mathcal L_{\mathrm{CE}}
=
-\log p_{\text{true class}}
}.
$$

The rest of this section explains where that expression comes from, why the logarithm is present, how it differs from accuracy, and why its derivative combines with softmax to produce the unusually clean error signal $p-y$.

### From a prediction to a probability distribution

Suppose an image must be assigned to one of three mutually exclusive classes,

$$
\{\text{cat},\text{dog},\text{bird}\},
$$

and the true answer is dog. The model might produce the probability vector

$$
p
=
\begin{bmatrix}
0.2\\
0.7\\
0.1
\end{bmatrix},
\qquad
p_c\ge0,
\qquad
\sum_c p_c=1.
$$

| Class | Model probability |
|---|---:|
| cat | $0.2$ |
| dog — true class | $0.7$ |
| bird | $0.1$ |

The model predicts dog because $0.7$ is the largest entry. That discrete answer does not fully describe what the model has learned. All three distributions below predict dog, but they represent very different states:

| Model | $p(\text{cat})$ | $p(\text{dog})$ | $p(\text{bird})$ | Interpretation |
|---|---:|---:|---:|---|
| A | $0.01$ | $0.98$ | $0.01$ | dog with high confidence |
| B | $0.30$ | $0.40$ | $0.30$ | dog, but uncertain |
| C | $0.39$ | $0.40$ | $0.21$ | dog by a very small margin |

Accuracy assigns all three the same value. A training objective must distinguish them.

### The correct-class probability becomes a loss

If the true class is $y$, define

$$
\boxed{
\mathcal L(p,y)
=
-\log p_y
}.
$$

For the example above,

$$
\mathcal L
=
-\log0.7
\approx
0.357.
$$

| Correct-class probability $p_y$ | Cross-entropy $-\log p_y$ | Interpretation |
|---:|---:|---|
| $1.00$ | $0$ | complete confidence in the truth |
| $0.90$ | $0.105$ | confident and correct |
| $0.70$ | $0.357$ | correct, but still improvable |
| $0.50$ | $0.693$ | uncertain |
| $0.10$ | $2.303$ | barely believes the truth |
| $0.01$ | $4.605$ | confidently wrong |
| $p_y\to0$ | $\mathcal L\to\infty$ | assigns almost no probability to what happened |

![cross-entropy-loss-curve]({{ '/assets/notes/neural-network-training/cross-entropy-loss-curve.png' | relative_url }})

The direction is exactly what training needs:

$$
p_y\uparrow
\quad\Longrightarrow\quad
-\log p_y\downarrow,
$$

with the ideal value

$$
p_y=1
\quad\Longrightarrow\quad
\mathcal L=0.
$$

The derivative with respect to the correct-class **probability** is

$$
\frac{\partial\mathcal L}{\partial p_y}
=
-\frac1{p_y}.
$$

Hence

$$
p_y\ll1
\quad\Longrightarrow\quad
\left\lvert
\frac{\partial\mathcal L}{\partial p_y}
\right\rvert
=
\frac1{p_y}
\gg1.
$$

Cross-entropy therefore penalizes a confident mistake much more strongly than an uncertain one.

> [!important] This is not yet the gradient sent into the network
> $-1/p_y$ is the derivative with respect to a probability. Neural networks normally produce logits, then apply softmax. After differentiating through softmax, the derivative with respect to the logits becomes the bounded and much cleaner expression $p-y$, derived below. Conflating these two derivatives makes cross-entropy look less stable than the actual combined computation.

### The full formula and one-hot labels

The multiclass cross-entropy formula is

$$
\boxed{
\mathcal L_{\mathrm{CE}}(y,p)
=
-\sum_{c=1}^{C}y_c\log p_c
}.
$$

Here $p_c$ is the model probability for class $c$, while $y_c$ describes the target distribution. For a single observed label, the target is usually encoded as a one-hot vector. If dog is the second class,

$$
y
=
\begin{bmatrix}
0\\
1\\
0
\end{bmatrix}.
$$

Substitution gives

$$
\begin{aligned}
\mathcal L_{\mathrm{CE}}(y,p)
&=
-\left(
0\log0.2
+1\log0.7
+0\log0.1
\right)\\
&=-\log0.7.
\end{aligned}
$$

Only the true-class term survives, so for a one-hot target

$$
-\sum_{c=1}^{C}y_c\log p_c
=
-\log p_y.
$$

These are not two different losses. The sum is the general distributional form; $-\log p_y$ is its single-label simplification.

> [!note] Soft targets retain the full sum
> If the target is not one-hot—for example label smoothing, distillation, or genuinely ambiguous labels—several $y_c$ values can be nonzero. The loss then asks the model to match an entire target distribution rather than place all probability on one class.

### Why the name “cross-entropy”?

For a probability distribution $q$, entropy is

$$
H(q)
=
-\sum_c q_c\log q_c.
$$

It measures the uncertainty inherent in $q$. If data follow a true distribution $q$ but are described using a model distribution $p$, the cross-entropy is

$$
\boxed{
H(q,p)
=
-\sum_c q_c\log p_c
}.
$$

The word *cross* refers to these two roles: outcomes are weighted according to $q$, while their probabilities are read from $p$.

Cross-entropy decomposes as

$$
\boxed{
H(q,p)
=
H(q)
+
D_{\mathrm{KL}}(q\Vert p)
},
$$

where

$$
D_{\mathrm{KL}}(q\Vert p)
=
\sum_c q_c\log\frac{q_c}{p_c}
\ge0.
$$

The data-generating distribution $q$ is fixed during model training, so $H(q)$ is constant with respect to the parameters. Consequently,

$$
\min_p H(q,p)
\quad\Longleftrightarrow\quad
\min_p D_{\mathrm{KL}}(q\Vert p),
$$

and equality is achieved when $p=q$.

> [!important] Distributional meaning
> Cross-entropy does more than reward the largest entry being correct. In expectation, it rewards reporting the complete probability distribution truthfully. This is why it is useful when confidence and calibration matter, not only the final class label.

For a one-hot target, $H(q)=0$ because the target contains no uncertainty. In that special case, the sample cross-entropy and $D_{\mathrm{KL}}(q\Vert p)$ have the same numerical value. That equality should not be assumed for soft targets.

### Why use a logarithm?

The logarithm is not an arbitrary curve-fitting choice. It connects probability, information, and likelihood.

#### 1. Negative log-probability measures surprise

Define the information content, or surprise, of an event with assigned probability $p$ as

$$
I(p)=-\log p.
$$

Then

$$
p\approx1
\quad\Longrightarrow\quad
I(p)\approx0,
$$

because an expected event is unsurprising, whereas

$$
p\approx0
\quad\Longrightarrow\quad
I(p)\gg1,
$$

because an event the model called nearly impossible is highly surprising. Cross-entropy is the expected surprise under the true distribution.

#### 2. Logarithms turn likelihood products into sums

For independent training examples $(x^{(i)},y^{(i)})$, the model assigns the complete observed label sequence the likelihood

$$
\mathcal P_\theta(\mathcal D)
=
\prod_{i=1}^{N}
p_\theta\!\left(y^{(i)}\mid x^{(i)}\right).
$$

Maximum-likelihood estimation chooses parameters that make the observed data most probable:

$$
\max_\theta
\prod_{i=1}^{N}
p_\theta\!\left(y^{(i)}\mid x^{(i)}\right).
$$

Because the logarithm is strictly increasing, it does not change the maximizer:

$$
\max_\theta
\sum_{i=1}^{N}
\log p_\theta\!\left(y^{(i)}\mid x^{(i)}\right).
$$

Changing maximization to minimization gives

$$
\boxed{
\min_\theta
-\sum_{i=1}^{N}
\log p_\theta\!\left(y^{(i)}\mid x^{(i)}\right)
}.
$$

This is the dataset cross-entropy objective. In ordinary probabilistic classification,

$$
\boxed{
\text{cross-entropy}
\equiv
\text{negative log-likelihood}
}
$$

describes the same expression from two viewpoints:

| Viewpoint | Interpretation |
|---|---|
| information theory | expected surprise when $p$ describes data generated by $q$ |
| statistics | negative log-likelihood of the observed labels under the model |

The logarithm also prevents a product of many probabilities from becoming numerically tiny and converts the dataset objective into a sum of per-example losses, which is compatible with mini-batch training.

#### 3. Confident errors should cost more than hesitant errors

If dog is the true class, compare

$$
p^{(A)}=(0.51,0.49,0),
$$

with

$$
p^{(B)}=(0.99,0.01,0).
$$

Both predict cat, so both have accuracy zero. But

$$
\mathcal L_A=-\log0.49\approx0.713,
$$

while

$$
\mathcal L_B=-\log0.01\approx4.605.
$$

Model B is not merely wrong; it is extremely confident that the truth is nearly impossible. Cross-entropy correctly treats that as a much more serious error.

### Why accuracy cannot normally be the training objective

For one example,

$$
\operatorname{Acc}(p,y)
=
\mathbf 1
\left[
\operatorname*{arg\,max}_c p_c=y
\right].
$$

It reports only zero or one. In the binary illustration below, as the correct-class probability rises from $0.10$ to $0.20$ to $0.40$, the model may be learning substantially while accuracy remains zero. After the $0.5$ decision boundary is crossed, accuracy jumps to one and then cannot distinguish $0.51$ confidence from $0.99$ confidence.

| Correct-class probability — binary case | Accuracy | Cross-entropy |
|---:|---:|---:|
| $0.10$ | $0$ | $2.303$ |
| $0.20$ | $0$ | $1.609$ |
| $0.40$ | $0$ | $0.916$ |
| $0.51$ | $1$ | $0.673$ |
| $0.80$ | $1$ | $0.223$ |
| $0.99$ | $1$ | $0.010$ |

Across large parameter regions, the predicted class does not change, so the accuracy is locally constant and its gradient is zero. At a decision boundary, it jumps and is not differentiable. It therefore supplies neither a reliable direction nor a measure of partial progress.

Cross-entropy provides a smooth surrogate:

$$
\boxed{
\text{accuracy records the final discrete decision;}
\quad
\text{cross-entropy records progress in probability space.}
}
$$

![cross-entropy-accuracy-and-gradient]({{ '/assets/notes/neural-network-training/cross-entropy-accuracy-and-gradient.png' | relative_url }})

The $0.5$ boundary in the left panel is the binary-class case. With more than two classes, the true class becomes the prediction when its probability exceeds every competing probability; it need not exceed $0.5$. The principle—piecewise-constant accuracy versus continuously changing cross-entropy—remains the same.

### Softmax converts logits into probabilities

A network usually produces unrestricted real-valued scores called logits,

$$
z
=
\begin{bmatrix}
z_1\\
\vdots\\
z_C
\end{bmatrix}
\in\mathbb R^C,
$$

rather than valid probabilities directly. Softmax converts them into a categorical distribution:

$$
\boxed{
p_c
=
\frac{e^{z_c}}
{\sum_{j=1}^{C}e^{z_j}}
}.
$$

Every output is positive and the outputs sum to one. For a one-hot target with true class $y$,

$$
\begin{aligned}
\mathcal L
&=-\log p_y\\
&=-\log
\left(
\frac{e^{z_y}}
{\sum_j e^{z_j}}
\right)\\
&=\boxed{
-z_y+\log\sum_j e^{z_j}
}.
\end{aligned}
$$

The first term rewards increasing the true-class logit; the second term couples it to all competing logits. Thus cross-entropy does not ask only for $z_y$ to be large in isolation. It asks for $z_y$ to be large **relative to its competitors**.

This relative nature appears in the shift invariance

$$
\operatorname{softmax}(z+a\mathbf1)
=
\operatorname{softmax}(z).
$$

Adding the same constant to every logit changes no probability and no cross-entropy value. Only logit differences matter.

### Deriving the central gradient: $p-y$

Starting from

$$
\mathcal L
=
-z_y
+
\log\sum_j e^{z_j},
$$

differentiate with respect to a particular logit $z_c$. The first term gives

$$
\frac{\partial(-z_y)}{\partial z_c}
=
-\mathbf1[c=y]
=
-y_c,
$$

while the log-sum-exp term gives

$$
\begin{aligned}
\frac{\partial}{\partial z_c}
\log\sum_j e^{z_j}
&=
\frac{1}{\sum_j e^{z_j}}
\frac{\partial}{\partial z_c}
\sum_j e^{z_j}\\
&=
\frac{e^{z_c}}{\sum_j e^{z_j}}\\
&=p_c.
\end{aligned}
$$

Therefore,

$$
\boxed{
\frac{\partial\mathcal L}{\partial z_c}
=
p_c-y_c
},
$$

or in vector form,

$$
\boxed{
\nabla_z\mathcal L
=
p-y
}.
$$

This is one of the central formulas of classification with neural networks. It says that the backward signal at the output is simply

$$
\text{model's current distribution}
-
\text{target distribution}.
$$

For

$$
p
=
\begin{bmatrix}
0.2\\
0.7\\
0.1
\end{bmatrix},
\qquad
y
=
\begin{bmatrix}
0\\
1\\
0
\end{bmatrix},
$$

the output gradient is

$$
\nabla_z\mathcal L
=
p-y
=
\begin{bmatrix}
0.2\\
-0.3\\
0.1
\end{bmatrix}.
$$

Under gradient descent,

$$
z_c
\leftarrow
z_c
-
\eta(p_c-y_c).
$$

Consequently,

$$
\begin{aligned}
z_{\text{cat}}&\downarrow,\\
z_{\text{dog}}&\uparrow,\\
z_{\text{bird}}&\downarrow.
\end{aligned}
$$

The true-class logit rises because its gradient is negative. Each incorrect logit falls in proportion to the probability currently assigned to that class. A highly believed wrong class therefore receives a stronger correction than an already unlikely wrong class.

> [!tip] A useful conservation check
> Because both $p$ and $y$ sum to one,
> $$
> \sum_c(p_c-y_c)=0.
> $$
> The logit gradients sum to zero, matching softmax's invariance to adding the same constant to every logit. Backpropagation is changing relative scores, not their common offset.

### Binary cross-entropy is the same idea with two outcomes

For a binary target $y\in\{0,1\}$, let $p$ denote the probability of class $1$. The probability of class $0$ is $1-p$. Binary cross-entropy is

$$
\boxed{
\mathcal L_{\mathrm{BCE}}
=
-\left[
y\log p
+
(1-y)\log(1-p)
\right]
}.
$$

The target again acts as a switch:

$$
y=1
\quad\Longrightarrow\quad
\mathcal L=-\log p,
$$

and

$$
y=0
\quad\Longrightarrow\quad
\mathcal L=-\log(1-p).
$$

In both cases,

$$
\mathcal L
=
-\log(\text{probability assigned to the observed answer}).
$$

| Task structure | Output transformation | Typical loss |
|---|---|---|
| one of $C$ mutually exclusive classes | one softmax over $C$ logits | multiclass cross-entropy |
| one binary decision | sigmoid on one logit | binary cross-entropy |
| several independent labels may all be true | one sigmoid per label | sum/mean of binary cross-entropies |

The last case is **multi-label**, not multiclass: an image may simultaneously contain a dog, a person, and a car, so the class probabilities should not be forced to sum to one.

### Real-training caveats

Cross-entropy supplies an excellent local training signal, but it cannot guarantee that the target itself is correct or aligned with the real task.

#### Confident fitting of label noise

Since

$$
p_y\to0
\quad\Longrightarrow\quad
-\log p_y\to\infty,
$$

a mislabeled example can exert strong pressure: the objective still asks the network to become confident in the recorded label. This is one reason to use data cleaning, regularization, early stopping, or losses designed for label noise.

#### Label smoothing changes the target

Instead of the one-hot target, label smoothing may use

$$
y_c^{(\varepsilon)}
=
\begin{cases}
1-\varepsilon,&c=y,\\
\dfrac{\varepsilon}{C-1},&c\ne y.
\end{cases}
$$

This prevents the target from demanding probability exactly one on a single class and can reduce extreme confidence. It also changes the meaning of the desired distribution, so its effect on probability calibration must be evaluated rather than assumed.

#### Low cross-entropy and high accuracy are related, not identical

A model can have unchanged accuracy but better cross-entropy because its confidence improved. Conversely, a few extremely confident mistakes can make cross-entropy poor even when accuracy is high. The metric to report depends on the real decision problem; the loss to optimize must provide a usable signal and remain aligned with that metric.

#### Numerical implementations should use logits directly

Computing

$$
p=\operatorname{softmax}(z)
$$

and then separately evaluating $\log p_y$ can underflow when logits have large magnitude. Stable libraries combine log-softmax and negative log-likelihood, using the log-sum-exp identity

$$
\log\sum_j e^{z_j}
=
m
+
\log\sum_j e^{z_j-m},
\qquad
m=\max_j z_j.
$$

This subtraction changes neither softmax nor the loss because of shift invariance, while preventing unnecessarily large exponentials.

> [!warning] Match the software interface to the mathematics
> A function such as `CrossEntropyLoss` typically expects raw logits and internally performs the stable softmax/log operation. Passing already-softmaxed probabilities can apply the transformation twice and produce the wrong objective and gradient. Always check whether an API expects logits, log-probabilities, or probabilities.

### Durable summary

For one-hot multiclass classification,

$$
\boxed{
\mathcal L_{\mathrm{CE}}(y,p)
=
-\sum_c y_c\log p_c
=
-\log p_y
}.
$$

With softmax probabilities

$$
p_c
=
\frac{e^{z_c}}{\sum_j e^{z_j}},
$$

the loss and its output gradient are

$$
\boxed{
\mathcal L
=
-z_y+\log\sum_j e^{z_j},
\qquad
\nabla_z\mathcal L
=
p-y
}.
$$

The conceptual chain is

$$
\boxed{
\text{discrete classification task}
\to
\text{probability model}
\to
\text{negative log-likelihood}
\to
\text{continuous error signal }p-y
\to
\text{backpropagation}
}.
$$

> [!summary] The central insight
> Accuracy says only whether the current decision crossed the boundary. Cross-entropy measures how much probability the model assigned to what actually happened. It is useful not merely because it is differentiable, but because its gradient communicates a graded, probability-aware discrepancy to the rest of the network.

This section makes the surrogate-loss idea in [When a gradient stops being useful](#when-a-gradient-stops-being-useful) concrete. The next section broadens the intervention toolbox: function-evaluation estimators when derivatives are unavailable, clipping when gradient scale is dangerous, and activation choices that change the local geometry of the composed objective.

---

## Connections

- The objective, sampler, unbiasedness, and noise model used by this update are developed in [Learning Objectives and Mini-Batch Gradient Information]({{ '/notes/mit6-7960-02-1-learning-objectives-and-mini-batch-gradient-information/' | relative_url }}).
- When a useful computation-graph derivative is unavailable, function-evaluation search is developed in [Gaussian Smoothing and Evolution Strategies]({{ '/notes/mit6-7960-02-3-gaussian-smoothing-and-evolution-strategies/' | relative_url }}).
- The deterministic descent mechanism serving as the comparison point is proved in [Gradient Descent, Smoothness, and the Descent Lemma]({{ '/notes/mit6-7960-01-3-gradient-descent-smoothness-and-the-descent-lemma/' | relative_url }}).
- The conditioning, smoothness, and practice boundaries behind these training dynamics are collected in [Conditioning and Practical Gradient Descent]({{ '/notes/mit6-7960-01-5-conditioning-and-practical-gradient-descent/' | relative_url }}).
