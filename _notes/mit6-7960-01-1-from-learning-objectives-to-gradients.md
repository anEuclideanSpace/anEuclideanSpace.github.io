---
layout: note
title: "From Learning Objectives to Gradients"
source_title: "MIT6.7960-01.1-From Learning Objectives to Gradients"
course: "MIT 6.7960"
sequence: "01.1"
source_context: "The Gradient, Steepest Descent, and the Guarantees of Gradient Descent"
permalink: "/notes/mit6-7960-01-1-from-learning-objectives-to-gradients/"
tags:
  - "math/optimization"
  - "deep-learning/foundations"
  - "topic/gradients"
---
Optimization theory begins with a scalar function, but learning problems begin with data, predictions, and losses. This note develops the entire bridge from a learning task to an optimization objective, explains why high-dimensional optimization must rely on local information, and then constructs the partial derivatives and gradient that provide its basic first-order interface. The three parts are kept together because the gradient is not an isolated formula: it is meaningful only after both the optimized function and the information problem have been defined.

![gradient-descent-landscape]({{ '/assets/notes/gradient-descent/gradient-descent-landscape.png' | relative_url }})

> [!seealso] Next note in this learning path
> *How to Train a Neural Net — Backpropagation and Differentiable Programming* continues from the geometry and guarantees developed here to computation graphs, reverse-mode differentiation, backpropagation through MLPs and DAGs, and differentiable programming.

> [!note] Scope after splitting
> The source link above preserves the broader continuation stated in the original text. Within the present concept-note graph, [Learning Objectives and Mini-Batch Gradient Information]({{ '/notes/mit6-7960-02-1-learning-objectives-and-mini-batch-gradient-information/' | relative_url }}) is the directly related continuation: it specializes the learning objective to empirical data access, mini-batch gradients, unbiasedness, and gradient noise.

## 0.0 From learning to optimization

Before any of the machinery below, one question deserves an answer: where does the scalar function \\(f\\) that the chapter spends its time minimizing actually come from? For a reader new to machine learning, the passage from "training a model" to "minimizing a function on \\(\mathbb{R}^n\\)" is itself a step worth making explicit.

**The objects.** In the basic **supervised-learning** setting, training starts from a dataset of input–label pairs

\\[\mathcal{D}=\{(a_i,b_i)\}_{i=1}^m,\\]

where each \\(a_i\\) is an input (an image, a sentence, a feature vector) and \\(b_i\\) its label. A **model** with parameters \\(\theta\in\mathbb{R}^n\\) turns an input into a prediction,

\\[\hat b_i=h_\theta(a_i),\\]

and a **loss function** \\(\ell\\) scores how wrong that single prediction is,

\\[\ell\big(h_\theta(a_i),\,b_i\big).\\]

Averaging the per-sample losses over the training set gives the **empirical risk**, i.e. the training loss,

\\[F(\theta)=\frac1m\sum_{i=1}^m \ell\big(h_\theta(a_i),\,b_i\big).\\]

> [!note] Regularization is part of the objective
> The pure empirical risk is the simplest baseline. A practical objective may include a regularizer \\(\mathcal R\\):
>
> \\[
> F_\lambda(\theta)
> =\frac1m\sum_{i=1}^m\ell\big(h_\theta(a_i),b_i\big)
> +\lambda\mathcal R(\theta),
> \qquad
> \lambda\ge0.
> \\]
>
> The optimizer still sees one scalar function; throughout the chapter, \\(F\\) may be read as the **total** objective. The smooth results apply when that total objective satisfies their assumptions. A non-smooth regularizer such as an \\(\ell_1\\) penalty belongs to the boundary discussed in §1.13. Unsupervised and self-supervised learning define different per-example objectives, but the same optimization abstraction begins once a scalar \\(F(\theta)\\) has been specified.

Training, as a computational goal, is the minimization

\\[\min_{\theta\in\mathbb{R}^n} F(\theta).\\]

**The dictionary to the rest of the chapter.** Once the learning objective is defined, the notation changes but the mathematical objects do not:

| Machine learning | Optimization | Role |
|---|---|---|
| \\(\theta\in\mathbb{R}^n\\) | \\(x\in\mathbb{R}^n\\) | the variable being changed |
| \\(F(\theta)\\) | \\(f(x)\\) | the scalar objective being minimized |
| \\(\nabla_\theta F(\theta)\\) | \\(\nabla f(x)\\) | the local first-order signal used to update the variable |
| training data \\((a_i,b_i)\\) | part of the definition of \\(F\\) | held fixed when differentiating with respect to the parameters |

Thus the learning problem and the abstract optimization problem are the same statement in two vocabularies,

\\[
\min_{\theta\in\mathbb{R}^n}F(\theta)
\qquad\longleftrightarrow\qquad
\min_{x\in\mathbb{R}^n}f(x),
\\]

and a gradient-descent update translates as

\\[
\theta_{k+1}=\theta_k-\eta\nabla_\theta F(\theta_k)
\qquad\longleftrightarrow\qquad
x_{k+1}=x_k-\eta\nabla f(x_k).
\\]

> [!important] What is the gradient taken with respect to?
> During training, the parameters move while the dataset is held fixed. The relevant gradient is therefore \\(\nabla_\theta F(\theta)\\), not a derivative with respect to an input \\(a_i\\). Input gradients answer a different question and are useful in settings such as sensitivity analysis and adversarial examples, but they are not the parameter update studied here.

> [!example]- A one-parameter sanity check
> Let \\(h_\theta(a)=\theta a\\) and use the single-example squared loss
> \\[
> \ell(\theta)=\frac12(\theta a-b)^2.
> \\]
> Holding \\(a\\) and \\(b\\) fixed gives
> \\[
> \frac{d\ell}{d\theta}=(\theta a-b)a.
> \\]
> At \\(a=2\\), \\(b=6\\), and \\(\theta=1\\), this derivative is \\(-8\\). A negative-gradient step therefore increases \\(\theta\\), exactly as it should: the current prediction is \\(2\\), below the target \\(6\\).

> [!tip] Optimization deliberately forgets semantics
> Once \\(F\\) is defined, the optimizer need not know whether the \\(a_i\\) are images, sentences, or feature vectors. It interacts with the learning problem through quantities such as \\(F(\theta)\\) and \\(\nabla_\theta F(\theta)\\). The task-specific meaning is compressed into the geometry of one function on parameter space.

**Two boundaries to keep in view from the start.** The optimization problem above is a deliberate abstraction. Before using it, separate what the objective measures from how its gradient is computed.

> [!warning] Optimization is not generalization
> The empirical risk
> \\[
> F(\theta)=\frac1m\sum_{i=1}^m \ell\big(h_\theta(a_i),b_i\big)
> \\]
> measures performance on the observed training set. What ultimately matters is usually the **population risk**
> \\[
> R(\theta)=\mathbb E_{(a,b)\sim P}\!\left[\ell\big(h_\theta(a),b\big)\right],
> \\]
> where \\(P\\) is the unknown data-generating distribution. The distinction is structural:
>
> | Quantity | Data used | Directly computable? |
> |---|---|---|
> | empirical risk \\(F(\theta)\\) | the finite training set | yes |
> | population risk \\(R(\theta)\\) | the full distribution \\(P\\) | generally no |
>
> Their difference, \\(R(\theta)-F(\theta)\\), is the **generalization gap**. A sufficiently expressive model may memorize the training set and achieve \\(F(\theta)\approx0\\) while \\(R(\theta)\\) remains large. A theorem showing that gradient descent lowers \\(F\\) establishes successful **optimization**; it does not by itself establish successful **learning**.

> [!abstract] Scope of this chapter: deterministic full-batch gradient descent
> The full gradient averages over all \\(m\\) training examples,
> \\[
> \nabla F(\theta)
> =\frac1m\sum_{i=1}^m
> \nabla_\theta\ell\big(h_\theta(a_i),b_i\big),
> \\]
> and the update is
> \\[
> \theta_{k+1}=\theta_k-\eta\nabla F(\theta_k).
> \\]
> With the objective, initialization, and step rule fixed, this iteration is deterministic. Practical training more commonly uses a sampled mini-batch \\(\mathcal B_k\\) and the estimator
> \\[
> g_k=\frac1{|\mathcal B_k|}\sum_{i\in\mathcal B_k}
> \nabla_\theta\ell\big(h_\theta(a_i),b_i\big),
> \qquad
> \theta_{k+1}=\theta_k-\eta g_k.
> \\]
> Typically \\(g_k\neq\nabla F(\theta_k)\\) for an individual batch. Under standard uniform-sampling conditions it is unbiased,
> \\[
> \mathbb E[g_k\mid\theta_k]=\nabla F(\theta_k),
> \\]
> so one may write \\(g_k=\nabla F(\theta_k)+\xi_k\\), with \\(\xi_k\\) representing sampling noise. This chapter begins with the full-batch case to isolate direction, step size, and curvature; §1.14 returns to what changes when stochasticity is introduced.

> [!tip] The distinction to remember
> **Optimization asks whether we minimize the objective; generalization asks whether it was the right objective.**

---

## 1.0 The central question

> [!definition] The unconstrained optimization problem
> Let
> \\[
> f:\mathbb{R}^n\to\mathbb{R},
> \qquad
> x=(x_1,\ldots,x_n)^\top\in\mathbb{R}^n.
> \\]
> This chapter studies
> \\[
> \min_{x\in\mathbb{R}^n} f(x).
> \\]
> Here \\(x\\) is the **optimization variable** and \\(f(x)\\) its scalar **objective value**. In machine learning, \\(x\\) may collect all trainable model parameters; the mathematics is not specific to neural-network weights. The domain is all of \\(\mathbb{R}^n\\), so no feasibility constraints are imposed.

> [!note]- Why say “make \\(f\\) small” rather than “find the global minimizer”?
> For a general non-convex objective, a global minimizer may be difficult to find or may not be attained at any finite point. Moreover, the statements
> \\[
> f(x_{k+1})<f(x_k),
> \qquad
> \|\nabla f(x_k)\|\to0,
> \qquad
> f(x_k)\to\inf_x f(x)
> \\]
> express three different strengths of guarantee: **descent**, **approach to stationarity**, and **approach to global optimality**. None should be silently substituted for another. Later sections state exactly which conclusion follows from each structural assumption on \\(f\\).

The difficulty is one of **information**, not intent. In modern machine learning, \\(n\\) may reach millions or billions, so the loss landscape cannot be surveyed as a whole. Even an impossibly coarse search with only \\(q\\) candidate values per coordinate would require

\\[
\underbrace{q\times q\times\cdots\times q}_{n\text{ coordinates}}=q^n
\\]

function evaluations. With \\(q=10\\) and only \\(n=100\\), this is already \\(10^{100}\\) candidate points.

> [!important] The information bottleneck
> An optimizer does not receive a global map of \\(f\\). At the current point \\(x\\), the scalable local query is
> \\[
> \big(f(x),\nabla f(x)\big)\in\mathbb R\times\mathbb R^n.
> \\]
> The value \\(f(x)\\) reports the current height; the gradient \\(\nabla f(x)\\) reports first-order change nearby. Neither directly reveals what lies far away. The central problem is therefore whether repeated **local** queries can support a **global** statement about the resulting iterates.

> [!note]- Why not use the full Hessian?
> Second-order curvature is encoded by the Hessian \\(\nabla^2 f(x)\in\mathbb R^{n\times n}\\). The sizes of the relevant objects scale differently:
>
> | Object | Shape | Number of entries |
> |---|---:|---:|
> | value \\(f(x)\\) | scalar | \\(1\\) |
> | gradient \\(\nabla f(x)\\) | vector | \\(n\\) |
> | Hessian \\(\nabla^2 f(x)\\) | matrix | \\(n^2\\) |
> | Hessian–vector product \\(\nabla^2 f(x)v\\) | vector | \\(n\\) |
>
> At \\(n=10^6\\), a full Hessian contains \\(10^{12}\\) entries, so explicitly forming or storing it is usually prohibitive. A Hessian–vector product extracts curvature along one chosen direction and can often be computed without materializing the matrix, which is why some scalable methods can use limited second-order information.

\\[
\boxed{\text{When can local information support a global guarantee?}}
\\]

> [!summary] Two questions organize the chapter
> 1. **Direction:** using only local first-order information, which unit direction decreases \\(f\\) fastest? Under the Euclidean norm and when \\(\nabla f(x)\neq0\\), the answer will be \\(-\nabla f(x)/\|\nabla f(x)\|\\); at a stationary point, first-order information selects no direction.
> 2. **Guarantee:** once a step rule is fixed, what can be proved about the entire sequence of iterates — not merely hoped for?
>
> The first question does not settle the second. A direction specifies a ray; an algorithm must also choose a step length:
> \\[
> \underbrace{-\frac{\nabla f(x)}{\|\nabla f(x)\|}}_{\text{unit direction}}
> \qquad\neq\qquad
> \underbrace{-\eta\nabla f(x)}_{\text{actual step}}.
> \\]
> The gradient selects the local direction, while the value \\(f(x)\\) records the height used to state descent. Turning those local quantities into progressively stronger guarantees requires progressively stronger structure:
>
> | Stage | Added structure or choice | Core conclusion | Section |
> |---|---|---|---:|
> | local geometry | differentiability + Euclidean norm | if \\(\nabla f\neq0\\), \\(-\nabla f\\) is the steepest-descent ray | §1.1–§1.5 |
> | iteration | choose a step scale \\(\eta\\) | \\(x_{k+1}=x_k-\eta\nabla f(x_k)\\) | §1.6 |
> | per-step descent | \\(L\\)-smoothness + \\(0<\eta<2/L\\) | every non-stationary step lowers \\(f\\) | §1.7–§1.9 |
> | long-run stationarity | additionally, \\(f\\) is bounded below | \\(\|\nabla f(x_k)\|\to0\\) | §1.10 |
> | fast global convergence | additionally, the PL condition | the function-value gap contracts geometrically | §1.11 |
>
> \\[
> \boxed{\text{local geometry}+\text{global structure}=\text{provable iterative guarantees}}
> \\]

This local-to-global passage is the load-bearing distinction: choosing a sensible direction is a geometric statement at one point, whereas proving convergence requires assumptions that control \\(f\\) across every point the iterates may visit.

---

## 1.1 Partial derivatives and the gradient

> [!definition] Partial derivatives as coordinate slices
> Fix a point \\(x\in\mathbb R^n\\) and a coordinate \\(i\\). The \\(i\\)-th standard basis vector
> \\[
> e_i=(0,\ldots,0,\underset{i}{1},0,\ldots,0)^\top
> \\]
> selects that coordinate, because
> \\[
> x+h e_i=(x_1,\ldots,x_{i-1},x_i+h,x_{i+1},\ldots,x_n)^\top.
> \\]
> Restrict \\(f\\) to this coordinate line by defining the one-variable slice
> \\[
> \varphi_i(h):=f(x+h e_i).
> \\]
> The \\(i\\)-th partial derivative is just the ordinary derivative of this slice at \\(h=0\\):
> \\[
> \frac{\partial f}{\partial x_i}(x)
> =\varphi_i'(0)
> =\lim_{h\to0}\frac{f(x+h e_i)-f(x)}{h}.
> \\]
> When all \\(n\\) partial derivatives exist, the **gradient** stacks these coordinate slopes into a vector:
> \\[
> \nabla f(x)
> =\begin{bmatrix}
> \dfrac{\partial f}{\partial x_1}(x)\\[3pt]
> \vdots\\[3pt]
> \dfrac{\partial f}{\partial x_n}(x)
> \end{bmatrix}
> \in\mathbb R^n.
> \\]

> [!example]- What does \\(x+h e_i\\) actually change?
> In \\(\mathbb R^3\\), take
> \\[
> x=\begin{bmatrix}2\\\\ -1\\\\ 5\end{bmatrix},
> \qquad
> e_2=\begin{bmatrix}0\\\\ 1\\\\ 0\end{bmatrix}.
> \\]
> Then
> \\[
> x+h e_2=\begin{bmatrix}2\\\\ -1+h\\\\ 5\end{bmatrix}:
> \\]
> only the second coordinate moves. Computing \\(\partial f/\partial x_2\\) means differentiating \\(f\\) along precisely this line through \\(x\\).

> [!note]- Type check
>
> | Object | Type | Role |
> |---|---|---|
> | \\(h\\) | scalar | displacement along one coordinate |
> | \\(e_i\\) | vector in \\(\mathbb R^n\\) | selects coordinate \\(i\\) |
> | \\(x+h e_i\\) | vector in \\(\mathbb R^n\\) | perturbed input |
> | \\(\partial f/\partial x_i\\) | scalar | slope along one coordinate axis |
> | \\(\nabla f(x)\\) | vector in \\(\mathbb R^n\\) | all coordinate slopes stacked together |

At this stage, \\(\nabla f(x)\\) is only a vector of slopes measured along the coordinate axes. We have not yet shown that it gives a single linear approximation valid in every direction, nor that it selects a steepest direction. Those stronger geometric meanings require differentiability and are established in the next sections.

> [!warning] Trap — existence of partials does not imply differentiability
> Differentiability has two necessary consequences:
> \\[
> f\text{ differentiable at }x
> \quad\Longrightarrow\quad
> \begin{cases}
> f\text{ is continuous at }x,\\
> \text{all partial derivatives exist at }x.
> \end{cases}
> \\]
> Neither implication reverses in general. The partial derivatives probe only the \\(n\\) coordinate lines \\(x+h e_i\\); differentiability demands one linear map whose error is uniformly negligible over **every** small displacement \\(\delta\\).

> [!example] A degree-zero homogeneous obstruction
> Define
> \\[
> f(x,y)=
> \begin{cases}
> \dfrac{xy}{x^2+y^2}, &(x,y)\neq(0,0),\\[5pt]
> 0, &(x,y)=(0,0).
> \end{cases}
> \\]
> The function vanishes on both coordinate axes, so
> \\[
> \frac{\partial f}{\partial x}(0,0)
> =\lim_{h\to0}\frac{f(h,0)-f(0,0)}{h}=0,
> \qquad
> \frac{\partial f}{\partial y}(0,0)=0.
> \\]
> Thus the vector of partials at the origin is \\(g=(0,0)^\top\\). Other paths expose what the coordinate probes miss:
>
> | Path to \\((0,0)\\) | Value of \\(f\\) |
> |---|---:|
> | \\(y=0\\) | \\(0\\) |
> | \\(x=0\\) | \\(0\\) |
> | \\(y=x\\) | \\(1/2\\) |
> | \\(y=-x\\) | \\(-1/2\\) |
>
> The structural reason is **degree-zero homogeneity**. On the punctured plane, for every \\(\lambda\neq0\\),
> \\[
> f(\lambda x,\lambda y)=f(x,y).
> \\]
> Scaling toward the origin changes the radius but never the value along a fixed ray. Equivalently, with \\(x=r\cos\theta\\) and \\(y=r\sin\theta\\),
> \\[
> f(r\cos\theta,r\sin\theta)=\frac12\sin(2\theta):
> \\]
> the radius disappears, leaving only the angle.
>
> More generally, a nonconstant degree-zero homogeneous function on \\(\mathbb R^n\setminus\{0\}\\) cannot have a continuous extension to the origin. If such an extension existed, then for any fixed \\(z\neq0\\),
> \\[
> f(z)=f(tz)\longrightarrow f(0)\qquad(t\to0^+),
> \\]
> forcing \\(f(z)=f(0)\\) for every \\(z\\) — contradicting nonconstancy. Here the path table already shows the contradiction, so \\(f\\) is not continuous at the origin and therefore cannot be differentiable there.
>
> The differentiability remainder fails even more directly. Since the partials force \\(g=0\\), take \\(\delta=(t,t)\\):
> \\[
> \frac{|f(\delta)-f(0)-g^\top\delta|}{\|\delta\|}
> =\frac{1/2}{\sqrt2\,|t|}
> \longrightarrow\infty,
> \\]
> whereas differentiability requires this ratio to tend to \\(0\\).

> [!tip] Coordinate slopes versus one linear model
> Partial derivatives answer \\(n\\) separate one-dimensional questions. Differentiability asks whether a **single** linear map answers all directions at once, with a uniform error rate.

![partial-derivatives-and-coordinate-slices]({{ '/assets/notes/gradient-descent/partial-derivatives-and-coordinate-slices.png' | relative_url }})
*The height of \\(f=xy/(x^2+y^2)\\) depends only on the angle around the origin: it is \\(0\\) along both axes (both partials vanish, black), yet climbs to \\(+\tfrac12\\) along \\(y=x\\) (green) and falls to \\(-\tfrac12\\) along \\(y=-x\\) (purple). No single value at the origin reconciles these, so \\(f\\) is discontinuous there and not differentiable.*

> [!example] A hand-computed gradient
> Consider the smooth function
> \\[
> f(x_1,x_2)=x_1^2+3x_1x_2,
> \qquad
> x=(x_1,x_2)^\top\in\mathbb R^2,
> \qquad
> f(x)\in\mathbb R.
> \\]
> When differentiating with respect to one coordinate, hold the other fixed. The calculation is easiest to scan term by term:
>
> | Term | \\(\partial/\partial x_1\\) | \\(\partial/\partial x_2\\) |
> |---|---:|---:|
> | \\(x_1^2\\) | \\(2x_1\\) | \\(0\\) |
> | \\(3x_1x_2\\) | \\(3x_2\\) | \\(3x_1\\) |
> | **Total** | \\(2x_1+3x_2\\) | \\(3x_1\\) |
>
> Stack the two scalar partials into the gradient column vector:
> \\[
> \nabla f(x_1,x_2)
> =\begin{bmatrix}
> 2x_1+3x_2\\[2pt]
> 3x_1
> \end{bmatrix}.
> \\]
> At \\(x=(1,2)^\top\\),
> \\[
> f(1,2)=7,
> \qquad
> \nabla f(1,2)
> =\begin{bmatrix}8\\\\ 3\end{bmatrix}.
> \\]
> Thus the instantaneous coordinate slopes at this point are \\(8\\) along the \\(x_1\\) axis and \\(3\\) along the \\(x_2\\) axis.

> [!note]- Derivatives describe infinitesimal change
> The value \\(\partial f/\partial x_1(1,2)=8\\) does not mean that every finite unit increase in \\(x_1\\) changes \\(f\\) by exactly \\(8\\). For a displacement \\(h\\),
> \\[
> f(1+h,2)-f(1,2)=8h+h^2.
> \\]
> The derivative supplies the linear term \\(8h\\); the leftover is negligible relative to \\(h\\) because
> \\[
> \frac{h^2}{|h|}=|h|\longrightarrow0.
> \\]

> [!example]- Verify a directional derivative from its definition
> Use the diagonal unit vector
> \\[
> u=\frac1{\sqrt2}\begin{bmatrix}1\\\\ 1\end{bmatrix},
> \qquad
> \|u\|=1.
> \\]
> The directional derivative is defined directly by
> \\[
> D_u f(1,2)
> =\lim_{h\to0}\frac{f((1,2)+hu)-f(1,2)}{h}.
> \\]
> Set \\(a=h/\sqrt2\\), so \\((1,2)+hu=(1+a,2+a)\\). Expanding,
> \\[
> \begin{aligned}
> f(1+a,2+a)
> &=(1+a)^2+3(1+a)(2+a)\\
> &=7+11a+4a^2\\
> &=7+\underbrace{\frac{11}{\sqrt2}h}_{\text{first order}}
> +\underbrace{2h^2}_{\text{higher order}}.
> \end{aligned}
> \\]
> Since \\(f(1,2)=7\\),
> \\[
> D_u f(1,2)
> =\lim_{h\to0}\left(\frac{11}{\sqrt2}+2h\right)
> =\frac{11}{\sqrt2}\approx7.78.
> \\]
> Only after this independent calculation do we compare it with
> \\[
> \nabla f(1,2)^\top u
> =\begin{bmatrix}8&3\end{bmatrix}
> \frac1{\sqrt2}\begin{bmatrix}1\\\\ 1\end{bmatrix}
> =\frac{11}{\sqrt2}.
> \\]
> The axis check is recovered as well: for \\(e_1=(1,0)^\top\\), both the definition and \\(\nabla f(1,2)^\top e_1\\) give \\(8\\).

> [!important] Verification is not proof
> The limit calculation did **not** assume \\(D_u f(x)=\nabla f(x)^\top u\\); it independently produced the same number in one example. This is evidence and a sanity check, not a proof of the general identity. Sections §1.2–§1.3 establish the identity for every direction when \\(f\\) is differentiable.

> [!summary] Coordinate slopes, the gradient, and a directional slope
>
> | Object | Type | Question answered |
> |---|---|---|
> | \\(\partial f/\partial x_i\\) | scalar | What is the slope along coordinate axis \\(e_i\\)? |
> | \\(\nabla f(x)\\) | vector | What vector represents all first-order change in Euclidean coordinates? |
> | \\(D_u f(x)\\) | scalar | What is the slope along the chosen direction \\(u\\)? |
> | \\(\nabla f(x)^\top u\\) | scalar | What slope does the gradient assign to direction \\(u\\)? |

> [!tip] The gradient is not “the slope”
> Once differentiability is established, the derivative is the linear map \\(u\mapsto D_u f(x)\\). The Euclidean inner product represents that map by the unique vector \\(\nabla f(x)\\):
> \\[
> D_u f(x)=\langle\nabla f(x),u\rangle.
> \\]
> The gradient's individual components depend on the chosen coordinate axes, but under an orthogonal change of coordinates both \\(u\\) and \\(\nabla f\\) transform together, leaving their inner product — the actual directional rate — unchanged.

> [!note]- Why use a unit direction?
> Let \\(v=(1,1)^\top=\sqrt2\,u\\). In this example,
> \\[
> D_v f(1,2)=11=\sqrt2\,D_u f(1,2).
> \\]
> The larger number does not describe a steeper geometric direction: \\(v\\) and \\(u\\) lie on the same ray, but \\(v\\) is longer. Fixing \\(\|u\|=1\\) removes this arbitrary scaling and makes directional slopes comparable. Section §1.3 develops this point, and §1.5 shows why the chosen norm determines what “steepest” means.

> [!summary] Three ways to obtain derivatives
>
> | Method | Operates on | Produces | Primary use |
> |---|---|---|---|
> | symbolic differentiation | a mathematical expression | another expression for the derivative | algebra and analytic derivations |
> | automatic differentiation | a program built from differentiable primitive operations | derivative values at the executed input | model training |
> | finite differences | repeated evaluations of \\(f\\) at perturbed inputs | a numerical approximation | gradient checking |

**Symbolic differentiation** transforms a formula for \\(f\\) into a formula for its derivative. For the running example,

\\[
f(x_1,x_2)=x_1^2+3x_1x_2
\quad\longmapsto\quad
\nabla f(x_1,x_2)=
\begin{bmatrix}2x_1+3x_2\\\\ 3x_1\end{bmatrix}.
\\]

It returns a reusable expression, but large programs need not admit a compact symbolic form and may suffer from expression swell.

**Automatic differentiation (autodiff)** decomposes the executed program into primitive operations and composes their local derivatives by the chain rule. It neither symbolically simplifies the whole program nor estimates slopes by perturbing the input. For the common machine-learning shape \\(f:\mathbb R^n\to\mathbb R\\), reverse-mode autodiff propagates sensitivity backward from the scalar output and returns all \\(n\\) partial derivatives in one reverse sweep; backpropagation is this organization of the chain rule on a neural-network computation graph. Section §1.14 compares its computational and memory costs with per-coordinate finite differences.

> [!important] “Exact up to floating point” has a boundary
> Autodiff computes the chain-rule derivative of the **executed differentiable operations**, subject to floating-point arithmetic. If an operation is not differentiable at the executed point, no classical derivative exists for autodiff to recover. Frameworks instead adopt a convention — for example, they commonly assign a chosen value to the derivative of \\(\operatorname{ReLU}(x)=\max(0,x)\\) at \\(x=0\\).

**Finite-difference gradient checking** approximates a partial from function values alone. The centered formula is

\\[
g_i^{\mathrm{FD}}(x;\varepsilon)
:=\frac{f(x+\varepsilon e_i)-f(x-\varepsilon e_i)}{2\varepsilon}
\approx\frac{\partial f}{\partial x_i}(x).
\\]

> [!note]- Why is the centered difference second-order accurate?
> If the one-dimensional coordinate slice is sufficiently smooth, Taylor expansion gives
> \\[
> \begin{aligned}
> f(x+\varepsilon e_i)
> &=f(x)+\varepsilon\partial_i f(x)
> +\frac{\varepsilon^2}{2}\partial_{ii}f(x)
> +\frac{\varepsilon^3}{6}\partial_{iii}f(x)+O(\varepsilon^4),\\
> f(x-\varepsilon e_i)
> &=f(x)-\varepsilon\partial_i f(x)
> +\frac{\varepsilon^2}{2}\partial_{ii}f(x)
> -\frac{\varepsilon^3}{6}\partial_{iii}f(x)+O(\varepsilon^4).
> \end{aligned}
> \\]
> Subtracting cancels the constant and even-order terms; dividing by \\(2\varepsilon\\) leaves
> \\[
> g_i^{\mathrm{FD}}(x;\varepsilon)
> =\partial_i f(x)+O(\varepsilon^2).
> \\]

> [!warning] A smaller \\(\varepsilon\\) is not always more accurate
> Two errors compete. A useful schematic model is
> \\[
> E(\varepsilon)
> \approx
> \underbrace{C_1\varepsilon^2}_{\text{truncation}}
> +\underbrace{C_2\frac{u_{\mathrm{mach}}}{\varepsilon}}_{\text{round-off and cancellation}},
> \\]
> where \\(u_{\mathrm{mach}}\\) is machine precision. A large \\(\varepsilon\\) leaves truncation error; a tiny \\(\varepsilon\\) subtracts nearly equal floating-point numbers and amplifies their error through division by \\(\varepsilon\\). The best scale depends on the function and parameter magnitudes.

> [!example]- Check the previous polynomial
> At \\((1,2)\\),
> \\[
> f(1+\varepsilon,2)=7+8\varepsilon+\varepsilon^2,
> \qquad
> f(1-\varepsilon,2)=7-8\varepsilon+\varepsilon^2.
> \\]
> Hence, in exact arithmetic,
> \\[
> g_1^{\mathrm{FD}}
> =\frac{16\varepsilon}{2\varepsilon}=8
> =\frac{\partial f}{\partial x_1}(1,2).
> \\]
> The centered formula is exact here because the even quadratic terms cancel; floating-point evaluation may still introduce round-off.

> [!tip] Autodiff computes; finite differences check
> A centered finite-difference gradient needs about \\(2n\\) evaluations of \\(f\\) for \\(n\\) parameters. Reverse-mode autodiff shares one chain-rule sweep across all parameters, which is why training uses backpropagation and reserves finite differences for small debugging checks.

> [!note]- A reliable gradient-check checklist
> - Fix the mini-batch and random seed; disable stochastic layers such as dropout.
> - Prefer double precision for the check.
> - Avoid known nondifferentiable points.
> - On a large problem, check a small subset of coordinates or a reduced test case.
> - Compare relative rather than only absolute error, for example
>   \\[
>   \operatorname{relerr}_i
>   =\frac{|g_i^{\mathrm{AD}}-g_i^{\mathrm{FD}}|}
>   {\max\!\left(1,|g_i^{\mathrm{AD}}|,|g_i^{\mathrm{FD}}|\right)}.
>   \\]

---

## Connections

- The coordinate derivatives constructed here become a uniform geometric model in [Differentiability, Directional Derivatives, and Steepest Descent]({{ '/notes/mit6-7960-01-2-differentiability-directional-derivatives-and-steepest-descent/' | relative_url }}).
- The same objective construction is specialized to neural-network training and stochastic data access in [Learning Objectives and Mini-Batch Gradient Information]({{ '/notes/mit6-7960-02-1-learning-objectives-and-mini-batch-gradient-information/' | relative_url }}).
