---
layout: note
title: "Differentiability, Directional Derivatives, and Steepest Descent"
source_title: "MIT6.7960-01.2-Differentiability, Directional Derivatives, and Steepest Descent"
course: "MIT 6.7960"
sequence: "01.2"
source_context: "The Gradient, Steepest Descent, and the Guarantees of Gradient Descent"
permalink: "/notes/mit6-7960-01-2-differentiability-directional-derivatives-and-steepest-descent/"
tags:
  - "math/multivariable-calculus"
  - "math/optimization"
  - "topic/steepest-descent"
---
Partial derivatives describe coordinate slices, but optimization needs one approximation that controls every sufficiently small displacement. This note develops differentiability as that uniform local linear model, derives directional derivatives from it, translates the result into level-set geometry, and proves the Euclidean steepest-direction theorem. These topics form one continuous argument: differentiability produces a linear map, directional derivatives query it, level sets reveal its geometry, and constrained optimization over unit directions identifies its extremal vector.

## 1.2 Differentiability: the local linear model

> [!definition] Differentiability at a point
> For a displacement
> $$
> \delta=(\delta_1,\ldots,\delta_n)^\top,
> \qquad
> x+\delta=(x_1+\delta_1,\ldots,x_n+\delta_n)^\top,
> $$
> define the error left by a candidate linear model $g$:
> $$
> r_x(\delta)
> :=f(x+\delta)-f(x)-\langle g,\delta\rangle,
> \qquad
> \langle g,\delta\rangle=\sum_{i=1}^n g_i\delta_i.
> $$
> The function $f$ is **differentiable at $x$** if there exists a vector $g\in\mathbb R^n$ such that
> $$
> r_x(\delta)=o(\|\delta\|),
> $$
> equivalently,
> $$
> \lim_{\delta\to0}
> \frac{|f(x+\delta)-f(x)-\langle g,\delta\rangle|}
> {\|\delta\|}=0.
> $$
> Thus, near $x$, the increment of $f$ is linear to first order:
> $$
> f(x+\delta)-f(x)=\langle g,\delta\rangle+o(\|\delta\|).
> $$

> [!important] The quantifiers are the point
> The little-$o$ condition means
> $$
> \forall\varepsilon>0\;\exists\rho>0:\quad
> 0<\|\delta\|<\rho
> \Longrightarrow
> |r_x(\delta)|\le\varepsilon\|\delta\|.
> $$
> After an admissible relative error $\varepsilon$ is chosen, one radius $\rho$ must make the **same** linear model accurate for every displacement inside that neighborhood. The radius cannot be chosen separately for each direction. Equivalently, the ratio must tend to zero along every sequence $\delta_k\to0$.

The division by $\|\delta\|$ is essential: an error may tend to zero without becoming negligible relative to the displacement.

| Remainder $r_x(\delta)$ | Ratio $\lvert r_x(\delta)\rvert/\|\delta\|$ | $o(\|\delta\|)$? |
|---|---:|---:|
| $\|\delta\|^2$ | $\|\delta\|\to0$ | yes |
| $\|\delta\|^{3/2}$ | $\|\delta\|^{1/2}\to0$ | yes |
| $0.1\|\delta\|$ | $0.1$ | no |
| $\|\delta\|$ | $1$ | no |

In particular, differentiability does **not** require a quadratic remainder. A bound $O(\|\delta\|^2)$ is sufficient but strictly stronger than the first-order requirement $o(\|\delta\|)$.

> [!tip] Linear increment, affine value model
> The map $\delta\mapsto\langle g,\delta\rangle$ is linear in the displacement. The full predictor
> $$
> m_x(\delta)=f(x)+\langle g,\delta\rangle
> $$
> is affine because of the constant term $f(x)$. Thus the **increment** has a linear model, while the nearby **function value** has an affine model.

> [!note]- Where does the definition come from?
> In one variable, differentiability means
> $$
> f(x+h)=f(x)+f'(x)h+o(|h|).
> $$
> The first-order term is a linear function of the scalar displacement $h$. In several variables the displacement becomes $\delta\in\mathbb R^n$, so the first-order term must be a scalar-valued linear functional $L_x(\delta)$. Every such functional on finite-dimensional Euclidean space has a unique vector representation,
> $$
> L_x(\delta)=\langle g,\delta\rangle.
> $$
> This linear functional is the Fréchet derivative; the proofs below establish that its representing vector is unique and equals $\nabla f(x)$.

> [!note]- Differentiability immediately implies continuity
> From the expansion and Cauchy–Schwarz,
> $$
> |f(x+\delta)-f(x)|
> \le \|g\|\,\|\delta\|+|r_x(\delta)|.
> $$
> Both terms tend to zero with $\delta$, so $f(x+\delta)\to f(x)$. This proves the implication used in §1.1:
> $$
> \text{differentiable at }x\quad\Longrightarrow\quad\text{continuous at }x.
> $$

> [!tip] What differentiability buys — and what it does not
> Differentiability supplies a qualitative local model. At a point with $\nabla f(x)\neq0$, that model predicts descent for a sufficiently small negative-gradient step, but it does not quantify how small the step must be or provide one step size safe across all iterates. $L$-smoothness (§1.7) will turn this qualitative little-$o$ statement into a uniform quadratic bound.

The next two arguments prove that the vector representing this local linear model is unique and that its coordinates are precisely the partial derivatives.

> [!abstract] Proof 1 — the representing vector is unique
> Suppose two vectors $g$ and $\tilde g$ both satisfy the differentiability expansion:
> $$
> \begin{aligned}
> f(x+\delta)&=f(x)+g^\top\delta+r_g(\delta),\\
> f(x+\delta)&=f(x)+\tilde g^\top\delta+r_{\tilde g}(\delta),
> \end{aligned}
> $$
> where $r_g(\delta)=o(\|\delta\|)$ and $r_{\tilde g}(\delta)=o(\|\delta\|)$. Subtracting gives
> $$
> (g-\tilde g)^\top\delta
> =r_{\tilde g}(\delta)-r_g(\delta).
> $$
> The right side is still little-$o$, because
> $$
> \frac{|r_{\tilde g}(\delta)-r_g(\delta)|}{\|\delta\|}
> \le
> \frac{|r_{\tilde g}(\delta)|}{\|\delta\|}
> +
> \frac{|r_g(\delta)|}{\|\delta\|}
> \longrightarrow0.
> $$
> Write $w:=g-\tilde g$. If $w\neq0$, test the disagreement along its own direction by setting $\delta=tw$ with $t\to0^+$:
> $$
> w^\top(tw)=t\|w\|^2,
> \qquad
> \|tw\|=t\|w\|.
> $$
> Hence
> $$
> \frac{|w^\top(tw)|}{\|tw\|}=\|w\|,
> $$
> a positive constant rather than a quantity tending to zero. This contradiction forces $w=0$, so
> $$
> \boxed{g=\tilde g.}
> $$
> A nonzero linear map cannot be $o(\|\delta\|)$: moving along the direction that represents its disagreement exposes it at first order.

> [!abstract] Proof 2 — the vector is the gradient
> Start from the unique local model
> $$
> f(x+\delta)=f(x)+g^\top\delta+r_x(\delta),
> \qquad
> r_x(\delta)=o(\|\delta\|).
> $$
> To isolate component $g_i$, choose the coordinate displacement $\delta=h e_i$. Then
> $$
> g^\top(h e_i)=h g_i,
> \qquad
> \|h e_i\|=|h|,
> $$
> so
> $$
> \frac{f(x+h e_i)-f(x)}{h}
> =g_i+\frac{r_x(h e_i)}{h}.
> $$
> The remainder vanishes after division:
> $$
> \left|\frac{r_x(h e_i)}{h}\right|
> =\frac{|r_x(h e_i)|}{\|h e_i\|}
> \longrightarrow0.
> $$
> Taking $h\to0$ therefore yields
> $$
> g_i
> =\lim_{h\to0}\frac{f(x+h e_i)-f(x)}{h}
> =\frac{\partial f}{\partial x_i}(x).
> $$
> This holds for every coordinate, hence
> $$
> \boxed{g=\nabla f(x).}
> $$

The two proofs use complementary test directions:

| Goal | Chosen displacement | Why it works |
|---|---|---|
| expose disagreement between two models | $\delta=t(g-\tilde g)$ | tests the difference along its own direction |
| extract one component of the model | $\delta=h e_i$ | isolates coordinate $i$ |

> [!tip] A reusable proof pattern
> To expose an entire vector, test along the vector itself. To extract one component, test along a standard basis vector.

> [!summary] What has actually been proved
> The argument is one-way:
> $$
> f\text{ differentiable at }x
> \quad\Longrightarrow\quad
> \begin{cases}
> \text{the first-order linear model is unique},\\
> \text{all partial derivatives exist at }x,\\
> g=\nabla f(x).
> \end{cases}
> $$
> It does **not** prove that the existence of all partial derivatives implies differentiability. The missing ingredient is uniform control of the remainder over all directions, exactly as the degree-zero homogeneous counterexample in §1.1 demonstrated.

> [!note]- A practical sufficient condition
> A standard theorem supplies a convenient route in the other direction:
> $$
> f\in C^1\text{ near }x
> \quad\Longrightarrow\quad
> f\text{ is differentiable at }x.
> $$
> Here $C^1$ means that all first partial derivatives exist in a neighborhood and are continuous. The useful implication ladder is
> $$
> f\in C^1\text{ near }x
> \Longrightarrow
> f\text{ differentiable at }x
> \Longrightarrow
> \begin{cases}
> f\text{ continuous at }x,\\
> \text{all partial derivatives exist at }x.
> \end{cases}
> $$
> The converses generally fail. Smooth combinations of polynomials, exponentials, and other $C^1$ primitives are usually certified by this theorem rather than by rebuilding the little-$o$ estimate from scratch.

This uniform local model — not merely the list of coordinate partials — is the foundation used by first-order optimization. The tangent-plane example below makes its remainder visible and then verifies the all-directions condition directly.

**A tangent-plane example.** Consider

$$
f(x_1,x_2)=x_1^2+2x_2^2,
\qquad
x=(1,1)^\top.
$$

At the base point,

$$
f(x)=3,
\qquad
\nabla f(x)=\begin{bmatrix}2\\4\end{bmatrix}.
$$

Writing a nearby point as $y=x+\delta$, the affine first-order model is the tangent plane

$$
\begin{aligned}
T_x(y)
&=f(x)+\nabla f(x)^\top(y-x)\\
&=3+2(y_1-1)+4(y_2-1)\\
&=2y_1+4y_2-3.
\end{aligned}
$$

It matches both value and first-order slope at the contact point:

$$
T_x(x)=f(x)=3,
\qquad
\nabla T_x=\nabla f(x)=\begin{bmatrix}2\\4\end{bmatrix}.
$$

![tangent-plane-and-quadratic-remainder]({{ '/assets/notes/gradient-descent/tangent-plane-and-quadratic-remainder.png' | relative_url }})
*The blue surface and orange tangent plane share value and slope at $x=(1,1)$; along the highlighted path $\delta=(h,-h)$, their vertical gap is the quadratic remainder $3h^2$. The gap vanishes relative to the distance travelled, even though it is nonzero at every nonzero step.*

> [!example]- Watch the remainder shrink along one path
> Take
> $$
> \delta=(h,-h)^\top.
> $$
> The tangent plane predicts
> $$
> f(x)+\nabla f(x)^\top\delta=3-2h,
> $$
> while the exact value is
> $$
> f(x+\delta)=(1+h)^2+2(1-h)^2=3-2h+3h^2.
> $$
> Hence
> $$
> r_x(\delta)=3h^2,
> \qquad
> \|\delta\|=\sqrt2\,|h|,
> $$
> where the absolute value is required because the limit is two-sided. Numerically:
>
> | $\delta$ | $\|\delta\|$ | tangent-plane prediction | true value | $\lvert r_x\rvert/\|\delta\|$ |
> |---|---:|---:|---:|---:|
> | $(0.1,-0.1)$ | $0.1414$ | $2.8$ | $2.83$ | $0.2121$ |
> | $(0.05,-0.05)$ | $0.0707$ | $2.9$ | $2.9075$ | $0.1061$ |
> | $(0.025,-0.025)$ | $0.0354$ | $2.95$ | $2.951875$ | $0.0530$ |
>
> Halving $|h|$ halves the displacement, quarters the quadratic remainder, and therefore halves the relative error:
> $$
> \frac{|r_x(\delta)|}{\|\delta\|}
> =\frac{3}{\sqrt2}|h|
> \longrightarrow0.
> $$

> [!warning] One path illustrates; a uniform bound proves
> The calculation above shows the model working along one ray. Differentiability requires the same relative-error conclusion for **every** small displacement. A single failed path can disprove a multivariable limit, but one successful path cannot prove it.

For an arbitrary displacement $\delta=(\delta_1,\delta_2)^\top$,

$$
\begin{aligned}
f((1,1)+\delta)
&=(1+\delta_1)^2+2(1+\delta_2)^2\\
&=3+2\delta_1+4\delta_2
+\underbrace{\delta_1^2+2\delta_2^2}_{r_x(\delta)}.
\end{aligned}
$$

The remainder admits a direction-independent bound:

$$
0\le r_x(\delta)
=\delta_1^2+2\delta_2^2
\le2(\delta_1^2+\delta_2^2)
=2\|\delta\|^2.
$$

After division by $\|\delta\|$,

$$
0\le\frac{|r_x(\delta)|}{\|\delta\|}
\le2\|\delta\|
\xrightarrow[\delta\to0]{}0.
$$

This uniform estimate — not the single-path table — proves differentiability.

| Device | Role |
|---|---|
| one-ray numerical table | makes the shrinking remainder visible |
| exact general remainder $r_x(\delta)=\delta_1^2+2\delta_2^2$ | exposes the error in every direction |
| uniform bound $r_x(\delta)\le2\|\delta\|^2$ | proves the multivariable little-$o$ limit |

> [!note]- Hessian preview: curvature is the remainder
> For this quadratic,
> $$
> \nabla^2f=\begin{bmatrix}2&0\\0&4\end{bmatrix},
> $$
> and the remainder is exactly
> $$
> r_x(\delta)=\frac12\delta^\top\nabla^2f\,\delta.
> $$
> The gradient controls the tangent plane's tilt; the Hessian controls how the surface bends away from it. Sections §1.7–§1.8 turn this curvature viewpoint into the uniform quadratic upper bound used by gradient descent.

> [!tip] What little-$o$ does not mean
> Here $r_x(h,-h)=3h^2>0$ at every nonzero step. The statement $r_x(\delta)=o(\|\delta\|)$ does not make the tangent plane exact at finite distance; it says only that its error becomes negligible **relative to the distance travelled** as $\delta\to0$.

---

## 1.3 The directional derivative

> [!definition] Directional derivative
> For any direction vector $v\in\mathbb R^n$, the **directional derivative** of $f$ at $x$ along $v$ is
> $$
> D_vf(x)
> :=\lim_{h\to0}\frac{f(x+hv)-f(x)}{h},
> $$
> whenever the limit exists. This is the ordinary derivative at $h=0$ of the one-variable slice $h\mapsto f(x+hv)$. A partial derivative is the special case
> $$
> D_{e_i}f(x)=\frac{\partial f}{\partial x_i}(x).
> $$

> [!note] Direction vector versus unit direction
> For general $v$,
> $$
> \|hv\|=|h|\,\|v\|,
> $$
> so $D_vf(x)$ reflects both the ray and the scale assigned to its direction vector. To compare geometric directions fairly, normalize
> $$
> u=\frac{v}{\|v\|},
> \qquad
> \|u\|=1.
> $$
> Then $|h|$ is the actual distance travelled along $x+hu$. Arbitrary $v$ is natural for the definition; unit $u$ is natural when asking which direction is steepest.

> [!abstract] Proof — differentiability gives every directional derivative at once
> If $f$ is differentiable at $x$, then
> $$
> f(x+\delta)
> =f(x)+\nabla f(x)^\top\delta+r_x(\delta),
> \qquad
> r_x(\delta)=o(\|\delta\|).
> $$
> Fix $v\neq0$ and substitute $\delta=hv$:
> $$
> f(x+hv)
> =f(x)+h\nabla f(x)^\top v+r_x(hv).
> $$
> After subtracting $f(x)$ and dividing by $h\neq0$,
> $$
> \frac{f(x+hv)-f(x)}{h}
> =\nabla f(x)^\top v+\frac{r_x(hv)}{h}.
> $$
> The last term vanishes, because
> $$
> \left|\frac{r_x(hv)}{h}\right|
> =\frac{|r_x(hv)|}{\|hv\|}\,\|v\|
> \longrightarrow0.
> $$
> Therefore
> $$
> \boxed{D_vf(x)=\nabla f(x)^\top v.}
> $$
> The case $v=0$ gives $D_0f(x)=0$ directly.

The coordinate form is a weighted sum of the partial derivatives:

$$
D_vf(x)
=\sum_{i=1}^n\frac{\partial f}{\partial x_i}(x)v_i.
$$

Because this is an inner product, differentiability makes the map $v\mapsto D_vf(x)$ linear:

$$
D_{\alpha v+\beta w}f(x)
=\alpha D_vf(x)+\beta D_wf(x).
$$

> [!tip] One vector, infinitely many slopes
> There are infinitely many directions, but a differentiable function needs only one gradient vector to encode all of their first-order rates:
> $$
> v\longmapsto\nabla f(x)^\top v.
> $$
> The direction $v$ is the query; the inner product returns the scalar slope.

For a unit direction $u$, let $\theta$ be its angle with $\nabla f(x)$. Then

$$
\boxed{
D_uf(x)
=\|\nabla f(x)\|\cos\theta
}
$$

and the geometry becomes immediate:

| Angle $\theta$ | $D_uf(x)$ | First-order behavior | Used in |
|---:|---:|---|---:|
| $0^\circ$ | $+\|\nabla f(x)\|$ | fastest increase | §1.5 |
| $90^\circ$ | $0$ | no first-order change | §1.4 |
| $180^\circ$ | $-\|\nabla f(x)\|$ | fastest decrease | §1.5 |

> [!warning]- Stronger trap — even all directional derivatives may not imply differentiability
> Define
> $$
> f(x,y)=
> \begin{cases}
> \dfrac{x^3}{x^2+y^2}, &(x,y)\neq(0,0),\\[5pt]
> 0, &(x,y)=(0,0).
> \end{cases}
> $$
> It is continuous at the origin, since
> $$
> |f(x,y)|\le|x|\longrightarrow0.
> $$
> For every nonzero direction $v=(a,b)$,
> $$
> f(ta,tb)=t\frac{a^3}{a^2+b^2},
> \qquad
> D_vf(0,0)=\frac{a^3}{a^2+b^2},
> $$
> so every directional derivative exists. But the directional-derivative map is not linear:
> $$
> D_{(1,0)}f=1,
> \qquad
> D_{(0,1)}f=0,
> \qquad
> D_{(1,1)}f=\frac12\neq1+0.
> $$
> Hence no single linear model can represent all directions, and $f$ is not differentiable at the origin.
>
> | Counterexample at the origin | Continuous? | All partials? | All directional derivatives? | Differentiable? |
> |---|---:|---:|---:|---:|
> | $xy/(x^2+y^2)$ from §1.1 | no | yes | no | no |
> | $x^3/(x^2+y^2)$ | yes | yes | yes | no |
>
> Even linearity of all pointwise directional derivatives, by itself, is not a substitute for the uniform little-$o$ remainder required by Fréchet differentiability. Existence along each fixed line is pointwise-in-direction information; differentiability is uniform-in-direction control.

The identity $D_vf(x)=\nabla f(x)^\top v$ is therefore a consequence of **differentiability**, not merely of having partial or directional derivatives. It is the bridge to the next two results: zero directional change characterizes tangent directions to level sets (§1.4), while maximizing and minimizing the inner product identifies the steepest directions (§1.5).

---

## 1.4 Geometry: the gradient is orthogonal to level sets

> [!definition] Level set and regular point
> For $c\in\mathbb R$, the **level set** of $f:\mathbb R^n\to\mathbb R$ is
> $$
> L_c:=\{x\in\mathbb R^n:f(x)=c\}.
> $$
> A point $x\in L_c$ is **regular** when
> $$
> \nabla f(x)\neq0.
> $$
> By the implicit function theorem, $L_c$ is locally a smooth $(n-1)$-dimensional hypersurface near every regular point: a contour curve when $n=2$, an ordinary surface when $n=3$, and a hypersurface in higher dimensions.

> [!abstract] Proof — the gradient annihilates every tangent direction
> Let $\gamma$ be a smooth curve confined to the level set, with
> $$
> \gamma(t_0)=x,
> \qquad
> f(\gamma(t))=c.
> $$
> Its velocity
> $$
> \gamma'(t_0)
> =\lim_{s\to0}\frac{\gamma(t_0+s)-\gamma(t_0)}{s}
> $$
> is tangent to $L_c$ at $x$. Differentiating the constant-value constraint by the chain rule gives
> $$
> 0
> =\frac{d}{dt}f(\gamma(t))\bigg|_{t=t_0}
> =\nabla f(x)^\top\gamma'(t_0).
> $$
> At a regular point, every tangent vector is the velocity of such a curve. Hence $\nabla f(x)$ is orthogonal to every tangent direction.

The precise tangent- and normal-space statements are

$$
\boxed{
T_xL_c
=\ker\big(\nabla f(x)^\top\big)
=\{v\in\mathbb R^n:\nabla f(x)^\top v=0\}
}
$$

and

$$
\boxed{
N_xL_c
=\operatorname{span}\{\nabla f(x)\}.
}
$$

Thus “the gradient is perpendicular to the level set” means that it is perpendicular to the level set's **tangent space at the point**.

![gradient-normal-to-level-set]({{ '/assets/notes/gradient-descent/gradient-normal-to-level-set.png' | relative_url }})
*For $f(x_1,x_2)=x_1^2+2x_2^2$, the parameter-space gradient at $x=(1,1)$ is normal to the ellipse $f=3$. The tangent vector $v=(2,-1)$ satisfies $\nabla f(x)^\top v=0$; $\nabla f$ crosses toward higher contours and $-\nabla f$ toward lower ones.*

> [!example]- The ellipse from §1.2
> Continue with
> $$
> f(x_1,x_2)=x_1^2+2x_2^2,
> \qquad
> x=(1,1),
> \qquad
> f(x)=3.
> $$
> The level set through $x$ is the ellipse
> $$
> x_1^2+2x_2^2=3,
> $$
> and
> $$
> \nabla f(x)=\begin{bmatrix}2\\4\end{bmatrix}.
> $$
> A tangent vector $v=(v_1,v_2)^\top$ must satisfy
> $$
> 2v_1+4v_2=0.
> $$
> Choosing $v=(2,-1)^\top$ gives
> $$
> \nabla f(x)^\top v
> =\begin{bmatrix}2&4\end{bmatrix}
> \begin{bmatrix}2\\-1\end{bmatrix}
> =0.
> $$

> [!tip] Tangent means no first-order change
> A tangent displacement $v\in T_xL_c$ satisfies
> $$
> D_vf(x)=\nabla f(x)^\top v=0.
> $$
> It need not keep $f$ exactly constant after a finite straight step; it says the change vanishes to first order. A curve that remains on the level set bends as necessary to preserve the value exactly.

To determine orientation, normalize the nonzero gradient:

$$
D_{\nabla f/\|\nabla f\|}f(x)
=\|\nabla f(x)\|>0,
\qquad
D_{-\nabla f/\|\nabla f\|}f(x)
=-\|\nabla f(x)\|<0.
$$

So $\nabla f$ points toward larger values and $-\nabla f$ toward smaller ones. Section §1.5 strengthens “up” and “down” to **steepest** ascent and descent.

> [!warning] What does $\nabla f(x)=0$ actually mean?
> The regular-level-set theorem no longer applies, and the gradient supplies no normal direction. The geometry must be inspected separately; zero gradient does **not** by itself imply that the level set has no geometric normal.
>
> | Function and zero level set | Geometry at the origin |
> |---|---|
> | $f(x,y)=x^2-y^2$, so $L_0=\{y=\pm x\}$ | crossing lines; no unique tangent or normal |
> | $f(x,y)=y^3$, so $L_0=\{y=0\}$ | a smooth line with a geometric normal, but $\nabla f=0$ fails to reveal it |
> | $f(x,y)=x^2+y^2$, so $L_0=\{(0,0)\}$ | an isolated point, not an $(n-1)$-dimensional level surface |
>
> In the saddle example, the identity $\nabla f^\top v=0$ degenerates to $0=0$ and carries no directional information. The correct conclusion at a critical point is therefore **“this theorem is silent,”** not automatically “no normal exists.”

---

## 1.5 The steepest-ascent theorem

> [!important] Theorem — Euclidean steepest directions
> Let $f:\mathbb R^n\to\mathbb R$ be differentiable at $x$ and write $g=\nabla f(x)$. If $g\neq0$, then among all Euclidean unit directions,
>
> $$
> \begin{aligned}
> \max_{\|u\|_2=1}D_uf(x)&=\|g\|_2,
> &\arg\max_{\|u\|_2=1}D_uf(x)&=\left\{\frac{g}{\|g\|_2}\right\},\\[2mm]
> \min_{\|u\|_2=1}D_uf(x)&=-\|g\|_2,
> &\arg\min_{\|u\|_2=1}D_uf(x)&=\left\{-\frac{g}{\|g\|_2}\right\}.
> \end{aligned}
> $$
>
> Therefore $g/\|g\|_2$ is the unit direction of steepest ascent and $-g/\|g\|_2$ is the unit direction of steepest descent.

> [!abstract] Proof — orthogonal decomposition
> By §1.3, differentiability gives
>
> $$
> D_uf(x)=\langle g,u\rangle.
> $$
>
> For any $\|u\|_2=1$, decompose $g$ into its projection onto $u$ and an orthogonal remainder:
>
> $$
> g=\langle g,u\rangle u+r,
> \qquad
> r:=g-\langle g,u\rangle u.
> $$
>
> The remainder is perpendicular to $u$ because
>
> $$
> \langle r,u\rangle
> =\langle g,u\rangle-\langle g,u\rangle\|u\|_2^2
> =0.
> $$
>
> Pythagoras therefore gives
>
> $$
> \|g\|_2^2
> =\langle g,u\rangle^2+\|r\|_2^2
> \quad\Longrightarrow\quad
> -\|g\|_2\le\langle g,u\rangle\le\|g\|_2.
> $$
>
> Equality in either bound requires $r=0$, so $u$ must be parallel or antiparallel to $g$. The sign selects the extremum:
>
> $$
> \langle g,u\rangle=\|g\|_2
> \Longleftrightarrow
> u=\frac{g}{\|g\|_2},
> \qquad
> \langle g,u\rangle=-\|g\|_2
> \Longleftrightarrow
> u=-\frac{g}{\|g\|_2}.
> $$
>
> This is Cauchy–Schwarz together with its equality condition. $\blacksquare$

![steepest-euclidean-direction]({{ '/assets/notes/gradient-descent/steepest-euclidean-direction.png' | relative_url }})
*A chosen unit direction $u$ ends on the unit circle, while its projection component $(g^\top u)u$ may extend farther along the same ray. The residual $r$ completes the vector sum and is perpendicular to $u$. The two points $\pm g/\|g\|_2$ are the unique extremizing directions when $g\neq0$.*

> [!warning] At a stationary point, the optimizer is not a single direction
> If $g=\nabla f(x)=0$, then $D_uf(x)=0$ for every unit $u$. Hence
>
> $$
> \arg\max_{\|u\|_2=1}D_uf(x)
> =\arg\min_{\|u\|_2=1}D_uf(x)
> =\{u:\|u\|_2=1\}.
> $$
>
> The first-order model does not prefer any direction. This does **not** determine whether $x$ is a minimum, maximum, saddle point, or a point whose behavior only appears at higher order.

> [!important] Unit direction versus the gradient-descent step
> When $g\neq0$, the theorem solves a unit-direction problem:
>
> $$
> u_{\mathrm{sd}}=-\frac{g}{\|g\|_2}.
> $$
>
> Gradient descent instead takes the finite step
>
> $$
> s=-\eta g
> =\eta\|g\|_2\,u_{\mathrm{sd}}.
> $$
>
> The normalized vector chooses the **direction**; the factor $\eta\|g\|_2$ chooses the **step length**. The next section studies this finite update rather than only its infinitesimal direction.

> [!warning] "Steepest" depends on how step length is measured
> The theorem above uses the Euclidean constraint $\|u\|_2=1$. A different norm produces a different unit ball and can therefore select a different steepest direction. The raw negative gradient is not intrinsically the steepest direction independently of geometry.

> [!note]- Extension — dual norms and non-Euclidean steepest descent
> This extension uses finite-dimensional norm duality, positive-definite matrices, and elementary quadratic optimization. It is not required for the first reading of ordinary gradient descent.
>
> Let $g=\nabla f(x)$. For an arbitrary norm $\|\cdot\|$, its **dual norm** is
>
> $$
> \boxed{\ \|g\|_*:=\max_{\|u\|\le 1}\langle g,u\rangle\ }.
> $$
>
> Thus the dual norm is exactly the largest first-order increase available inside one unit of the chosen geometry. By symmetry of the unit ball,
>
> $$
> \min_{\|u\|\le 1}\langle g,u\rangle=-\|g\|_*.
> $$
>
> The original norm constrains the **step** $u$; the dual norm measures the size of the **linear functional** $u\mapsto\langle g,u\rangle$. This is the general form of the Euclidean argument above.
>
> For $g\neq0$, the following table shows representative minimizers.
>
> | Step geometry | A representative unit steepest-descent direction |
> |---|---|
> | $\|u\|_2\le1$ | $-g/\|g\|_2$ |
> | $\|u\|_\infty\le1$ | $-\operatorname{sign}(g)$ |
> | $\|u\|_1\le1$ | $-\operatorname{sign}(g_j)e_j$, where $j\in\arg\max_i|g_i|$ |
> | $\|u\|_A:=\sqrt{u^\top Au}\le1$ | $-A^{-1}g/\sqrt{g^\top A^{-1}g}$ |
>
> In the $\ell_1$ row the maximizer may be nonunique when several coordinates tie. In the $\ell_\infty$ row, coordinates for which $g_i=0$ may be chosen arbitrarily in $[-1,1]$.
>
> #### Quadratic norms
>
> Suppose $A$ is symmetric positive definite and
>
> $$
> \|u\|_A=\sqrt{u^\top Au}.
> $$
>
> The $A$-unit steepest-descent problem is
>
> $$
> \min_{u^\top Au=1}\langle g,u\rangle.
> $$
>
> With $w=A^{1/2}u$, the constraint becomes $\|w\|_2=1$ and
>
> $$
> \langle g,u\rangle
> =\langle A^{-1/2}g,w\rangle.
> $$
>
> Applying the Euclidean theorem to $w$ gives
>
> $$
> \boxed{
> u_A^\star
> =-\frac{A^{-1}g}{\sqrt{g^\top A^{-1}g}},
> \qquad
> \|g\|_{A,*}=\sqrt{g^\top A^{-1}g}
> }.
> $$
>
> The matrix $A$ changes what counts as a unit step: a Euclidean sphere becomes an ellipsoid. Consequently the steepest ray changes from $-g$ to $-A^{-1}g$.
>
> #### Why the unnormalized direction $-A^{-1}g$ appears
>
> Consider the local quadratic model
>
> $$
> m(s)=f(x)+\langle g,s\rangle+\frac12s^\top As.
> $$
>
> Differentiating with respect to $s$ and setting the result to zero yields
>
> $$
> \nabla_s m(s)=g+As=0
> \quad\Longrightarrow\quad
> \boxed{\ s^\star=-A^{-1}g\ }.
> $$
>
> The normalized vector $u_A^\star$ answers **which $A$-unit direction is steepest**; the unnormalized step $s^\star$ answers **which step minimizes the quadratic model**. They lie on the same ray but solve different optimization problems. This distinction is the bridge to preconditioning and Newton-type methods.

---

## Connections

- The partial derivatives and gradient represented by this local linear model are constructed in [From Learning Objectives to Gradients]({{ '/notes/mit6-7960-01-1-from-learning-objectives-to-gradients/' | relative_url }}).
- Turning the Euclidean steepest direction into a controlled finite update leads to [Gradient Descent, Smoothness, and the Descent Lemma]({{ '/notes/mit6-7960-01-3-gradient-descent-smoothness-and-the-descent-lemma/' | relative_url }}).
