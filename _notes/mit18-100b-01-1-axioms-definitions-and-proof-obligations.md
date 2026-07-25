---
layout: note
title: "Axioms, Definitions, and Proof Obligations"
source_title: "MIT18.100B-01.1-Axioms, Definitions, and Proof Obligations"
course: "MIT 18.100B"
sequence: "01.1"
source_context: "Spring 2025, Lecture 1 and Problem Set 1, Problems 1 and 4"
permalink: "/notes/mit18-100b-01-1-axioms-definitions-and-proof-obligations/"
tags:
  - "math/real-analysis"
  - "math/algebra/ordered-fields"
  - "math/order-theory"
  - "topic/proof-writing"
---
Elementary algebra often conceals its logical dependencies. Identities such as

$$
(-1)(-1)=1
$$

feel computationally obvious, and statements such as “one polynomial is eventually larger than another” feel as though they should automatically define an order. In an axiomatic argument, however, neither familiarity nor plausibility is sufficient. Every conclusion must be traced to an axiom, a previously proved lemma, or a definition whose full set of obligations has been verified.

Two recurring proof failures arise at this stage:

1. an axiom proves that an object exists, but the proof silently identifies that object with a particular candidate;
2. an argument proves that two alternatives cannot both hold, but does not prove that at least one alternative must hold.

The first failure appears when proving \(1>0\) in an ordered field. The second appears when verifying an eventual-positivity order on integer polynomials. Together they illustrate a general principle:

> [!important] Definitions and axioms create obligations
> An existential axiom requires both an available witness and a justification that a proposed object is that witness.
>
> A statement that “exactly one” alternative holds requires both **exhaustiveness** and **mutual exclusivity**.

## 1. Ordered-field axioms as a dependency system

Let \(\mathbb F\) be an ordered field with \(1\neq0\). The relevant algebraic objects are:

- \(0\), the additive identity;
- \(1\), the multiplicative identity;
- \(-a\), an additive inverse of \(a\);
- \(a^{-1}\), a multiplicative inverse of a nonzero \(a\);
- the strict order relation \(<\).

The notation does not itself prove any identity. For example, \(-1\) is introduced as an additive inverse of \(1\):

$$
1+(-1)=0.
$$

This tells us what \(-1\) does under addition. It does not directly tell us what \(-1\) does under multiplication.

Similarly, the multiplicative-inverse axiom M5 says

$$
\forall a\in\mathbb F,\qquad
a\neq0
\Longrightarrow
\exists b\in\mathbb F
\text{ such that }ab=1.
$$

The witness \(b\) may be denoted by \(a^{-1}\) or \(1/a\). For \(a=-1\), M5 gives

$$
\exists b\in\mathbb F
\quad\text{such that}\quad
(-1)b=1.
$$

It does not yet say that \(b=-1\).

This distinction is the logical center of Problem 1.

## 2. Problem 1: proving \(1>0\)

> [!question] Original statement — Problem 1
> Let \(\mathbb F\) be an ordered field with \(1\neq0\). Show that \(1>0\).
>  
> **Hint:** Show \((-1)(-1)=1\) first.

The target is

$$
\boxed{0<1.}
$$

Order axiom O1 and the hypothesis \(1\neq0\) reduce the proof to two alternatives:

$$
0<1
\qquad\text{or}\qquad
1<0.
$$

To eliminate the second alternative, the suggested intermediate result is

$$
\boxed{(-1)(-1)=1.}
$$

### 2.1 Why the multiplicative-inverse argument is circular

A tempting argument is:

> Since \(-1\neq0\), axiom M5 gives a multiplicative inverse of \(-1\). Therefore \((-1)(-1)=1\).

The first sentence is valid. The second does not follow. M5 supplies some element \(b\) satisfying

$$
(-1)b=1,
$$

but the desired equation requires the additional identification

$$
b=-1.
$$

That identification is precisely what remains to be proved.

Giving the existential witness the name \(1/(-1)\) changes no part of the logic. The notation records its defining property,

$$
(-1)\frac1{-1}=1,
$$

but it does not establish

$$
\frac1{-1}=-1.
$$

Indeed, proving \(1/(-1)=-1\) would normally be done by first showing that \(-1\) itself satisfies the defining equation. Using M5 to assert that equation reverses the dependency.

> [!warning] Existence is not identification
> From
> $$
> \exists b,\qquad (-1)b=1,
> $$
> one cannot infer
> $$
> (-1)(-1)=1
> $$
> without separately proving that the witness \(b\) equals \(-1\).

The algebraic intuition was correct: \(-1\) is its own multiplicative inverse. The defect was not the intended conclusion but the unsupported choice of witness.

### 2.2 A foundational lemma: multiplication by zero

Before proving \((-1)(-1)=1\), derive the identity

$$
\boxed{a\cdot0=0}
$$

for every \(a\in\mathbb F\).

Because \(0\) is the additive identity,

$$
0+0=0.
$$

Multiplying by \(a\) and using distributivity gives

$$
a\cdot0
=
a(0+0)
=
a\cdot0+a\cdot0.
$$

Add the additive inverse of \(a\cdot0\) to both sides:

$$
a\cdot0+\bigl(-(a\cdot0)\bigr)
=
(a\cdot0+a\cdot0)+\bigl(-(a\cdot0)\bigr).
$$

The left side is \(0\). Associativity on the right gives

$$
\begin{aligned}
0
&=
a\cdot0+
\left(
a\cdot0+\bigl(-(a\cdot0)\bigr)
\right)\\
&=
a\cdot0+0\\
&=
a\cdot0.
\end{aligned}
$$

Therefore

$$
\boxed{a\cdot0=0.}
$$

This familiar identity is a theorem derived from the field axioms, not one of the listed axioms itself.

### 2.3 Deriving \((-1)(-1)=1\)

By definition of the additive inverse,

$$
1+(-1)=0.
$$

Multiply this equality by \(-1\). Using the zero-product lemma and distributivity,

$$
\begin{aligned}
0
&=(-1)\cdot0\\
&=(-1)\bigl(1+(-1)\bigr)\\
&=(-1)\cdot1+(-1)(-1).
\end{aligned}
$$

The multiplicative identity axiom gives

$$
1\cdot(-1)=-1.
$$

By commutativity,

$$
(-1)\cdot1=-1.
$$

Hence

$$
0=-1+(-1)(-1).
$$

Add \(1\) to both sides:

$$
\begin{aligned}
1
&=1+0\\
&=1+\bigl(-1+(-1)(-1)\bigr)\\
&=\bigl(1+(-1)\bigr)+(-1)(-1)\\
&=0+(-1)(-1)\\
&=(-1)(-1).
\end{aligned}
$$

Thus

$$
\boxed{(-1)(-1)=1.}
$$

No multiplicative inverse was used. The proof depends only on additive inverses, the multiplicative identity, commutativity, distributivity, and the previously derived zero-product lemma.

### 2.4 Eliminating the possibility \(1<0\)

By O1, exactly one of

$$
0<1,\qquad 0=1,\qquad 1<0
$$

holds. The hypothesis \(1\neq0\) eliminates the middle alternative.

Assume for contradiction that

$$
1<0.
$$

Apply the order-and-addition axiom OA, adding \(-1\) to both sides:

$$
1+(-1)<0+(-1).
$$

Therefore

$$
0<-1.
$$

Both factors \(-1\) are now positive. The order-and-multiplication axiom OM gives

$$
0<(-1)(-1).
$$

Using the algebraic identity just proved,

$$
0<1.
$$

The assumption \(1<0\) and the conclusion \(0<1\) imply \(1<1\) by transitivity O2, contradicting O1. Therefore \(1<0\) is impossible. The only remaining alternative is

$$
\boxed{0<1.}
$$

> [!summary] Dependency chain for Problem 1
> $$
> \begin{aligned}
> \text{field axioms}
> &\Longrightarrow a\cdot0=0,\\
> a\cdot0=0\text{ and distributivity}
> &\Longrightarrow (-1)(-1)=1,\\
> (-1)(-1)=1\text{ and the order axioms}
> &\Longrightarrow 0<1.
> \end{aligned}
> $$
>
> The order argument is short only after its algebraic dependency has been established without circularity.

## 3. Problem 4: eventual positivity as an order

> [!question] Original statement — Problem 4
> Let \(\mathbb M\) be the set of polynomials with integer coefficients:
> $$
> \mathbb M
> :=
> \left\{
> f(x)=a_0+a_1x+a_2x^2+\cdots+a_nx^n
> \;\middle|\;
> a_i\in\mathbb Z
> \right\}.
> $$
> Define \(0\prec f\) when \(f(x)>0\) for all sufficiently large \(x\). More precisely,
> $$
> 0\prec f
> \iff
> \exists M>0\;\forall x>M,\quad f(x)>0.
> $$
> Then define \(f\prec g\) if \(0\prec g-f\). Show that \((\mathbb M,\prec)\) is an ordered set by verifying O1 and O2.
>
> The following fact may be used directly: if
> $$
> f(x)=a_0+a_1x+\cdots+a_nx^n
> $$
> has \(a_n>0\), then \(0\prec f\).

Let \(\mathbb M=\mathbb Z[x]\) be the set of polynomials with integer coefficients:

$$
\mathbb M
=
\left\{
f(x)=a_0+a_1x+\cdots+a_nx^n
\;\middle|\;
a_i\in\mathbb Z
\right\}.
$$

First define

$$
0\prec f
$$

to mean that \(f\) is positive for all sufficiently large real inputs. Formally,

$$
0\prec f
\iff
\exists M>0\;
\forall x\in\mathbb R,\qquad
x>M\Longrightarrow f(x)>0.
$$

Then define

$$
f\prec g
\iff
0\prec(g-f).
$$

Equivalently,

$$
f\prec g
\iff
\exists M>0\;
\forall x>M,\qquad
g(x)-f(x)>0.
$$

Thus \(f\prec g\) means that \(g(x)\) eventually lies strictly above \(f(x)\). It does not require \(g(x)>f(x)\) for every real \(x\), nor does it compare the coefficient lists coordinate by coordinate.

To show that \((\mathbb M,\prec)\) is an ordered set, one must verify O1 and O2:

- for every \(f,g\in\mathbb M\), exactly one of
  \[
  f\prec g,\qquad f=g,\qquad g\prec f
  \]
  holds;
- if \(f\prec g\) and \(g\prec h\), then \(f\prec h\).

### 3.1 “Exactly one” contains two separate claims

The phrase “exactly one” decomposes as follows:

$$
\boxed{
\text{exactly one}
=
\text{at least one}
+
\text{at most one}.
}
$$

For the three alternatives in O1, this means:

1. **Exhaustiveness:** at least one of \(f\prec g\), \(f=g\), or \(g\prec f\) holds.
2. **Mutual exclusivity:** no two of them can hold simultaneously.

Proving only that the alternatives cannot coexist establishes “at most one.” It leaves open the possibility that none of them holds. That missing possibility is the main logical gap in the attempted proof.

### 3.2 The eventual sign of a nonzero polynomial

The problem permits the following fact:

> [!theorem] Eventual sign is determined by the leading coefficient
> If
> $$
> p(x)=c_0+c_1x+\cdots+c_nx^n
> $$
> has \(c_n>0\), then
> $$
> \exists M>0\;\forall x>M,\qquad p(x)>0.
> $$
> In the notation above, \(0\prec p\).

The crucial coefficient is not merely some positive coefficient. It is the coefficient of the highest nonzero power.

For example,

$$
p(x)=x-100x^2
$$

has a positive coefficient of \(x\), but its leading coefficient is \(-100\). Therefore \(p(x)\) is negative for all sufficiently large positive \(x\). Finding one positive coefficient does not determine the eventual sign.

> [!note]- Why the leading-coefficient fact is true
> Suppose \(n\ge1\), \(c_n>0\), and define
> $$
> C=\sum_{i=0}^{n-1}|c_i|.
> $$
> For \(x\ge1\), every \(x^i\) with \(i<n\) satisfies \(x^i\le x^{n-1}\). Hence
> $$
> \begin{aligned}
> p(x)
> &=c_nx^n+\sum_{i=0}^{n-1}c_ix^i\\
> &\ge c_nx^n-\sum_{i=0}^{n-1}|c_i|x^i\\
> &\ge c_nx^n-Cx^{n-1}\\
> &=x^{n-1}(c_nx-C).
> \end{aligned}
> $$
> Choose
> $$
> M>\max\left\{1,\frac{C}{c_n}\right\}.
> $$
> If \(x>M\), then \(x^{n-1}>0\) and \(c_nx-C>0\), so \(p(x)>0\). The constant case \(n=0\) is immediate.

This estimate formalizes the statement that the highest-degree term eventually dominates all lower-degree terms.

### 3.3 Exhaustiveness: proving that at least one alternative holds

Fix \(f,g\in\mathbb M\).

If \(f=g\), the equality alternative holds.

Now suppose \(f\neq g\). Define the difference polynomial

$$
d=g-f.
$$

Because \(f\neq g\), the polynomial \(d\) is nonzero. Write

$$
d(x)=c_0+c_1x+\cdots+c_kx^k,
$$

where

$$
c_k\neq0
$$

and \(k\) is the greatest index with a nonzero coefficient. Thus \(c_k\) is the leading coefficient of \(d\).

Because \(c_k\) is a nonzero integer, exactly one of the following holds:

$$
c_k>0
\qquad\text{or}\qquad
c_k<0.
$$

If \(c_k>0\), the leading-coefficient fact shows that

$$
0\prec d.
$$

Since \(d=g-f\), this is exactly

$$
f\prec g.
$$

If \(c_k<0\), then

$$
-d=f-g
$$

has positive leading coefficient \(-c_k\). Therefore

$$
0\prec(-d)=f-g,
$$

which is exactly

$$
g\prec f.
$$

Consequently, whenever \(f\neq g\), one of \(f\prec g\) and \(g\prec f\) holds. Including the equality case proves exhaustiveness:

$$
\boxed{
f\prec g
\quad\text{or}\quad
f=g
\quad\text{or}\quad
g\prec f.
}
$$

This is the step that cannot be replaced by proving only that the alternatives are incompatible.

### 3.4 Mutual exclusivity: proving that at most one holds

First suppose \(f=g\). Then

$$
g-f=0.
$$

The zero polynomial is not eventually strictly positive, so \(f\prec g\) is false. Similarly, \(g\prec f\) is false. Equality is therefore incompatible with either strict relation.

It remains to rule out the possibility that both \(f\prec g\) and \(g\prec f\) hold.

Suppose they did. By the definition of \(f\prec g\), there would exist \(M_1>0\) such that

$$
x>M_1
\Longrightarrow
g(x)-f(x)>0.
$$

By the definition of \(g\prec f\), there would exist \(M_2>0\) such that

$$
x>M_2
\Longrightarrow
f(x)-g(x)>0.
$$

Choose

$$
M=\max\{M_1,M_2\}.
$$

For every \(x>M\), both inequalities would hold. Adding them gives

$$
\bigl(g(x)-f(x)\bigr)
+
\bigl(f(x)-g(x)\bigr)
>
0+0,
$$

and hence

$$
0>0,
$$

a contradiction. Therefore \(f\prec g\) and \(g\prec f\) cannot both hold.

Combining this with the equality case proves mutual exclusivity. Together with exhaustiveness, it establishes O1.

> [!important] What the incomplete argument proved
> Showing
> $$
> f\prec g
> \Longrightarrow
> \neg(f=g)\land\neg(g\prec f)
> $$
> is useful, but it proves only mutual exclusivity.
>
> O1 additionally requires a construction showing that when \(f\neq g\), the leading coefficient of \(g-f\) forces one of the two strict comparisons.

### 3.5 Transitivity

Suppose

$$
f\prec g
\qquad\text{and}\qquad
g\prec h.
$$

Then there exist \(M_1,M_2>0\) such that

$$
x>M_1
\Longrightarrow
g(x)-f(x)>0
$$

and

$$
x>M_2
\Longrightarrow
h(x)-g(x)>0.
$$

Set

$$
M=\max\{M_1,M_2\}.
$$

If \(x>M\), both inequalities hold simultaneously. Therefore

$$
\begin{aligned}
h(x)-f(x)
&=\bigl(h(x)-g(x)\bigr)
+\bigl(g(x)-f(x)\bigr)\\
&>0+0\\
&=0.
\end{aligned}
$$

Thus \(h-f\) is eventually positive, which means

$$
f\prec h.
$$

Therefore \(\prec\) satisfies O2.

The use of \(\max\{M_1,M_2\}\) is a reusable pattern. Each hypothesis supplies its own eventual threshold; taking the maximum creates one region on which all required eventual statements hold simultaneously.

### 3.6 Conclusion

O1 follows from exhaustiveness and mutual exclusivity. O2 follows by combining the two eventual inequalities beyond a common threshold. Hence

$$
\boxed{(\mathbb M,\prec)\text{ is an ordered set}.}
$$

This order is controlled by the leading coefficient of the difference polynomial. More explicitly, for distinct \(f\) and \(g\),

$$
f\prec g
\iff
\text{the leading coefficient of }g-f\text{ is positive}.
$$

Thus the order is equivalent to comparing the highest-degree coefficient at which the two polynomials differ. Lower-degree behavior matters only after all higher-degree coefficients have cancelled.

## 4. Shared proof architecture

Problems 1 and 4 appear to concern different objects, but their logical structures are closely related.

| Problem | Available statement | Missing obligation |
|---|---|---|
| Multiplicative inverse | some \(b\) satisfies \((-1)b=1\) | prove that the desired candidate is \(b=-1\) |
| Order trichotomy | no two alternatives can coexist | prove that at least one alternative occurs |
| Eventual inequalities | each statement has its own threshold | construct a common threshold |
| Polynomial coefficients | some coefficient has a sign | identify the highest nonzero coefficient |

### 4.1 Witnesses must be constructed or identified

An existential statement has the form

$$
\exists y,\qquad P(y).
$$

It permits the use of an unspecified witness satisfying \(P\). It does not permit the replacement of that witness by a preferred candidate \(z\). To use \(z\), one must prove

$$
P(z).
$$

In Problem 1, M5 supplies some inverse of \(-1\), but proving that \(-1\) itself is the inverse requires the independent algebraic derivation

$$
(-1)(-1)=1.
$$

### 4.2 “Exactly one” must be split before proving it

For propositions \(A,B,C\), “exactly one holds” contains two directions of work:

$$
A\lor B\lor C
$$

and

$$
\neg(A\land B),\qquad
\neg(A\land C),\qquad
\neg(B\land C).
$$

The first line is exhaustiveness. The second group is mutual exclusivity. Neither implies the other.

In Problem 4, the leading coefficient proves exhaustiveness; contradiction between two eventual inequalities proves mutual exclusivity.

### 4.3 Quantifiers determine the proof move

The definition

$$
f\prec g
\iff
\exists M>0\;\forall x>M,\qquad
g(x)-f(x)>0
$$

contains an existential threshold followed by a universal tail condition.

To prove such a statement, one must:

1. construct a threshold \(M\);
2. take an arbitrary \(x>M\);
3. establish the desired inequality.

To combine two such statements, preserve both witnesses and replace them by a common one:

$$
M=\max\{M_1,M_2\}.
$$

Dropping the thresholds and writing only that a polynomial is “positive for large \(x\)” may preserve intuition, but it hides the exact step required in proofs of transitivity and incompatibility.

> [!summary] Proof-obligation checklist
> Before declaring an axiomatic or definition-based proof complete, ask:
>
> 1. What is the exact quantified statement I may use?
> 2. Does it give existence, uniqueness, or both?
> 3. If I selected a particular witness, did I verify it?
> 4. If the target says “exactly one,” did I prove both existence and exclusivity?
> 5. If several eventual statements are used, did I combine their thresholds?
> 6. If a polynomial’s eventual sign matters, did I identify the highest nonzero coefficient?
> 7. Can every displayed equality or implication be traced to an axiom, definition, or proved lemma?

The purpose of axiomatic analysis is not to distrust familiar mathematics. It is to expose which facts are primitive, which are derived, and which logical obligation turns an intuitive argument into a proof.
