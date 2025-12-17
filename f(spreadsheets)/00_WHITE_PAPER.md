# f(Spreadsheets): A Tensor-Algebraic Foundation for Tabular Computation

## Spreadsheet Tensor Theory (STT) — Complete Mathematical Framework

**Version 1.0**  
**Auto-Workspace-AI / Intent Tensor Theory Institute**  
**Authors:** Achilles, Claude (Anthropic), Contributors to the Intent Tensor Theory

---

# Abstract

We present **Spreadsheet Tensor Theory (STT)**, a formal mathematical framework that reconceptualizes spreadsheets as rank-3 tensors over a heterogeneous value space. This work establishes that every spreadsheet is intrinsically a tensor object $\mathbf{S} \in V^{N_s \times N_r \times N_c}$, where standard operations (SUM, VLOOKUP, SUMIF, pivot tables) correspond to well-defined tensor contractions, slices, and masked reductions.

We prove four fundamental theorems: (1) the **Acyclicity Theorem**, establishing that well-formed spreadsheets correspond to nilpotent dependency matrices; (2) the **Fixed Point Theorem**, guaranteeing unique evaluation convergence; (3) the **Parallelization Theorem**, enabling O(K) sequential complexity regardless of cell count; and (4) the **Change Propagation Theorem**, bounding incremental recalculation.

We demonstrate that STT is a discrete instantiation of **Intent Tensor Theory (ITT)**, mapping the continuous collapse geometry of $\nabla\Phi$ fields onto discrete cell references. The Operational Value Tensor $V = A_{ijk} R^i I^j E^k$ projects directly onto spreadsheet structure, making business mathematics computable in standard tools.

Reference implementations in Python and JavaScript are provided, along with analysis showing that production-grade "meta-scripting" patterns already implement STT principles without naming them.

**Keywords:** tensor algebra, spreadsheet computation, dependency graphs, business mathematics, Intent Tensor Theory, formal semantics

---

# 1. Introduction

## 1.1 The Hidden Mathematical Structure

Spreadsheets are the world's most widely deployed computational substrate. Over 750 million users interact with spreadsheet software, collectively representing the largest installed base of any programming paradigm. Yet despite their ubiquity, spreadsheets have been treated primarily as *tools* rather than as *mathematical objects* worthy of rigorous formalization.

We argue this is a fundamental oversight.

A spreadsheet is not merely a grid of cells. It is a **rank-3 tensor** with:
- **Sheet dimension** (depth): organizing parallel computational spaces
- **Row dimension** (vertical): indexing entity instances
- **Column dimension** (horizontal): indexing attributes and measures

This tensor structure is not metaphorical. It is literal, and it has profound implications for how we understand, optimize, and extend spreadsheet computation.

## 1.2 Contributions

This paper makes the following contributions:

1. **Formal tensor definition** of spreadsheets with complete value space specification
2. **Recursive self-similarity axiom** proving scale-invariant structure
3. **Dependency tensor** encoding formula references as rank-6 binary tensor
4. **Four fundamental theorems** with complete proofs
5. **Operational mapping** of standard functions to tensor operations
6. **Bridge to Intent Tensor Theory** connecting discrete computation to continuous collapse geometry
7. **Reference implementations** in Python and JavaScript
8. **Analysis of production patterns** demonstrating implicit STT in existing code

## 1.3 Relation to Prior Work

Existing spreadsheet formalization falls into four categories:

| Approach | Representative Work | Limitation |
|----------|-------------------|------------|
| Operational Semantics | Sestoft et al. (2020) | Evaluation rules without algebraic structure |
| Type Systems | Erwig & Abraham (2006) | Error detection without computational model |
| Functional Programming | Peyton Jones et al. (2003) | Language design without tensor formalization |
| Category Theory | Baylor et al. (2022) | Composition without dimensional analysis |

**No prior work treats spreadsheets as tensor-algebraic objects.** This represents a genuine gap in the literature, which STT addresses.

---

# 2. Formal Definitions

## 2.1 The Spreadsheet Tensor

**Definition 2.1.1 (Spreadsheet Tensor).**
A spreadsheet is a rank-3 tensor over a value space $V$:

$$\boxed{\mathbf{S} \in V^{N_s \times N_r \times N_c}}$$

Where:
- $N_s \in \mathbb{N}^+$ is the sheet count (depth dimension)
- $N_r \in \mathbb{N}^+$ is the row count (vertical dimension)
- $N_c \in \mathbb{N}^+$ is the column count (horizontal dimension)
- $V$ is the heterogeneous value space

Cell access uses mixed index notation:

$$S^{s}_{rc} \equiv \mathbf{S}[s][r][c] \equiv \mathbf{S}_{s,r,c}$$

**Definition 2.1.2 (Value Space).**
The value space is a tagged union with type discriminator:

$$V = \mathbb{R} \cup \Sigma^* \cup \mathbb{B} \cup \mathcal{E} \cup \{\emptyset\}$$

Where:
- $\mathbb{R}$ = real numbers (including integers as subset)
- $\Sigma^* = \bigcup_{n=0}^{\infty} \Sigma^n$ = strings over Unicode alphabet $\Sigma$
- $\mathbb{B} = \{\top, \bot\}$ = boolean values
- $\mathcal{E} = \{\texttt{\#REF!}, \texttt{\#VALUE!}, \texttt{\#DIV/0!}, \texttt{\#NAME?}, \texttt{\#NULL!}, \texttt{\#N/A}, \texttt{\#NUM!}\}$ = error states
- $\emptyset$ = null/empty cell (unit type)

**Remark.** The value space is *not* a ring or field — it lacks closure under standard operations. This heterogeneity is essential to spreadsheet semantics and distinguishes STT from pure numerical tensor frameworks.

## 2.2 The Formula Tensor

**Definition 2.2.1 (Formula Language).**
Let $\mathcal{L}$ be the language of valid spreadsheet formulas, defined by the grammar:

$$\mathcal{L} ::= \text{ref} \mid \text{lit} \mid \text{func}(\mathcal{L}^*) \mid \mathcal{L} \oplus \mathcal{L}$$

Where:
- $\text{ref}$ = cell reference (absolute or relative)
- $\text{lit}$ = literal value in $V$
- $\text{func}$ = built-in or user-defined function
- $\oplus$ = binary operator (+, -, *, /, &, etc.)

**Definition 2.2.2 (Formula Tensor).**
The formula tensor parallels the value tensor:

$$\mathbf{F} \in (\mathcal{L} \cup \{\emptyset\})^{N_s \times N_r \times N_c}$$

A cell contains either a formula ($F^s_{rc} \in \mathcal{L}$) or is a literal value ($F^s_{rc} = \emptyset$).

**Definition 2.2.3 (Reference Extraction).**
Define the reference extraction function:

$$\text{refs}: \mathcal{L} \rightarrow \mathcal{P}(\mathbb{N}^3)$$

That returns the set of all cell coordinates referenced by a formula.

## 2.3 The Dependency Tensor

**Definition 2.3.1 (Dependency Tensor).**
The dependency structure is a rank-6 binary tensor:

$$\mathbf{D} \in \{0,1\}^{N_s \times N_r \times N_c \times N_s \times N_r \times N_c}$$

With entries:

$$D^{s_1 s_2}_{r_1 c_1 r_2 c_2} = \begin{cases} 
1 & \text{if } (s_2, r_2, c_2) \in \text{refs}(F^{s_1}_{r_1 c_1}) \\
0 & \text{otherwise}
\end{cases}$$

**Interpretation:** $D^{s_1 s_2}_{r_1 c_1 r_2 c_2} = 1$ means cell $(s_1, r_1, c_1)$ directly references cell $(s_2, r_2, c_2)$.

**Definition 2.3.2 (Flattened Dependency Matrix).**
For computational tractability, define the linearization:

$$\phi: \mathbb{N}^3 \rightarrow \mathbb{N}, \quad \phi(s, r, c) = s \cdot (N_r \cdot N_c) + r \cdot N_c + c$$

The flattened dependency matrix is:

$$\mathbf{A} \in \{0,1\}^{M \times M}, \quad M = N_s \cdot N_r \cdot N_c$$

With $A_{ij} = D^{s_1 s_2}_{r_1 c_1 r_2 c_2}$ where $i = \phi(s_1, r_1, c_1)$ and $j = \phi(s_2, r_2, c_2)$.

---

# 3. The Recursive Self-Similarity Axiom

## 3.1 Structural Recursion

**Axiom 3.1.1 (Self-Similar Decomposition).**
The spreadsheet tensor admits recursive decomposition at each rank:

$$\mathbf{S} = \bigoplus_{s=1}^{N_s} \mathbf{S}^{(s)}$$

Where $\mathbf{S}^{(s)} \in V^{N_r \times N_c}$ is the rank-2 sheet tensor:

$$\mathbf{S}^{(s)} = \bigoplus_{r=1}^{N_r} \mathbf{S}^{(s)}_r$$

Where $\mathbf{S}^{(s)}_r \in V^{N_c}$ is the rank-1 row tensor:

$$\mathbf{S}^{(s)}_r = \bigoplus_{c=1}^{N_c} S^{s}_{rc}$$

Where $S^{s}_{rc} \in V$ is the rank-0 cell (scalar).

The direct sum $\bigoplus$ denotes structural composition preserving independence.

## 3.2 Scale Invariance

**Definition 3.2.1 (Projection Operator).**
Let $\Pi_k$ be the projection onto rank-$k$ substructure:

$$\Pi_3(\mathbf{S}) = \mathbf{S}$$
$$\Pi_2(\mathbf{S}) = \{\mathbf{S}^{(s)}\}_{s=1}^{N_s}$$
$$\Pi_1(\mathbf{S}) = \{\mathbf{S}^{(s)}_r\}_{s,r}$$
$$\Pi_0(\mathbf{S}) = \{S^s_{rc}\}_{s,r,c}$$

**Theorem 3.2.2 (Scale Invariance).**
The containment relation is scale-invariant:

$$\Pi_k(\mathbf{S}) \cong \Pi_k(\Pi_{k+1}(\mathbf{S}))$$

**Proof.**
The structure of a sheet within a workbook (how $\mathbf{S}^{(s)}$ sits in $\mathbf{S}$) is isomorphic to the structure of a row within a sheet (how $\mathbf{S}^{(s)}_r$ sits in $\mathbf{S}^{(s)}$). Both are indexed collections of lower-rank tensors with independent entries. The isomorphism is given by the index relabeling $s \mapsto r$, $r \mapsto c$. ∎

## 3.3 The Recursive Constructor

**Definition 3.3.1 (Rank-n Array Constructor).**
Define the recursive array constructor:

$$\mathcal{R}^{(n)}: \text{Type} \times \mathbb{N} \rightarrow \text{Type}$$
$$\mathcal{R}^{(n)}(T, d) = T^d$$

**Theorem 3.3.2 (Recursive Structure Equation).**
The spreadsheet tensor satisfies:

$$\boxed{\mathbf{S} = \mathcal{R}^{(3)}\left(\mathcal{R}^{(2)}\left(\mathcal{R}^{(1)}(V, N_c), N_r\right), N_s\right)}$$

Expanding:
- $\mathcal{R}^{(1)}(V, N_c) = V^{N_c}$ — row as array of values
- $\mathcal{R}^{(2)}(V^{N_c}, N_r) = (V^{N_c})^{N_r}$ — sheet as array of rows
- $\mathcal{R}^{(3)}((V^{N_c})^{N_r}, N_s) = ((V^{N_c})^{N_r})^{N_s}$ — workbook as array of sheets

**Proof.**
By associativity of Cartesian product:
$$((V^{N_c})^{N_r})^{N_s} \cong V^{N_c \times N_r \times N_s} \cong V^{N_s \times N_r \times N_c}$$

The recursive and flat constructions are definitionally equivalent. ∎

---

# 4. Fundamental Theorems

## 4.1 The Acyclicity Theorem

**Theorem 4.1.1 (Acyclicity Condition).**
A spreadsheet is well-formed if and only if its dependency matrix is nilpotent:

$$\boxed{\text{well-formed}(\mathbf{S}) \iff \exists k \in \mathbb{N} : \mathbf{A}^k = \mathbf{0}}$$

**Proof.**

($\Rightarrow$) Assume the spreadsheet is well-formed, meaning it contains no circular references.

The dependency graph $G = (V, E)$ where $V = \{1, \ldots, M\}$ (linearized cells) and $E = \{(i,j) : A_{ij} = 1\}$ is therefore a directed acyclic graph (DAG).

Every DAG admits a topological ordering: a bijection $\sigma: V \rightarrow \{1, \ldots, M\}$ such that $(i,j) \in E \implies \sigma(i) < \sigma(j)$.

In this ordering, the adjacency matrix $\mathbf{A}$ becomes strictly upper triangular (all nonzero entries above the diagonal).

For any strictly upper triangular $M \times M$ matrix, we have $\mathbf{A}^M = \mathbf{0}$.

**Proof of sub-claim:** Let $\mathbf{A}$ be strictly upper triangular. Then $(\mathbf{A}^k)_{ij}$ counts paths of length exactly $k$ from $i$ to $j$. Any path in a DAG increases the topological index at each step. A path of length $M$ would require $M+1$ distinct vertices, but only $M$ exist. Thus $\mathbf{A}^M = \mathbf{0}$. ∎ (sub-claim)

($\Leftarrow$) Assume $\exists k : \mathbf{A}^k = \mathbf{0}$.

Suppose for contradiction that there exists a cycle: vertices $v_1 \rightarrow v_2 \rightarrow \cdots \rightarrow v_m \rightarrow v_1$.

Then for any $n \in \mathbb{N}$, there is a path of length $nm$ from $v_1$ to $v_1$ (traverse the cycle $n$ times).

Thus $(\mathbf{A}^{nm})_{v_1, v_1} \geq 1$ for all $n$.

This contradicts $\mathbf{A}^k = \mathbf{0}$ for sufficiently large $nm > k$.

Therefore no cycle exists, and the spreadsheet is well-formed. ∎

**Corollary 4.1.2.** The minimum $k$ such that $\mathbf{A}^k = \mathbf{0}$ equals the longest path length in the dependency graph plus one.

## 4.2 The Fixed Point Theorem

**Definition 4.2.1 (Evaluation Operator).**
The evaluation operator maps formula and state tensors to a new state tensor:

$$\mathcal{E}: (\mathcal{L} \cup \{\emptyset\})^{N_s \times N_r \times N_c} \times V^{N_s \times N_r \times N_c} \rightarrow V^{N_s \times N_r \times N_c}$$

Defined cell-wise:

$$\mathcal{E}(\mathbf{F}, \mathbf{S})^s_{rc} = \begin{cases}
S^s_{rc} & \text{if } F^s_{rc} = \emptyset \\
\text{eval}(F^s_{rc}, \mathbf{S}) & \text{otherwise}
\end{cases}$$

Where $\text{eval}$ interprets the formula given current cell values.

**Definition 4.2.2 (Dependency Level).**
For cell $i$ (linearized index), define the level function:

$$\ell(i) = \begin{cases}
0 & \text{if } \sum_j A_{ij} = 0 \\
1 + \max_{j: A_{ij}=1} \ell(j) & \text{otherwise}
\end{cases}$$

Let $K = \max_i \ell(i)$ be the maximum dependency depth.

Let $\mathbf{L}_k = \{i : \ell(i) = k\}$ be the level-$k$ cell set.

**Theorem 4.2.3 (Fixed Point Convergence).**
For a well-formed spreadsheet, evaluation converges to a unique fixed point:

$$\boxed{\mathbf{S}^* = \lim_{n \to \infty} \mathcal{E}^n(\mathbf{F}, \mathbf{S}_0) \quad \text{satisfying} \quad \mathcal{E}(\mathbf{F}, \mathbf{S}^*) = \mathbf{S}^*}$$

**Proof.**

*Existence:*

By Theorem 4.1.1, the dependency graph is a DAG.

Construct $\mathbf{S}^*$ by level-order evaluation:

**Base case ($k=0$):** Cells in $\mathbf{L}_0$ have no dependencies. They are either:
- Literals: $S^{*s}_{rc} = S^s_{rc}$ (preserved)
- Formulas with no references: $S^{*s}_{rc} = \text{eval}(F^s_{rc}, \cdot)$ (deterministic)

**Inductive step ($k > 0$):** Assume $\mathbf{S}^*$ is defined for all cells in $\mathbf{L}_0 \cup \cdots \cup \mathbf{L}_{k-1}$.

For cell $i \in \mathbf{L}_k$, all referenced cells $j$ satisfy $\ell(j) < k$ (by definition of level).

Thus $S^*_j$ is already defined for all dependencies of cell $i$.

Define $S^*_i = \text{eval}(F_i, \mathbf{S}^*)$ — this is well-defined since all referenced values exist.

After $K+1$ iterations, all cells have defined values, yielding $\mathbf{S}^*$.

*Fixed Point Property:*

We verify $\mathcal{E}(\mathbf{F}, \mathbf{S}^*) = \mathbf{S}^*$.

For any cell $i$:
- If $F_i = \emptyset$: $\mathcal{E}(\mathbf{F}, \mathbf{S}^*)_i = S^*_i$ ✓
- If $F_i \in \mathcal{L}$: $\mathcal{E}(\mathbf{F}, \mathbf{S}^*)_i = \text{eval}(F_i, \mathbf{S}^*) = S^*_i$ by construction ✓

*Uniqueness:*

Let $\mathbf{S}'$ be another fixed point.

**Base case:** For $i \in \mathbf{L}_0$, $S'_i = S^*_i$ (both determined by literals or constant formulas).

**Inductive step:** Assume $S'_j = S^*_j$ for all $j \in \mathbf{L}_0 \cup \cdots \cup \mathbf{L}_{k-1}$.

For $i \in \mathbf{L}_k$:
$$S'_i = \text{eval}(F_i, \mathbf{S}') = \text{eval}(F_i, \mathbf{S}^*) = S^*_i$$

The second equality holds because $F_i$ only references cells in lower levels, where $\mathbf{S}' = \mathbf{S}^*$ by hypothesis.

By induction, $\mathbf{S}' = \mathbf{S}^*$. ∎

## 4.3 The Parallelization Theorem

**Theorem 4.3.1 (Level-Parallel Evaluation).**
Cells at the same dependency level can be evaluated in parallel:

$$\boxed{\mathbf{S}^*|_{\mathbf{L}_k} = \text{parallel\_map}(\lambda i. \text{eval}(F_i, \mathbf{S}^*|_{\mathbf{L}_{<k}}), \mathbf{L}_k)}$$

**Proof.**

Let $i, j \in \mathbf{L}_k$ be distinct cells at level $k$.

*Claim:* $i$ does not depend on $j$ and $j$ does not depend on $i$.

*Proof of claim:* Suppose $A_{ij} = 1$ (cell $i$ references cell $j$). Then $\ell(i) \geq \ell(j) + 1$ by definition of level. But $\ell(i) = \ell(j) = k$, contradiction. Similarly for $A_{ji} = 1$. ∎ (claim)

Thus the evaluations of $i$ and $j$ share no read-write dependencies — they only read from levels $< k$ (already computed) and write to distinct locations.

By the principle of parallel independence, all cells in $\mathbf{L}_k$ can be evaluated concurrently. ∎

**Corollary 4.3.2 (Complexity Bound).**
Evaluation requires at most $K + 1$ sequential steps, regardless of total cell count $M$.

$$T_{parallel} = O(K) \quad \text{vs} \quad T_{sequential} = O(M)$$

For typical spreadsheets with shallow dependency depth ($K \ll M$), this represents massive parallelization potential.

## 4.4 The Change Propagation Theorem

**Definition 4.4.1 (Transitive Closure).**
The transitive closure of dependency matrix $\mathbf{A}$ is:

$$\mathbf{A}^+ = \sum_{k=1}^{K} \mathbf{A}^k$$

Where $(\mathbf{A}^+)_{ij} > 0$ iff there exists a directed path from $i$ to $j$.

**Theorem 4.4.2 (Affected Cell Set).**
Given a change at cells $C \subseteq \{1, \ldots, M\}$, the affected cells requiring recalculation are:

$$\boxed{\text{affected}(C) = \{j : \exists i \in C, (\mathbf{A}^T)^+_{ij} > 0\} = \{j : (\mathbf{A}^T)^+ \mathbf{1}_C \neq 0\}_j}$$

Where $\mathbf{A}^T$ is the transpose (reversing edge direction to find dependents rather than dependencies).

**Proof.**

Cell $j$ needs recalculation iff its value depends (directly or transitively) on some changed cell $i \in C$.

$j$ depends on $i$ iff there is a path from $j$ to $i$ in the dependency graph, i.e., $(\mathbf{A}^+)_{ji} > 0$.

Equivalently, there is a path from $i$ to $j$ in the *reversed* graph, i.e., $((\mathbf{A}^T)^+)_{ij} > 0$.

Thus:
$$\text{affected}(C) = \bigcup_{i \in C} \{j : ((\mathbf{A}^T)^+)_{ij} > 0\}$$

In matrix form, this is the support of $(\mathbf{A}^T)^+ \mathbf{1}_C$. ∎

**Corollary 4.4.3 (Incremental Complexity).**
Recalculation after changes to $C$ requires evaluating only $|\text{affected}(C)|$ cells, not all $M$ cells.

---

# 5. Operations as Tensor Transformations

## 5.1 Slice Operations

**Definition 5.1.1 (Tensor Slicing).**
Standard spreadsheet range selections correspond to tensor slices:

| Operation | Notation | Type Signature |
|-----------|----------|----------------|
| Sheet selection | $\mathbf{S}[s, :, :]$ | $V^{N_s \times N_r \times N_c} \rightarrow V^{N_r \times N_c}$ |
| Row selection | $\mathbf{S}[s, r, :]$ | $V^{N_s \times N_r \times N_c} \rightarrow V^{N_c}$ |
| Column selection | $\mathbf{S}[s, :, c]$ | $V^{N_s \times N_r \times N_c} \rightarrow V^{N_r}$ |
| Cell selection | $\mathbf{S}[s, r, c]$ | $V^{N_s \times N_r \times N_c} \rightarrow V$ |
| Range selection | $\mathbf{S}[s, r_1:r_2, c_1:c_2]$ | $V^{N_s \times N_r \times N_c} \rightarrow V^{(r_2-r_1+1) \times (c_2-c_1+1)}$ |

## 5.2 Contraction Operations (Aggregations)

**Definition 5.2.1 (Tensor Contraction).**
Aggregation functions perform tensor contraction — summing over indices:

$$\text{SUM}(\mathbf{S}[s, r_1:r_2, c]) = \sum_{r=r_1}^{r_2} S^s_{rc} = S^s_{ic} \delta^i_{[r_1, r_2]}$$

Where $\delta^i_{[r_1, r_2]} = 1$ if $i \in [r_1, r_2]$, else $0$.

**Definition 5.2.2 (General Aggregation).**
All standard aggregations are contractions with different combining functions:

| Function | Tensor Operation | Combining Operator |
|----------|-----------------|-------------------|
| SUM | $\sum_i S_i$ | Addition |
| PRODUCT | $\prod_i S_i$ | Multiplication |
| COUNT | $\sum_i 1$ | Cardinality |
| AVERAGE | $(\sum_i S_i) / (\sum_i 1)$ | Mean |
| MAX | $\max_i S_i$ | Supremum |
| MIN | $\min_i S_i$ | Infimum |

## 5.3 Indexed Selection (VLOOKUP/INDEX-MATCH)

**Definition 5.3.1 (VLOOKUP as Tensor Operation).**

$$\text{VLOOKUP}(v, \mathbf{S}[s, :, c_1:c_n], k) = S^s_{r^* c_k}$$

Where the match index is:
$$r^* = \min\{r : S^s_{rc_1} = v\}$$

**Tensor decomposition:**

1. **Match tensor:** $M_r = \mathbb{1}[S^s_{rc_1} = v]$ (indicator function)
2. **First-match selection:** $r^* = \arg\min_r \{r : M_r = 1\}$
3. **Value extraction:** Result $= S^s_{r^* c_k}$

This is a **conditional slice** — slicing based on computed predicate.

## 5.4 Masked Contraction (SUMIF/COUNTIF)

**Definition 5.4.1 (SUMIF as Masked Tensor Contraction).**

$$\text{SUMIF}(\mathbf{S}[s,:,c_{cond}], \text{cond}, \mathbf{S}[s,:,c_{sum}]) = \sum_r M_r \cdot S^s_{r,c_{sum}}$$

Where the mask tensor is:
$$M_r = \mathbb{1}[\text{cond}(S^s_{r,c_{cond}})]$$

**Generalization:** All conditional aggregations follow this pattern:

$$\text{AGGIF}(\text{range}, \text{cond}, \text{values}) = \text{AGG}_{i: \text{cond}(range_i)} \text{values}_i$$

This is **masked contraction** — the fundamental pattern of conditional computation.

## 5.5 Pivot Tables as Tensor Restructuring

**Definition 5.5.1 (Pivot Operation).**
A pivot table performs:

1. **Group-by:** Partition rows by unique values in key columns
2. **Aggregate:** Apply contraction within each partition
3. **Reshape:** Restructure into new tensor dimensions

Let $\mathbf{S} \in V^{N_r \times N_c}$ with key column $c_k$ having $U$ unique values.

The pivot produces $\mathbf{P} \in V^{U \times N_c'}$ where:

$$P_{u, c'} = \text{AGG}\{S_{r, c_{val}} : S_{r, c_k} = u\}$$

This is a **dimension exchange** — trading row cardinality for key-indexed structure.

---

# 6. The Master Equations

## 6.1 The State Equation

**Theorem 6.1.1 (Spreadsheet State Dynamics).**
The complete state of a spreadsheet at time $t$ is:

$$\boxed{\mathbf{S}(t) = \mathcal{E}\left(\mathbf{F}, \mathbf{S}(t^-)\right) \cdot (\mathbf{I} - \mathbf{P}) + \mathbf{X}(t) \cdot \mathbf{P}}$$

Where:
- $\mathbf{S}(t) \in V^{N_s \times N_r \times N_c}$ = value tensor at time $t$
- $\mathbf{S}(t^-)$ = value tensor immediately before input event
- $\mathbf{F}$ = formula tensor (static between structural edits)
- $\mathcal{E}$ = evaluation operator
- $\mathbf{X}(t) \in V^{N_s \times N_r \times N_c}$ = external input tensor at time $t$
- $\mathbf{P} \in \{0,1\}^{N_s \times N_r \times N_c}$ = input mask ($P^s_{rc} = 1$ iff cell receives input)
- $\mathbf{I}$ = all-ones tensor (identity for this masking operation)
- Products are Hadamard (element-wise)

**Interpretation:** Each cell either:
- Receives external input: masked by $\mathbf{P}$, value from $\mathbf{X}(t)$
- Computes from formula: masked by $(\mathbf{I} - \mathbf{P})$, value from $\mathcal{E}$

## 6.2 The Computation Equation

**Theorem 6.2.1 (Layered Evaluation).**
For a well-formed spreadsheet with maximum dependency depth $K$:

$$\boxed{\mathbf{S}^* = \sum_{k=0}^{K} \mathcal{E}_k\left(\mathbf{F} \circ \mathbf{L}_k\right)}$$

Where:
- $\mathbf{L}_k \in \{0,1\}^M$ = indicator for level-$k$ cells
- $\mathcal{E}_k$ = evaluation restricted to level-$k$ cells
- $\circ$ = Hadamard product (masking)
- Sum accumulates results across levels

**Interpretation:** Total evaluation decomposes into sequential level evaluations, each parallelizable.

## 6.3 The Recursive Structure Equation

**Theorem 6.3.1 (Self-Similar Construction).**

$$\boxed{\mathbf{S} = \mathcal{R}^{(3)}\left(\mathcal{R}^{(2)}\left(\mathcal{R}^{(1)}(V)\right)\right) \cong V^{N_s \times N_r \times N_c}}$$

**Interpretation:** The spreadsheet is a recursively constructed tensor, with each dimension adding one level of array nesting. This recursive view and the flat tensor view are mathematically equivalent but offer different computational perspectives.

---

# 7. Bridge to Intent Tensor Theory

## 7.1 The Correspondence Principle

**Theorem 7.1.1 (STT-ITT Correspondence).**
Spreadsheet Tensor Theory is a discrete instantiation of Intent Tensor Theory. The mapping is:

| ITT Concept | ITT Symbol | STT Analog | STT Symbol |
|-------------|-----------|------------|------------|
| Scalar potential | $\Phi$ | Cell value | $S^s_{rc}$ |
| Collapse vector | $\nabla\Phi$ | Formula reference | $\text{refs}(F^s_{rc})$ |
| Recursive curl | $\nabla \times \vec{F}$ | Circular reference | $\text{cycle}(\mathbf{A})$ |
| Curvature lock | $\nabla^2\Phi$ | Aggregation function | SUM, AVG |
| Charge emission | $\rho_q$ | Output cell | Result cells |
| Phase memory | $S_\theta$ | Cached values | Memoization |
| Tensor lock | $T_{ij}$ | Fixed point | $\mathbf{S}^*$ |

**Proof sketch.** 
The ITT collapse operator $\nabla\Phi$ represents directional flow of intent. In STT, formula references create directed edges in the dependency graph — this IS directional flow of computational intent.

The ITT recursive curl $\nabla \times \vec{F} \neq 0$ indicates a closed loop in the collapse field. In STT, this corresponds to circular references — cycles in the dependency graph — which are forbidden in standard semantics (well-formedness requires acyclicity).

The ITT curvature lock $\nabla^2\Phi$ represents second-order collapse creating stable structure. In STT, aggregation functions (SUM, AVG) compute second-order properties of cell collections, "locking in" a summary value.

The correspondence is not metaphorical but structural. ∎

## 7.2 Dimensional Mapping

**Definition 7.2.1 (Business Primitive Mapping).**
The spreadsheet tensor dimensions map to organizational primitives:

| Tensor Dimension | Index | Business Interpretation |
|-----------------|-------|------------------------|
| Sheet | $s$ | Business unit, time period, scenario |
| Row | $r$ | Entity instance (employee, transaction, product) |
| Column | $c$ | Attribute, measure, KPI |

**Example:** A three-sheet workbook with:
- Sheet 1: Q1 data
- Sheet 2: Q2 data  
- Sheet 3: Q3 data

This is a tensor $\mathbf{S} \in V^{3 \times N_r \times N_c}$ where the sheet dimension indexes time periods.

## 7.3 The Operational Value Tensor

**Definition 7.3.1 (AIRE Spreadsheet).**
The f(AutoWorkspace) Operational Value Tensor:

$$V = A_{ijk} R^i I^j E^k$$

Can be instantiated in spreadsheet form as:

$$\mathbf{S}_{AIRE} \in V^{N_{unit} \times N_{entity} \times 4}$$

Where the 4 columns represent:
- Column 1: Autonomy score $A$
- Column 2: Intent clarity $I$  
- Column 3: Resource availability $R$
- Column 4: Execution capacity $E$

The Operational Value for entity $(u, e)$ is:

$$V_{ue} = \sum_{c=1}^{4} w_c \cdot S^u_{e,c}$$

Where $\vec{w} = (w_A, w_I, w_R, w_E)$ are component weights.

## 7.4 Intent Cohesion in Spreadsheet Terms

**Definition 7.4.1 (Structural Alignment).**
Define the expected dependency pattern $\mathbf{E}$ based on structural conventions:
- Vertical references (same column): hierarchical rollups
- Horizontal references (same row): entity attribute relationships
- Cross-sheet references: period comparisons

**Definition 7.4.2 (Intent Cohesion Ratio).**
The Intent Cohesion ratio in spreadsheet terms:

$$\boxed{\theta_{cohesion} = \frac{\|\mathbf{D} \cap \mathbf{E}\|}{\|\mathbf{D} - \mathbf{E}\| + \epsilon} = \frac{R_{Align}}{R_{Drift}}}$$

Where:
- $\mathbf{D}$ = actual dependency tensor
- $\mathbf{E}$ = expected dependency tensor
- $\|\cdot\|$ = tensor norm (count of nonzero entries)
- $R_{Align} = \|\mathbf{D} \cap \mathbf{E}\|$ = aligned references
- $R_{Drift} = \|\mathbf{D} - \mathbf{E}\|$ = drifting references
- $\epsilon$ = small constant preventing division by zero

**Interpretation:** High $\theta_{cohesion}$ indicates the spreadsheet follows structural conventions. Low $\theta_{cohesion}$ suggests ad-hoc references that may indicate technical debt or design drift.

**This makes Intent Cohesion a computable metric on any spreadsheet.**

---

# 8. Reference Implementation

## 8.1 Python Implementation

```python
"""
Spreadsheet Tensor Theory — Reference Implementation
Auto-Workspace-AI / Intent Tensor Theory Institute
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import re

class ValueType(Enum):
    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"
    ERROR = "error"
    NULL = "null"

@dataclass
class Cell:
    """Rank-0 tensor element."""
    value: Any
    value_type: ValueType
    formula: Optional[str] = None
    
    @property
    def is_formula(self) -> bool:
        return self.formula is not None

class SpreadsheetTensor:
    """
    Rank-3 tensor representation of a spreadsheet.
    S ∈ V^{N_s × N_r × N_c}
    """
    
    def __init__(self, n_sheets: int, n_rows: int, n_cols: int):
        self.N_s = n_sheets
        self.N_r = n_rows
        self.N_c = n_cols
        self.M = n_sheets * n_rows * n_cols  # Total cells
        
        # Value tensor S
        self._data: Dict[Tuple[int, int, int], Cell] = {}
        
        # Formula tensor F
        self._formulas: Dict[Tuple[int, int, int], str] = {}
        
        # Dependency tensor D (sparse representation)
        self._dependencies: Dict[Tuple[int, int, int], Set[Tuple[int, int, int]]] = {}
        
        # Cached dependency matrix A
        self._dep_matrix: Optional[np.ndarray] = None
        self._dep_matrix_dirty = True
        
        # Level assignments
        self._levels: Dict[Tuple[int, int, int], int] = {}
        self._max_level = 0
    
    # ========== Index Operations ==========
    
    def _linearize(self, s: int, r: int, c: int) -> int:
        """φ(s,r,c) = s·(N_r·N_c) + r·N_c + c"""
        return s * (self.N_r * self.N_c) + r * self.N_c + c
    
    def _delinearize(self, i: int) -> Tuple[int, int, int]:
        """Inverse of linearize."""
        s = i // (self.N_r * self.N_c)
        remainder = i % (self.N_r * self.N_c)
        r = remainder // self.N_c
        c = remainder % self.N_c
        return (s, r, c)
    
    # ========== Cell Access (Tensor Indexing) ==========
    
    def __getitem__(self, key: Tuple[int, int, int]) -> Any:
        """S^s_{rc} access."""
        if key in self._data:
            return self._data[key].value
        return None  # Empty cell
    
    def __setitem__(self, key: Tuple[int, int, int], value: Any):
        """Set cell value (marks as literal, clears formula)."""
        s, r, c = key
        vtype = self._infer_type(value)
        self._data[key] = Cell(value=value, value_type=vtype, formula=None)
        self._dep_matrix_dirty = True
        if key in self._dependencies:
            del self._dependencies[key]
    
    def set_formula(self, key: Tuple[int, int, int], formula: str):
        """Set cell formula and compute dependencies."""
        s, r, c = key
        self._formulas[key] = formula
        
        # Parse references
        refs = self._parse_references(formula)
        self._dependencies[key] = refs
        
        # Mark for recomputation
        self._dep_matrix_dirty = True
        
        # Create cell with formula
        self._data[key] = Cell(value=None, value_type=ValueType.NULL, formula=formula)
    
    # ========== Slice Operations (Definition 5.1) ==========
    
    def sheet(self, s: int) -> 'SheetView':
        """S[s, :, :] — rank reduction 3→2."""
        return SheetView(self, s)
    
    def row(self, s: int, r: int) -> List[Any]:
        """S[s, r, :] — rank reduction 3→1."""
        return [self[s, r, c] for c in range(self.N_c)]
    
    def column(self, s: int, c: int) -> List[Any]:
        """S[s, :, c] — rank reduction 3→1."""
        return [self[s, r, c] for r in range(self.N_r)]
    
    def range(self, s: int, r1: int, r2: int, c1: int, c2: int) -> List[List[Any]]:
        """S[s, r1:r2, c1:c2] — subspace extraction."""
        return [[self[s, r, c] for c in range(c1, c2+1)] for r in range(r1, r2+1)]
    
    # ========== Dependency Analysis (Section 3) ==========
    
    def _parse_references(self, formula: str) -> Set[Tuple[int, int, int]]:
        """Extract cell references from formula."""
        refs = set()
        # Pattern: Sheet1!A1 or A1 or $A$1
        pattern = r"(?:(\w+)!)?\$?([A-Z]+)\$?(\d+)"
        for match in re.finditer(pattern, formula, re.IGNORECASE):
            sheet_name, col_str, row_str = match.groups()
            s = 0  # Default sheet (extend for multi-sheet)
            c = self._col_to_index(col_str)
            r = int(row_str) - 1  # 0-indexed
            if 0 <= r < self.N_r and 0 <= c < self.N_c:
                refs.add((s, r, c))
        return refs
    
    def _col_to_index(self, col: str) -> int:
        """Convert column letter to index (A=0, B=1, ...)."""
        result = 0
        for char in col.upper():
            result = result * 26 + (ord(char) - ord('A') + 1)
        return result - 1
    
    def build_dependency_matrix(self) -> np.ndarray:
        """Construct flattened dependency matrix A."""
        if not self._dep_matrix_dirty and self._dep_matrix is not None:
            return self._dep_matrix
        
        A = np.zeros((self.M, self.M), dtype=np.int8)
        
        for cell, deps in self._dependencies.items():
            i = self._linearize(*cell)
            for dep in deps:
                j = self._linearize(*dep)
                A[i, j] = 1
        
        self._dep_matrix = A
        self._dep_matrix_dirty = False
        return A
    
    def is_well_formed(self) -> bool:
        """Theorem 3.3: Check ∃k: A^k = 0."""
        A = self.build_dependency_matrix()
        A_power = A.copy()
        
        for k in range(1, self.M + 1):
            if np.all(A_power == 0):
                return True
            A_power = A_power @ A
        
        return False
    
    def compute_levels(self) -> Dict[Tuple[int, int, int], int]:
        """Compute dependency levels for all cells."""
        A = self.build_dependency_matrix()
        levels = {}
        
        # Initialize: cells with no dependencies are level 0
        for i in range(self.M):
            if np.sum(A[i, :]) == 0:
                levels[i] = 0
        
        # Iteratively assign levels
        changed = True
        while changed:
            changed = False
            for i in range(self.M):
                if i in levels:
                    continue
                deps = np.where(A[i, :] == 1)[0]
                if all(d in levels for d in deps):
                    levels[i] = 1 + max(levels[d] for d in deps)
                    changed = True
        
        # Convert to coordinate form
        self._levels = {self._delinearize(i): lvl for i, lvl in levels.items()}
        self._max_level = max(levels.values()) if levels else 0
        
        return self._levels
    
    def get_level_sets(self) -> Dict[int, List[Tuple[int, int, int]]]:
        """Return L_k sets for each level k."""
        if not self._levels:
            self.compute_levels()
        
        level_sets = {}
        for cell, lvl in self._levels.items():
            if lvl not in level_sets:
                level_sets[lvl] = []
            level_sets[lvl].append(cell)
        
        return level_sets
    
    # ========== Evaluation (Section 4) ==========
    
    def evaluate(self) -> 'SpreadsheetTensor':
        """
        Fixed Point Evaluation (Theorem 4.3).
        Returns S* satisfying E(F, S*) = S*.
        """
        if not self.is_well_formed():
            raise ValueError("Spreadsheet contains circular references")
        
        level_sets = self.get_level_sets()
        
        # Evaluate level by level
        for k in range(self._max_level + 1):
            cells_at_level = level_sets.get(k, [])
            
            # Theorem 7.1: These can be evaluated in parallel
            for cell in cells_at_level:
                if cell in self._formulas:
                    value = self._eval_formula(self._formulas[cell])
                    vtype = self._infer_type(value)
                    self._data[cell] = Cell(value=value, value_type=vtype, 
                                           formula=self._formulas[cell])
        
        return self
    
    def _eval_formula(self, formula: str) -> Any:
        """Evaluate a formula given current cell values."""
        # Simple evaluator — extend for full formula language
        expr = formula.lstrip('=')
        
        # Replace cell references with values
        pattern = r"(?:(\w+)!)?\$?([A-Z]+)\$?(\d+)"
        
        def replace_ref(match):
            sheet_name, col_str, row_str = match.groups()
            s = 0
            c = self._col_to_index(col_str)
            r = int(row_str) - 1
            val = self[s, r, c]
            if val is None:
                return "0"
            elif isinstance(val, str):
                return f'"{val}"'
            return str(val)
        
        expr = re.sub(pattern, replace_ref, expr, flags=re.IGNORECASE)
        
        # Handle SUM, AVERAGE, etc.
        expr = self._expand_functions(expr)
        
        try:
            return eval(expr)
        except:
            return "#VALUE!"
    
    def _expand_functions(self, expr: str) -> str:
        """Expand spreadsheet functions to Python."""
        # SUM(range) → sum([values])
        expr = re.sub(r'SUM\(([^)]+)\)', r'sum([\1])', expr, flags=re.IGNORECASE)
        expr = re.sub(r'AVERAGE\(([^)]+)\)', r'(sum([\1])/len([\1]))', expr, flags=re.IGNORECASE)
        expr = re.sub(r'COUNT\(([^)]+)\)', r'len([\1])', expr, flags=re.IGNORECASE)
        expr = re.sub(r'MAX\(([^)]+)\)', r'max([\1])', expr, flags=re.IGNORECASE)
        expr = re.sub(r'MIN\(([^)]+)\)', r'min([\1])', expr, flags=re.IGNORECASE)
        return expr
    
    def _infer_type(self, value: Any) -> ValueType:
        """Infer value type."""
        if value is None:
            return ValueType.NULL
        elif isinstance(value, bool):
            return ValueType.BOOLEAN
        elif isinstance(value, (int, float)):
            return ValueType.NUMBER
        elif isinstance(value, str):
            if value.startswith('#'):
                return ValueType.ERROR
            return ValueType.STRING
        return ValueType.NULL
    
    # ========== Change Propagation (Theorem 7.2) ==========
    
    def get_affected_cells(self, changed: Set[Tuple[int, int, int]]) -> Set[Tuple[int, int, int]]:
        """Compute affected(C) using transitive closure."""
        A = self.build_dependency_matrix()
        A_T = A.T  # Transpose to find dependents
        
        # Compute transitive closure A^+
        A_plus = np.zeros_like(A_T, dtype=np.float64)
        A_power = A_T.astype(np.float64)
        
        for _ in range(self.M):
            A_plus += A_power
            A_power = A_power @ A_T
            if np.all(A_power == 0):
                break
        
        # Find affected cells
        affected = set()
        for cell in changed:
            i = self._linearize(*cell)
            for j in range(self.M):
                if A_plus[i, j] > 0:
                    affected.add(self._delinearize(j))
        
        return affected
    
    # ========== Contraction Operations (Section 5.2) ==========
    
    def SUM(self, s: int, r1: int, r2: int, c: int) -> float:
        """Tensor contraction: Σ S^s_{rc} over r ∈ [r1, r2]."""
        return sum(self[s, r, c] or 0 for r in range(r1, r2 + 1))
    
    def SUMIF(self, s: int, cond_col: int, cond: Callable, sum_col: int) -> float:
        """Masked tensor contraction: Σ M_r · S^s_{r,sum_col}."""
        total = 0
        for r in range(self.N_r):
            cond_val = self[s, r, cond_col]
            if cond(cond_val):
                total += self[s, r, sum_col] or 0
        return total
    
    def VLOOKUP(self, value: Any, s: int, c_search: int, c_return: int) -> Any:
        """Indexed selection: find r* where S^s_{r,c_search} = value."""
        for r in range(self.N_r):
            if self[s, r, c_search] == value:
                return self[s, r, c_return]
        return "#N/A"
    
    # ========== Intent Cohesion (Section 7.4) ==========
    
    def compute_intent_cohesion(self) -> float:
        """
        θ_cohesion = R_Align / R_Drift
        """
        A = self.build_dependency_matrix()
        
        # Define expected pattern E (same-column and same-row references)
        E = np.zeros_like(A)
        
        for i in range(self.M):
            s_i, r_i, c_i = self._delinearize(i)
            for j in range(self.M):
                s_j, r_j, c_j = self._delinearize(j)
                
                # Expected: same column (vertical rollup) or same row (horizontal calc)
                if (s_i == s_j) and (c_i == c_j or r_i == r_j):
                    E[i, j] = 1
        
        # Compute alignment metrics
        D_and_E = A * E  # Aligned references
        D_minus_E = A * (1 - E)  # Drifting references
        
        R_align = np.sum(D_and_E)
        R_drift = np.sum(D_minus_E)
        
        epsilon = 1e-6
        theta = R_align / (R_drift + epsilon)
        
        return theta


class SheetView:
    """Rank-2 view into a spreadsheet (single sheet)."""
    
    def __init__(self, parent: SpreadsheetTensor, sheet_index: int):
        self._parent = parent
        self._s = sheet_index
    
    def __getitem__(self, key: Tuple[int, int]) -> Any:
        r, c = key
        return self._parent[self._s, r, c]
    
    def __setitem__(self, key: Tuple[int, int], value: Any):
        r, c = key
        self._parent[self._s, r, c] = value


# ========== Example Usage ==========

if __name__ == "__main__":
    # Create 1-sheet, 5-row, 3-column tensor
    S = SpreadsheetTensor(n_sheets=1, n_rows=5, n_cols=3)
    
    # Set literal values (Column A: names, Column B: values)
    S[0, 0, 0] = "Item1"
    S[0, 0, 1] = 100
    S[0, 1, 0] = "Item2"
    S[0, 1, 1] = 200
    S[0, 2, 0] = "Item3"
    S[0, 2, 1] = 300
    
    # Set formulas (Column C: computed)
    S.set_formula((0, 0, 2), "=B1*1.1")  # 10% markup
    S.set_formula((0, 1, 2), "=B2*1.1")
    S.set_formula((0, 2, 2), "=B3*1.1")
    
    # Sum formula in row 4
    S.set_formula((0, 3, 1), "=B1+B2+B3")  # Total
    S.set_formula((0, 3, 2), "=C1+C2+C3")  # Total markup
    
    # Check well-formedness
    print(f"Well-formed: {S.is_well_formed()}")
    
    # Compute dependency levels
    levels = S.compute_levels()
    print(f"Dependency levels: {levels}")
    print(f"Max level (K): {S._max_level}")
    
    # Evaluate
    S.evaluate()
    
    # Print results
    print("\nEvaluated tensor:")
    for r in range(4):
        row = [S[0, r, c] for c in range(3)]
        print(f"  Row {r}: {row}")
    
    # Intent Cohesion
    theta = S.compute_intent_cohesion()
    print(f"\nIntent Cohesion θ: {theta:.2f}")
```

## 8.2 JavaScript Implementation

```javascript
/**
 * Spreadsheet Tensor Theory — JavaScript Reference Implementation
 * Auto-Workspace-AI / Intent Tensor Theory Institute
 */

class SpreadsheetTensor {
  /**
   * Rank-3 tensor: S ∈ V^{N_s × N_r × N_c}
   */
  constructor(nSheets, nRows, nCols) {
    this.N_s = nSheets;
    this.N_r = nRows;
    this.N_c = nCols;
    this.M = nSheets * nRows * nCols;
    
    // Sparse storage
    this.data = new Map();      // (s,r,c) -> value
    this.formulas = new Map();  // (s,r,c) -> formula string
    this.deps = new Map();      // (s,r,c) -> Set of (s,r,c) dependencies
    
    // Cached computations
    this.levels = null;
    this.maxLevel = 0;
  }
  
  // ===== Index Operations =====
  
  _key(s, r, c) {
    return `${s},${r},${c}`;
  }
  
  _parseKey(key) {
    return key.split(',').map(Number);
  }
  
  _linearize(s, r, c) {
    return s * (this.N_r * this.N_c) + r * this.N_c + c;
  }
  
  _delinearize(i) {
    const s = Math.floor(i / (this.N_r * this.N_c));
    const remainder = i % (this.N_r * this.N_c);
    const r = Math.floor(remainder / this.N_c);
    const c = remainder % this.N_c;
    return [s, r, c];
  }
  
  // ===== Cell Access =====
  
  get(s, r, c) {
    return this.data.get(this._key(s, r, c)) ?? null;
  }
  
  set(s, r, c, value) {
    const key = this._key(s, r, c);
    this.data.set(key, value);
    this.formulas.delete(key);
    this.deps.delete(key);
    this.levels = null;
  }
  
  setFormula(s, r, c, formula) {
    const key = this._key(s, r, c);
    this.formulas.set(key, formula);
    this.deps.set(key, this._parseReferences(formula));
    this.levels = null;
  }
  
  // ===== Reference Parsing =====
  
  _parseReferences(formula) {
    const refs = new Set();
    const pattern = /(?:(\w+)!)?\$?([A-Z]+)\$?(\d+)/gi;
    let match;
    
    while ((match = pattern.exec(formula)) !== null) {
      const [, sheetName, colStr, rowStr] = match;
      const s = 0; // Default sheet
      const c = this._colToIndex(colStr);
      const r = parseInt(rowStr) - 1;
      
      if (r >= 0 && r < this.N_r && c >= 0 && c < this.N_c) {
        refs.add(this._key(s, r, c));
      }
    }
    
    return refs;
  }
  
  _colToIndex(col) {
    let result = 0;
    for (const char of col.toUpperCase()) {
      result = result * 26 + (char.charCodeAt(0) - 64);
    }
    return result - 1;
  }
  
  // ===== Dependency Analysis =====
  
  buildDependencyMatrix() {
    const A = Array(this.M).fill(null).map(() => Array(this.M).fill(0));
    
    for (const [cellKey, depSet] of this.deps) {
      const [s1, r1, c1] = this._parseKey(cellKey);
      const i = this._linearize(s1, r1, c1);
      
      for (const depKey of depSet) {
        const [s2, r2, c2] = this._parseKey(depKey);
        const j = this._linearize(s2, r2, c2);
        A[i][j] = 1;
      }
    }
    
    return A;
  }
  
  isWellFormed() {
    const A = this.buildDependencyMatrix();
    let Ak = A.map(row => [...row]);
    
    for (let k = 1; k <= this.M; k++) {
      if (Ak.every(row => row.every(v => v === 0))) {
        return true;
      }
      Ak = this._matmul(Ak, A);
    }
    
    return false;
  }
  
  _matmul(A, B) {
    const n = A.length;
    const result = Array(n).fill(null).map(() => Array(n).fill(0));
    
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        for (let k = 0; k < n; k++) {
          result[i][j] += A[i][k] * B[k][j];
        }
      }
    }
    
    return result;
  }
  
  computeLevels() {
    const A = this.buildDependencyMatrix();
    const levels = new Map();
    
    // Level 0: no dependencies
    for (let i = 0; i < this.M; i++) {
      if (A[i].every(v => v === 0)) {
        levels.set(i, 0);
      }
    }
    
    // Iterative level assignment
    let changed = true;
    while (changed) {
      changed = false;
      for (let i = 0; i < this.M; i++) {
        if (levels.has(i)) continue;
        
        const depIndices = A[i]
          .map((v, j) => v === 1 ? j : -1)
          .filter(j => j >= 0);
        
        if (depIndices.every(j => levels.has(j))) {
          const maxDepLevel = Math.max(...depIndices.map(j => levels.get(j)));
          levels.set(i, maxDepLevel + 1);
          changed = true;
        }
      }
    }
    
    // Convert to coordinate form
    this.levels = new Map();
    for (const [i, lvl] of levels) {
      this.levels.set(this._key(...this._delinearize(i)), lvl);
    }
    this.maxLevel = Math.max(...levels.values());
    
    return this.levels;
  }
  
  getLevelSets() {
    if (!this.levels) this.computeLevels();
    
    const levelSets = new Map();
    for (const [key, lvl] of this.levels) {
      if (!levelSets.has(lvl)) levelSets.set(lvl, []);
      levelSets.get(lvl).push(key);
    }
    
    return levelSets;
  }
  
  // ===== Evaluation =====
  
  evaluate() {
    if (!this.isWellFormed()) {
      throw new Error("Spreadsheet contains circular references");
    }
    
    const levelSets = this.getLevelSets();
    
    // Level-by-level evaluation (parallelizable within each level)
    for (let k = 0; k <= this.maxLevel; k++) {
      const cellsAtLevel = levelSets.get(k) || [];
      
      for (const key of cellsAtLevel) {
        if (this.formulas.has(key)) {
          const formula = this.formulas.get(key);
          const value = this._evalFormula(formula);
          this.data.set(key, value);
        }
      }
    }
    
    return this;
  }
  
  _evalFormula(formula) {
    let expr = formula.replace(/^=/, '');
    
    // Replace cell references
    const pattern = /(?:(\w+)!)?\$?([A-Z]+)\$?(\d+)/gi;
    expr = expr.replace(pattern, (match, sheet, col, row) => {
      const s = 0;
      const c = this._colToIndex(col);
      const r = parseInt(row) - 1;
      const val = this.get(s, r, c);
      return val === null ? '0' : typeof val === 'string' ? `"${val}"` : val;
    });
    
    // Expand functions
    expr = expr.replace(/SUM\(([^)]+)\)/gi, (_, args) => {
      const values = args.split(',').map(v => parseFloat(v.trim()) || 0);
      return values.reduce((a, b) => a + b, 0);
    });
    
    try {
      return eval(expr);
    } catch {
      return '#VALUE!';
    }
  }
  
  // ===== Tensor Operations =====
  
  SUM(s, r1, r2, c) {
    let total = 0;
    for (let r = r1; r <= r2; r++) {
      total += this.get(s, r, c) || 0;
    }
    return total;
  }
  
  SUMIF(s, condCol, condition, sumCol) {
    let total = 0;
    for (let r = 0; r < this.N_r; r++) {
      if (condition(this.get(s, r, condCol))) {
        total += this.get(s, r, sumCol) || 0;
      }
    }
    return total;
  }
  
  VLOOKUP(value, s, searchCol, returnCol) {
    for (let r = 0; r < this.N_r; r++) {
      if (this.get(s, r, searchCol) === value) {
        return this.get(s, r, returnCol);
      }
    }
    return '#N/A';
  }
  
  // ===== Intent Cohesion =====
  
  computeIntentCohesion() {
    const A = this.buildDependencyMatrix();
    let RAlign = 0;
    let RDrift = 0;
    
    for (let i = 0; i < this.M; i++) {
      const [si, ri, ci] = this._delinearize(i);
      
      for (let j = 0; j < this.M; j++) {
        if (A[i][j] === 0) continue;
        
        const [sj, rj, cj] = this._delinearize(j);
        
        // Expected: same sheet AND (same column OR same row)
        const isAligned = (si === sj) && (ci === cj || ri === rj);
        
        if (isAligned) {
          RAlign++;
        } else {
          RDrift++;
        }
      }
    }
    
    const epsilon = 1e-6;
    return RAlign / (RDrift + epsilon);
  }
}

// ===== Predictive Map Pattern (from production scripts) =====

class PredictiveMap {
  /**
   * This pattern IS the tensor evaluation pattern:
   * 1. Build state tensor (map) before execution
   * 2. Compute eligibility mask (metaRows)
   * 3. Execute in parallel (batch operations)
   */
  
  constructor(sourceSheet, destSheet) {
    this.source = sourceSheet;
    this.dest = destSheet;
    this.map = {};        // State tensor slice
    this.metaRows = {};   // Mask tensor
  }
  
  buildMap(rows) {
    // This IS constructing S^s_{rc} before evaluation
    for (const row of rows) {
      this.map[row.index] = {
        sourceValue: row.sourceValue,
        destinationValue: row.destinationValue,
        metadata: this.generateMetadata(row)
      };
    }
    return this;
  }
  
  buildMask(rows) {
    // This IS constructing M_r for masked contraction
    for (const row of rows) {
      this.metaRows[row.index] = {
        conditionBlank: row.sourceValue === '',
        conditionNotBlank: row.sourceValue !== '',
        isSelected: row.selected === true,
        isFetchable: this.checkFetchable(row)
      };
    }
    return this;
  }
  
  generateMetadata(row) {
    return {
      timestamp: Date.now(),
      hash: this.hash(row.sourceValue)
    };
  }
  
  checkFetchable(row) {
    return row.sourceValue && 
           row.sourceValue.startsWith('http') && 
           !row.destinationValue;
  }
  
  hash(str) {
    // Simple hash for demo
    return str ? str.length : 0;
  }
  
  execute() {
    // Parallel execution of eligible rows
    const eligible = Object.entries(this.metaRows)
      .filter(([_, meta]) => meta.isFetchable)
      .map(([idx, _]) => this.map[idx]);
    
    // This IS parallel_map(E, F ∘ L_k)
    return Promise.all(eligible.map(row => this.processRow(row)));
  }
  
  processRow(row) {
    // Individual cell evaluation
    return Promise.resolve(row);
  }
}

module.exports = { SpreadsheetTensor, PredictiveMap };
```

---

# 9. Production Pattern Analysis

## 9.1 The Meta-Scripting Architecture

Production Google Apps Scripts implementing high-performance spreadsheet operations already embody STT principles without naming them.

**Pattern 1: Predictive Maps = State Tensor Construction**

```javascript
// Production code
const map = {};
for (const row of rows) {
  map[row] = {
    sourceValue,
    destinationValue,
    metadata: generateMetadata(sourceValue, destinationValue)
  };
}
```

**STT Interpretation:** This constructs a slice of the state tensor $\mathbf{S}[s, :, c_1:c_n]$ before any computation. The separation of observation from action is the tensor pattern: first build the mathematical object, then operate on it.

**Pattern 2: MetaRows = Mask Tensor**

```javascript
// Production code
const metaRows = {};
metaRows[rowIndex] = {
  conditionBlank: sourceValue === '',
  conditionNOTBlank: sourceValue !== '',
  isSelected: checkbox === true,
  notSelected: checkbox === false
};
```

**STT Interpretation:** This is exactly the mask tensor $\mathbf{M} \in \{0,1\}^{N_r}$ used in SUMIF-style operations. Each row carries its own execution eligibility — the script doesn't ask "should I process this?" at runtime; it already knows from the precomputed mask.

**Pattern 3: Batch Fetch = Parallel Level Evaluation**

```javascript
// Production code
const responses = UrlFetchApp.fetchAll(urls);
```

**STT Interpretation:** This is Theorem 7.1 in action. All fetch operations are at the same dependency level (all depend only on the URL column, none depend on each other), so they can be parallelized. The code implicitly recognizes the level structure.

**Pattern 4: Cache = Memoized Evaluation**

```javascript
// Production code
if (cache[url]) {
  return cache[url];
}
const result = fetch(url);
cache[url] = result;
return result;
```

**STT Interpretation:** This implements the fixed-point property. Once a cell is evaluated, re-evaluation produces the same result. Caching exploits this mathematical guarantee.

## 9.2 Mapping Production Code to STT

| Production Pattern | STT Concept | Mathematical Form |
|-------------------|-------------|-------------------|
| `map[row] = {...}` | State tensor slice | $\mathbf{S}[s, :, c_{src}:c_{dst}]$ |
| `metaRows[row].isEligible` | Mask tensor | $M_r \in \{0,1\}$ |
| `fetchAll(urls)` | Parallel evaluation | $\text{parallel\_map}(\mathcal{E}, \mathbf{L}_k)$ |
| `cache[key] = value` | Fixed point memoization | $\mathcal{E}(\mathbf{F}, \mathbf{S}^*) = \mathbf{S}^*$ |
| `writeBatch(ranges, values)` | Tensor slice assignment | $\mathbf{S}[s, r_1:r_2, c] \leftarrow \vec{v}$ |
| `topological processing` | Level-order evaluation | $\sum_k \mathcal{E}_k(\mathbf{F} \circ \mathbf{L}_k)$ |

The production code IS implementing STT — the mathematical structure was always there, just unnamed and implicit.

---

# 10. Open Research Questions

## 10.1 Categorical Formulation

**Question:** Can the recursive structure be expressed as a terminal coalgebra?

The self-similar decomposition $\mathbf{S} = \bigoplus_s \bigoplus_r \bigoplus_c S^s_{rc}$ suggests a coalgebraic structure where spreadsheets are greatest fixed points of a structure functor.

## 10.2 Differential Structure

**Question:** Is there a meaningful notion of $\frac{\partial S^s_{rc}}{\partial S^{s'}_{r'c'}}$ for sensitivity analysis?

This would enable automatic differentiation through spreadsheet computations, with applications to:
- What-if analysis
- Goal seeking
- Optimization

## 10.3 Quantum Extension

**Question:** Can spreadsheet superposition enable probabilistic scenario modeling?

A quantum spreadsheet tensor $|\mathbf{S}\rangle \in \mathcal{H}^{N_s \times N_r \times N_c}$ could represent superpositions of scenarios, collapsing to classical values upon observation.

## 10.4 Compression

**Question:** What is the minimum description length of a spreadsheet given its tensor structure?

The formula tensor $\mathbf{F}$ is typically much sparser than the value tensor $\mathbf{S}$. Kolmogorov complexity analysis could yield optimal compression schemes.

## 10.5 Type-Theoretic Foundations

**Question:** Can dependent types capture the constraint that formulas only reference existing cells?

A dependently-typed spreadsheet would have well-formedness guaranteed by construction, eliminating runtime circular reference checks.

---

# 11. Conclusion

We have established that spreadsheets are rank-3 tensors — not metaphorically, but literally. This reconceptualization yields:

1. **Formal foundations** for spreadsheet computation via tensor algebra
2. **Proof of key properties** (acyclicity, convergence, parallelization, change propagation)
3. **Bridge to Intent Tensor Theory** connecting discrete computation to continuous collapse geometry
4. **Computable metrics** (Intent Cohesion) for spreadsheet quality assessment
5. **Recognition** that production patterns already implement STT implicitly

The spreadsheet is the world's most widely deployed tensor processor. By naming its mathematical structure, we enable rigorous analysis, optimization, and extension of the most important computational substrate in business.

---

# References

1. Sestoft, P., et al. (2020). "On the Semantics for Spreadsheets with Sheet-Defined Functions." *Journal of Computer Languages*.

2. Erwig, M., & Abraham, R. (2006). "UCheck: A Spreadsheet Type Checker for End Users." *Journal of Visual Languages & Computing*.

3. Peyton Jones, S., et al. (2003). "A User-Centred Approach to Functions in Excel." *ICFP*.

4. Williams, J., et al. (2020). "Spill: A calculus for array formulas in spreadsheets." *ESOP*.

5. Iverson, K. (1962). *A Programming Language*. Wiley.

6. Knight, A. (2025). "Intent Tensor Theory: Recursive Collapse Geometry." *Intent Tensor Theory Institute*.

7. Auto-Workspace-AI. (2025). "f(AutoWorkspace): Business Mathematics Foundation." GitHub Repository.

---

**Document Version:** 1.0  
**Status:** Complete Draft  
**Contact:** Auto-Workspace-AI / Intent Tensor Theory Institute  
**Repository:** https://github.com/Sensei-Intent-Tensor/0.0_business_math_foundation_principals/tree/main/f(spreadsheets)

