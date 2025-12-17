# f(Spreadsheets)

## Spreadsheet Tensor Theory — A Mathematical Foundation

[![Auto-Workspace-AI](https://img.shields.io/badge/Auto--Workspace--AI-Framework-blue)](https://auto-workspace-ai.com)
[![Intent Tensor Theory](https://img.shields.io/badge/ITT-Institute-purple)](https://intent-tensor-theory.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## The Core Declaration

```
A spreadsheet is a rank-3 tensor.

        S ∈ V^{N_s × N_r × N_c}

Not metaphorically — literally.
```

---

## What is f(Spreadsheets)?

**f(Spreadsheets)** treats the spreadsheet as a mathematical function `f(x)` — an object that can be formally analyzed, computed, and reasoned about. This repository contains the complete **Spreadsheet Tensor Theory (STT)** framework, which establishes that:

1. Every spreadsheet is intrinsically a **rank-3 tensor**
2. Standard operations (SUM, VLOOKUP, SUMIF) are **tensor contractions**
3. Dependency graphs encode as **rank-6 binary tensors**
4. Well-formedness equals **nilpotent dependency matrices**
5. Evaluation converges to a **unique fixed point**

This is not an analogy. This is what spreadsheets ARE.

---

## The Three Master Equations

### 1. The State Equation
*How spreadsheets respond to input:*

```
S(t) = E(F, S(t⁻)) · (I - P) + X(t) · P
```

### 2. The Computation Equation
*How evaluation propagates:*

```
S* = Σₖ Eₖ(F ∘ Lₖ)
```

### 3. The Recursive Structure Equation
*The self-similarity:*

```
S = R⁽³⁾(R⁽²⁾(R⁽¹⁾(V)))
```

---

## Repository Structure

```
f(spreadsheets)/
├── README.md                    # This file
├── 00_WHITE_PAPER.md           # Complete academic framework
├── 01_FORMAL_PROOFS.md         # Rigorous mathematical proofs
├── 02_PRODUCTION_PATTERNS.md   # How production code implements STT
├── 03_ITT_BRIDGE.md            # Connection to Intent Tensor Theory
├── spreadsheet_tensor.py       # Python reference implementation
├── spreadsheet_tensor.js       # JavaScript/GAS implementation
└── examples/
    ├── basic_usage.py
    ├── intent_cohesion.py
    └── incremental_recalc.py
```

---

## Quick Start

### Python

```python
from spreadsheet_tensor import SpreadsheetTensor

# Create tensor: 1 sheet, 10 rows, 5 columns
S = SpreadsheetTensor(n_sheets=1, n_rows=10, n_cols=5)

# Set values
S[0, 0, 0] = "Revenue"
S[0, 0, 1] = 1000
S[0, 1, 0] = "Costs"  
S[0, 1, 1] = 600

# Set formula
S.set_formula(0, 2, 1, "=B1-B2")  # Profit = Revenue - Costs

# Check well-formedness (no circular refs)
assert S.is_well_formed()

# Evaluate to fixed point
S.evaluate()

# Result
print(S[0, 2, 1])  # 400
```

### JavaScript

```javascript
const { SpreadsheetTensor } = require('./spreadsheet_tensor');

// Create tensor
const S = new SpreadsheetTensor(1, 10, 5);

// Set values and formulas
S.set(0, 0, 1, 1000);
S.set(0, 1, 1, 600);
S.setFormula(0, 2, 1, '=B1-B2');

// Evaluate
S.evaluate();

console.log(S.get(0, 2, 1));  // 400
```

---

## Key Concepts

### Tensor Dimensions → Business Primitives

| Dimension | Index | Business Meaning |
|-----------|-------|------------------|
| Sheet | `s` | Business unit, time period, scenario |
| Row | `r` | Entity instance (employee, product, transaction) |
| Column | `c` | Attribute, measure, KPI |

### Operations → Tensor Algebra

| Spreadsheet | Tensor Operation |
|-------------|-----------------|
| `SUM(A1:A10)` | Contraction over row index |
| `VLOOKUP(v, range, col)` | Conditional slice |
| `SUMIF(cond, sum)` | Masked contraction |
| `INDEX(range, r, c)` | Direct tensor access |
| Pivot Table | Dimension exchange |

### Metrics

**Intent Cohesion** measures structural alignment:

```
θ_cohesion = R_Align / R_Drift
```

- High θ → References follow expected patterns
- Low θ → Ad-hoc references, potential technical debt

---

## The Four Fundamental Theorems

### Theorem 1: Acyclicity
A spreadsheet is well-formed iff its dependency matrix is nilpotent:
```
∃k : Aᵏ = 0
```

### Theorem 2: Fixed Point
Evaluation converges to unique fixed point:
```
S* = lim(n→∞) Eⁿ(F, S₀)
```

### Theorem 3: Parallelization
Cells at same dependency level evaluate in parallel:
```
T_parallel = O(K) vs T_sequential = O(M)
```

### Theorem 4: Change Propagation
Affected cells form transitive closure:
```
affected(C) = {j : (Aᵀ)⁺_ij > 0, i ∈ C}
```

---

## Connection to Intent Tensor Theory

STT is a discrete instantiation of the broader **Intent Tensor Theory** framework:

| ITT Concept | STT Realization |
|-------------|-----------------|
| Scalar potential Φ | Cell value S^s_rc |
| Collapse vector ∇Φ | Formula reference |
| Recursive curl | Circular reference |
| Curvature lock | Aggregation function |
| Tensor lock | Fixed point S* |

The spreadsheet is the world's most widely deployed tensor processor. By naming its mathematical structure, we enable rigorous analysis, optimization, and extension.

---

## Why This Matters

### For Developers
- **Optimization**: Parallel evaluation by dependency level
- **Incremental**: Only recalculate affected cells
- **Verification**: Formal proofs of correctness

### For Business
- **Quantified Quality**: Intent Cohesion metric
- **Structural Analysis**: Dependency visualization
- **f(AutoWorkspace)**: Business math becomes computable

### For Research
- **Novel Formalization**: First tensor-algebraic treatment
- **Bridge to ITT**: Connects to broader mathematical framework
- **Open Questions**: Categorical structure, differential forms, quantum extension

---

## Prior Art & Contributions

**What Existed:**
- Operational semantics (evaluation rules)
- Type systems (error detection)
- Functional programming models
- Category-theoretic treatments

**What We Formalized:**
- Tensor-algebraic structure
- Dependency as rank-6 tensor
- Operations as tensor contractions
- Recursive self-similarity axiom
- Intent Cohesion metric

---

## Installation

### Python
```bash
pip install numpy
# Then copy spreadsheet_tensor.py to your project
```

### JavaScript/Node.js
```bash
# Copy spreadsheet_tensor.js to your project
const { SpreadsheetTensor } = require('./spreadsheet_tensor');
```

### Google Apps Script
```javascript
// Copy the SpreadsheetTensor class into your GAS project
// Works directly with Sheets API
```

---

## Contributing

This is an open mathematical framework. Contributions welcome:

1. **Proofs**: Additional theorems or simplified proofs
2. **Implementations**: Other languages (Rust, Go, etc.)
3. **Applications**: Real-world use cases
4. **Extensions**: Categorical formulation, differential structure

---

## References

1. **Intent Tensor Theory Institute** — [intent-tensor-theory.com](https://intent-tensor-theory.com)
2. **Auto-Workspace-AI** — [auto-workspace-ai.com](https://auto-workspace-ai.com)
3. **f(AutoWorkspace)** — Business Mathematics Foundation

---

## License

MIT License — Use freely, attribute appropriately.

---

## Contact

**Auto-Workspace-AI / Intent Tensor Theory Institute**

- GitHub: [@Sensei-Intent-Tensor](https://github.com/Sensei-Intent-Tensor)
- Framework: [0.0_business_math_foundation_principals](https://github.com/Sensei-Intent-Tensor/0.0_business_math_foundation_principals)

---

*"The spreadsheet is the world's accidental universal business tensor. We're making it intentional."*
