"""
Spreadsheet Tensor Theory — Python Reference Implementation
============================================================

Auto-Workspace-AI / Intent Tensor Theory Institute
Version 1.0

A spreadsheet is a rank-3 tensor: S ∈ V^{N_s × N_r × N_c}

This implementation provides:
- SpreadsheetTensor class with full tensor operations
- Dependency analysis and level computation
- Fixed-point evaluation with parallelization support
- Intent Cohesion metric computation
- VLOOKUP, SUMIF, and aggregation as tensor operations
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import re
from concurrent.futures import ThreadPoolExecutor, as_completed


# =============================================================================
# §1. VALUE SPACE DEFINITION
# =============================================================================

class ValueType(Enum):
    """
    Tagged union discriminator for V = ℝ ∪ Σ* ∪ 𝔹 ∪ ℰ ∪ {∅}
    """
    NUMBER = "number"      # ℝ (reals, including integers)
    STRING = "string"      # Σ* (strings over Unicode)
    BOOLEAN = "boolean"    # 𝔹 = {⊤, ⊥}
    ERROR = "error"        # ℰ = {#REF!, #VALUE!, ...}
    NULL = "null"          # ∅ (empty cell)


class SpreadsheetError(Enum):
    """Error states in ℰ"""
    REF = "#REF!"
    VALUE = "#VALUE!"
    DIV_ZERO = "#DIV/0!"
    NAME = "#NAME?"
    NULL = "#NULL!"
    NA = "#N/A"
    NUM = "#NUM!"


@dataclass
class Cell:
    """
    Rank-0 tensor element S^s_{rc} ∈ V
    """
    value: Any
    value_type: ValueType
    formula: Optional[str] = None
    
    @property
    def is_formula(self) -> bool:
        return self.formula is not None
    
    @property
    def is_error(self) -> bool:
        return self.value_type == ValueType.ERROR
    
    def __repr__(self):
        if self.formula:
            return f"Cell({self.value}, formula='{self.formula}')"
        return f"Cell({self.value})"


# =============================================================================
# §2. SPREADSHEET TENSOR CLASS
# =============================================================================

class SpreadsheetTensor:
    """
    Rank-3 tensor representation of a spreadsheet.
    
    S ∈ V^{N_s × N_r × N_c}
    
    Where:
        N_s = number of sheets (depth dimension)
        N_r = number of rows (vertical dimension)
        N_c = number of columns (horizontal dimension)
        V = heterogeneous value space
    """
    
    def __init__(self, n_sheets: int, n_rows: int, n_cols: int):
        """
        Initialize empty spreadsheet tensor.
        
        Args:
            n_sheets: Number of sheets (N_s)
            n_rows: Number of rows per sheet (N_r)
            n_cols: Number of columns per sheet (N_c)
        """
        # Tensor dimensions
        self.N_s = n_sheets
        self.N_r = n_rows
        self.N_c = n_cols
        self.M = n_sheets * n_rows * n_cols  # Total cells
        
        # Value tensor S (sparse storage)
        self._data: Dict[Tuple[int, int, int], Cell] = {}
        
        # Formula tensor F
        self._formulas: Dict[Tuple[int, int, int], str] = {}
        
        # Dependency tensor D (sparse: cell -> set of referenced cells)
        self._dependencies: Dict[Tuple[int, int, int], Set[Tuple[int, int, int]]] = {}
        
        # Reverse dependencies (cell -> cells that depend on it)
        self._dependents: Dict[Tuple[int, int, int], Set[Tuple[int, int, int]]] = {}
        
        # Cached computations
        self._dep_matrix: Optional[np.ndarray] = None
        self._dep_matrix_dirty: bool = True
        self._levels: Dict[Tuple[int, int, int], int] = {}
        self._level_sets: Dict[int, List[Tuple[int, int, int]]] = {}
        self._max_level: int = 0
    
    # =========================================================================
    # §2.1 INDEX OPERATIONS
    # =========================================================================
    
    def _linearize(self, s: int, r: int, c: int) -> int:
        """
        Linearization function φ(s, r, c) = s·(N_r·N_c) + r·N_c + c
        
        Maps 3D coordinates to 1D index for matrix operations.
        """
        return s * (self.N_r * self.N_c) + r * self.N_c + c
    
    def _delinearize(self, i: int) -> Tuple[int, int, int]:
        """
        Inverse linearization φ^{-1}(i) = (s, r, c)
        """
        s = i // (self.N_r * self.N_c)
        remainder = i % (self.N_r * self.N_c)
        r = remainder // self.N_c
        c = remainder % self.N_c
        return (s, r, c)
    
    def _validate_coords(self, s: int, r: int, c: int) -> bool:
        """Check if coordinates are within tensor bounds."""
        return (0 <= s < self.N_s and 
                0 <= r < self.N_r and 
                0 <= c < self.N_c)
    
    # =========================================================================
    # §2.2 CELL ACCESS (TENSOR INDEXING)
    # =========================================================================
    
    def __getitem__(self, key: Tuple[int, int, int]) -> Any:
        """
        Cell access: S^s_{rc}
        
        Usage: value = S[s, r, c]
        """
        if key in self._data:
            return self._data[key].value
        return None
    
    def __setitem__(self, key: Tuple[int, int, int], value: Any):
        """
        Set literal cell value (clears any formula).
        
        Usage: S[s, r, c] = value
        """
        s, r, c = key
        if not self._validate_coords(s, r, c):
            raise IndexError(f"Cell {key} out of bounds")
        
        vtype = self._infer_type(value)
        self._data[key] = Cell(value=value, value_type=vtype, formula=None)
        
        # Clear formula and dependencies
        if key in self._formulas:
            del self._formulas[key]
        if key in self._dependencies:
            # Remove from dependents of referenced cells
            for ref in self._dependencies[key]:
                if ref in self._dependents:
                    self._dependents[ref].discard(key)
            del self._dependencies[key]
        
        self._invalidate_cache()
    
    def get_cell(self, s: int, r: int, c: int) -> Optional[Cell]:
        """Get full Cell object (value + metadata)."""
        return self._data.get((s, r, c))
    
    def set_formula(self, s: int, r: int, c: int, formula: str):
        """
        Set cell formula and compute dependencies.
        
        Args:
            s, r, c: Cell coordinates
            formula: Formula string (e.g., "=A1+B2")
        """
        key = (s, r, c)
        if not self._validate_coords(s, r, c):
            raise IndexError(f"Cell {key} out of bounds")
        
        # Store formula
        self._formulas[key] = formula
        
        # Parse and store dependencies
        old_deps = self._dependencies.get(key, set())
        new_deps = self._parse_references(formula, s)
        self._dependencies[key] = new_deps
        
        # Update reverse dependency map
        for ref in old_deps - new_deps:
            if ref in self._dependents:
                self._dependents[ref].discard(key)
        for ref in new_deps - old_deps:
            if ref not in self._dependents:
                self._dependents[ref] = set()
            self._dependents[ref].add(key)
        
        # Create cell with null value (will be computed)
        self._data[key] = Cell(value=None, value_type=ValueType.NULL, formula=formula)
        
        self._invalidate_cache()
    
    def _invalidate_cache(self):
        """Mark cached computations as dirty."""
        self._dep_matrix_dirty = True
        self._levels = {}
        self._level_sets = {}
    
    def _infer_type(self, value: Any) -> ValueType:
        """Infer ValueType from Python value."""
        if value is None:
            return ValueType.NULL
        elif isinstance(value, bool):
            return ValueType.BOOLEAN
        elif isinstance(value, (int, float, np.number)):
            return ValueType.NUMBER
        elif isinstance(value, str):
            if value.startswith('#') and value.endswith('!'):
                return ValueType.ERROR
            return ValueType.STRING
        elif isinstance(value, SpreadsheetError):
            return ValueType.ERROR
        return ValueType.NULL
    
    # =========================================================================
    # §2.3 SLICE OPERATIONS (Definition 5.1)
    # =========================================================================
    
    def sheet(self, s: int) -> 'SheetSlice':
        """
        Sheet selection: S[s, :, :] ∈ V^{N_r × N_c}
        
        Rank reduction 3→2.
        """
        if not (0 <= s < self.N_s):
            raise IndexError(f"Sheet index {s} out of bounds")
        return SheetSlice(self, s)
    
    def row(self, s: int, r: int) -> List[Any]:
        """
        Row selection: S[s, r, :] ∈ V^{N_c}
        
        Rank reduction 3→1.
        """
        return [self[s, r, c] for c in range(self.N_c)]
    
    def column(self, s: int, c: int) -> List[Any]:
        """
        Column selection: S[s, :, c] ∈ V^{N_r}
        
        Rank reduction 3→1.
        """
        return [self[s, r, c] for r in range(self.N_r)]
    
    def range(self, s: int, r1: int, r2: int, c1: int, c2: int) -> List[List[Any]]:
        """
        Range selection: S[s, r1:r2, c1:c2] ∈ V^{(r2-r1+1) × (c2-c1+1)}
        
        Subspace extraction.
        """
        return [[self[s, r, c] for c in range(c1, c2 + 1)] 
                for r in range(r1, r2 + 1)]
    
    def flat_range(self, s: int, r1: int, r2: int, c1: int, c2: int) -> List[Any]:
        """Flattened range for aggregation operations."""
        values = []
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                v = self[s, r, c]
                if v is not None:
                    values.append(v)
        return values
    
    # =========================================================================
    # §3. REFERENCE PARSING
    # =========================================================================
    
    def _parse_references(self, formula: str, current_sheet: int = 0) -> Set[Tuple[int, int, int]]:
        """
        Extract cell references from formula.
        
        Handles:
            - Simple refs: A1, B2
            - Absolute refs: $A$1, A$1, $A1
            - Sheet refs: Sheet1!A1
            - Range refs: A1:B5 (expands to all cells)
        """
        refs = set()
        
        # Pattern for individual cell references
        cell_pattern = r"(?:(['\"]?[\w\s]+['\"]?)!)?\$?([A-Z]+)\$?(\d+)"
        
        # Pattern for ranges
        range_pattern = r"(?:(['\"]?[\w\s]+['\"]?)!)?\$?([A-Z]+)\$?(\d+):\$?([A-Z]+)\$?(\d+)"
        
        # Handle ranges first (expand to individual cells)
        for match in re.finditer(range_pattern, formula, re.IGNORECASE):
            sheet_name, col1_str, row1_str, col2_str, row2_str = match.groups()
            s = self._sheet_name_to_index(sheet_name) if sheet_name else current_sheet
            c1 = self._col_to_index(col1_str)
            c2 = self._col_to_index(col2_str)
            r1 = int(row1_str) - 1
            r2 = int(row2_str) - 1
            
            for r in range(min(r1, r2), max(r1, r2) + 1):
                for c in range(min(c1, c2), max(c1, c2) + 1):
                    if self._validate_coords(s, r, c):
                        refs.add((s, r, c))
        
        # Handle individual cell references (exclude those in ranges)
        formula_no_ranges = re.sub(range_pattern, '', formula, flags=re.IGNORECASE)
        for match in re.finditer(cell_pattern, formula_no_ranges, re.IGNORECASE):
            sheet_name, col_str, row_str = match.groups()
            s = self._sheet_name_to_index(sheet_name) if sheet_name else current_sheet
            c = self._col_to_index(col_str)
            r = int(row_str) - 1
            
            if self._validate_coords(s, r, c):
                refs.add((s, r, c))
        
        return refs
    
    def _col_to_index(self, col: str) -> int:
        """Convert column letter(s) to 0-based index: A=0, B=1, ..., Z=25, AA=26, ..."""
        result = 0
        for char in col.upper():
            result = result * 26 + (ord(char) - ord('A') + 1)
        return result - 1
    
    def _index_to_col(self, idx: int) -> str:
        """Convert 0-based index to column letter(s)."""
        result = ""
        idx += 1
        while idx > 0:
            idx, remainder = divmod(idx - 1, 26)
            result = chr(ord('A') + remainder) + result
        return result
    
    def _sheet_name_to_index(self, name: Optional[str]) -> int:
        """Convert sheet name to index (placeholder - extend for named sheets)."""
        if name is None:
            return 0
        # Strip quotes
        name = name.strip("'\"")
        # For now, assume Sheet1, Sheet2, etc.
        if name.lower().startswith('sheet'):
            try:
                return int(name[5:]) - 1
            except ValueError:
                pass
        return 0
    
    # =========================================================================
    # §4. DEPENDENCY ANALYSIS (Section 3)
    # =========================================================================
    
    def build_dependency_matrix(self) -> np.ndarray:
        """
        Construct flattened dependency matrix A ∈ {0,1}^{M×M}.
        
        A_{ij} = 1 iff cell i references cell j.
        """
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
        """
        Theorem 3.3: Check well-formedness (acyclicity).
        
        A spreadsheet is well-formed iff ∃k: A^k = 0 (nilpotent).
        """
        A = self.build_dependency_matrix()
        
        # Optimization: check for cycles using DFS instead of matrix powers
        # for large spreadsheets
        if self.M > 1000:
            return self._is_acyclic_dfs()
        
        A_power = A.copy().astype(np.float64)
        
        for k in range(1, self.M + 1):
            if np.all(A_power == 0):
                return True
            A_power = A_power @ A
            # Prevent overflow
            A_power = np.clip(A_power, 0, 1)
        
        return False
    
    def _is_acyclic_dfs(self) -> bool:
        """Check acyclicity using depth-first search (more efficient for large graphs)."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {cell: WHITE for cell in self._dependencies}
        
        def has_cycle(cell):
            color[cell] = GRAY
            for dep in self._dependencies.get(cell, set()):
                if dep in color:
                    if color[dep] == GRAY:
                        return True
                    if color[dep] == WHITE and has_cycle(dep):
                        return True
            color[cell] = BLACK
            return False
        
        for cell in self._dependencies:
            if color.get(cell, WHITE) == WHITE:
                if has_cycle(cell):
                    return False
        return True
    
    def compute_levels(self) -> Dict[Tuple[int, int, int], int]:
        """
        Compute dependency levels for all cells (Definition 4.2.2).
        
        ℓ(i) = 0 if cell has no dependencies
        ℓ(i) = 1 + max{ℓ(j) : A_{ij} = 1} otherwise
        """
        if self._levels:
            return self._levels
        
        A = self.build_dependency_matrix()
        levels: Dict[int, int] = {}
        
        # Initialize: cells with no dependencies are level 0
        for i in range(self.M):
            if np.sum(A[i, :]) == 0:
                levels[i] = 0
        
        # Iteratively assign levels
        max_iterations = self.M
        for _ in range(max_iterations):
            changed = False
            for i in range(self.M):
                if i in levels:
                    continue
                
                deps = np.where(A[i, :] == 1)[0]
                if len(deps) == 0:
                    levels[i] = 0
                    changed = True
                elif all(d in levels for d in deps):
                    levels[i] = 1 + max(levels[d] for d in deps)
                    changed = True
            
            if not changed:
                break
        
        # Convert to coordinate form
        self._levels = {self._delinearize(i): lvl for i, lvl in levels.items()}
        
        # Also store all cells without formulas at level 0
        for s in range(self.N_s):
            for r in range(self.N_r):
                for c in range(self.N_c):
                    key = (s, r, c)
                    if key not in self._levels:
                        self._levels[key] = 0
        
        self._max_level = max(self._levels.values()) if self._levels else 0
        
        return self._levels
    
    def get_level_sets(self) -> Dict[int, List[Tuple[int, int, int]]]:
        """
        Return L_k = {cells at level k} for each level.
        
        Used for parallel evaluation (Theorem 7.1).
        """
        if self._level_sets:
            return self._level_sets
        
        if not self._levels:
            self.compute_levels()
        
        level_sets: Dict[int, List[Tuple[int, int, int]]] = {}
        for cell, lvl in self._levels.items():
            if lvl not in level_sets:
                level_sets[lvl] = []
            level_sets[lvl].append(cell)
        
        self._level_sets = level_sets
        return level_sets
    
    # =========================================================================
    # §5. EVALUATION (Section 4)
    # =========================================================================
    
    def evaluate(self, parallel: bool = False, max_workers: int = 4) -> 'SpreadsheetTensor':
        """
        Fixed-point evaluation (Theorem 4.3).
        
        Computes S* satisfying E(F, S*) = S*.
        
        Args:
            parallel: If True, use parallel evaluation within levels
            max_workers: Number of parallel workers
            
        Returns:
            Self (for chaining)
        """
        if not self.is_well_formed():
            raise ValueError("Spreadsheet contains circular references (not well-formed)")
        
        level_sets = self.get_level_sets()
        
        # Evaluate level by level
        for k in range(self._max_level + 1):
            cells_at_level = level_sets.get(k, [])
            
            # Filter to only cells with formulas
            formula_cells = [c for c in cells_at_level if c in self._formulas]
            
            if parallel and len(formula_cells) > 1:
                # Theorem 7.1: Cells at same level can be evaluated in parallel
                self._evaluate_level_parallel(formula_cells, max_workers)
            else:
                for cell in formula_cells:
                    self._evaluate_cell(cell)
        
        return self
    
    def _evaluate_level_parallel(self, cells: List[Tuple[int, int, int]], max_workers: int):
        """Evaluate all cells at a level in parallel."""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._compute_cell_value, cell): cell 
                      for cell in cells}
            
            for future in as_completed(futures):
                cell = futures[future]
                try:
                    value = future.result()
                    self._set_computed_value(cell, value)
                except Exception as e:
                    self._set_computed_value(cell, SpreadsheetError.VALUE)
    
    def _evaluate_cell(self, cell: Tuple[int, int, int]):
        """Evaluate a single cell."""
        value = self._compute_cell_value(cell)
        self._set_computed_value(cell, value)
    
    def _compute_cell_value(self, cell: Tuple[int, int, int]) -> Any:
        """Compute the value of a formula cell."""
        formula = self._formulas.get(cell)
        if formula is None:
            return self[cell]
        return self._eval_formula(formula, cell[0])
    
    def _set_computed_value(self, cell: Tuple[int, int, int], value: Any):
        """Store computed value while preserving formula."""
        formula = self._formulas.get(cell)
        vtype = self._infer_type(value)
        self._data[cell] = Cell(value=value, value_type=vtype, formula=formula)
    
    def _eval_formula(self, formula: str, current_sheet: int = 0) -> Any:
        """
        Evaluate a formula given current cell values.
        
        This is a simplified evaluator. Extend for full formula language support.
        """
        expr = formula.lstrip('=')
        
        # Replace range references with lists
        range_pattern = r"(?:(['\"]?[\w\s]+['\"]?)!)?\$?([A-Z]+)\$?(\d+):\$?([A-Z]+)\$?(\d+)"
        
        def replace_range(match):
            sheet_name, col1, row1, col2, row2 = match.groups()
            s = self._sheet_name_to_index(sheet_name) if sheet_name else current_sheet
            c1, c2 = self._col_to_index(col1), self._col_to_index(col2)
            r1, r2 = int(row1) - 1, int(row2) - 1
            
            values = []
            for r in range(min(r1, r2), max(r1, r2) + 1):
                for c in range(min(c1, c2), max(c1, c2) + 1):
                    v = self[s, r, c]
                    if isinstance(v, (int, float)):
                        values.append(v)
                    elif v is not None:
                        values.append(0)
            return str(values)
        
        expr = re.sub(range_pattern, replace_range, expr, flags=re.IGNORECASE)
        
        # Replace cell references with values
        cell_pattern = r"(?:(['\"]?[\w\s]+['\"]?)!)?\$?([A-Z]+)\$?(\d+)"
        
        def replace_cell(match):
            sheet_name, col_str, row_str = match.groups()
            s = self._sheet_name_to_index(sheet_name) if sheet_name else current_sheet
            c = self._col_to_index(col_str)
            r = int(row_str) - 1
            
            val = self[s, r, c]
            if val is None:
                return "0"
            elif isinstance(val, str):
                return f'"{val}"'
            elif isinstance(val, bool):
                return str(val)
            return str(val)
        
        expr = re.sub(cell_pattern, replace_cell, expr, flags=re.IGNORECASE)
        
        # Expand spreadsheet functions to Python
        expr = self._expand_functions(expr)
        
        try:
            result = eval(expr)
            return result
        except ZeroDivisionError:
            return SpreadsheetError.DIV_ZERO
        except (ValueError, TypeError):
            return SpreadsheetError.VALUE
        except Exception:
            return SpreadsheetError.VALUE
    
    def _expand_functions(self, expr: str) -> str:
        """Expand spreadsheet functions to Python equivalents."""
        # SUM
        expr = re.sub(r'SUM\s*\(\s*(\[[^\]]+\])\s*\)', r'sum(\1)', expr, flags=re.IGNORECASE)
        # AVERAGE
        expr = re.sub(r'AVERAGE\s*\(\s*(\[[^\]]+\])\s*\)', 
                     r'(sum(\1)/len(\1) if len(\1) > 0 else 0)', expr, flags=re.IGNORECASE)
        # COUNT
        expr = re.sub(r'COUNT\s*\(\s*(\[[^\]]+\])\s*\)', r'len(\1)', expr, flags=re.IGNORECASE)
        # MAX
        expr = re.sub(r'MAX\s*\(\s*(\[[^\]]+\])\s*\)', r'max(\1)', expr, flags=re.IGNORECASE)
        # MIN
        expr = re.sub(r'MIN\s*\(\s*(\[[^\]]+\])\s*\)', r'min(\1)', expr, flags=re.IGNORECASE)
        # IF
        expr = re.sub(r'IF\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)',
                     r'(\2 if \1 else \3)', expr, flags=re.IGNORECASE)
        return expr
    
    # =========================================================================
    # §6. CHANGE PROPAGATION (Theorem 7.2)
    # =========================================================================
    
    def get_affected_cells(self, changed: Set[Tuple[int, int, int]]) -> Set[Tuple[int, int, int]]:
        """
        Compute affected(C) for incremental recalculation.
        
        affected(C) = {j : (A^T)^+_{ij} > 0 for some i ∈ C}
        
        Uses reverse dependency map for efficiency.
        """
        affected = set()
        to_process = list(changed)
        
        while to_process:
            cell = to_process.pop()
            for dependent in self._dependents.get(cell, set()):
                if dependent not in affected:
                    affected.add(dependent)
                    to_process.append(dependent)
        
        return affected
    
    def recalculate(self, changed: Set[Tuple[int, int, int]]) -> Set[Tuple[int, int, int]]:
        """
        Incremental recalculation after changes.
        
        Only re-evaluates affected cells, not entire tensor.
        
        Returns:
            Set of cells that were recalculated
        """
        affected = self.get_affected_cells(changed)
        
        # Re-evaluate affected cells in level order
        if not self._levels:
            self.compute_levels()
        
        # Group affected cells by level
        by_level: Dict[int, List[Tuple[int, int, int]]] = {}
        for cell in affected:
            lvl = self._levels.get(cell, 0)
            if lvl not in by_level:
                by_level[lvl] = []
            by_level[lvl].append(cell)
        
        # Evaluate in level order
        for k in sorted(by_level.keys()):
            for cell in by_level[k]:
                if cell in self._formulas:
                    self._evaluate_cell(cell)
        
        return affected
    
    # =========================================================================
    # §7. TENSOR OPERATIONS (Section 5)
    # =========================================================================
    
    def SUM(self, s: int, r1: int, r2: int, c1: int, c2: int) -> float:
        """
        Tensor contraction: Σ S^s_{rc} over specified range.
        
        SUM(S[s, r1:r2, c1:c2]) = Σ_r Σ_c S^s_{rc}
        """
        total = 0.0
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                val = self[s, r, c]
                if isinstance(val, (int, float)):
                    total += val
        return total
    
    def SUMIF(self, s: int, cond_r1: int, cond_r2: int, cond_c: int,
              condition: Callable[[Any], bool],
              sum_c: int) -> float:
        """
        Masked tensor contraction (Definition 5.4.1).
        
        SUMIF = Σ_r M_r · S^s_{r, sum_c}
        where M_r = 1[condition(S^s_{r, cond_c})]
        """
        total = 0.0
        for r in range(cond_r1, cond_r2 + 1):
            cond_val = self[s, r, cond_c]
            if condition(cond_val):
                sum_val = self[s, r, sum_c]
                if isinstance(sum_val, (int, float)):
                    total += sum_val
        return total
    
    def COUNTIF(self, s: int, r1: int, r2: int, c: int,
                condition: Callable[[Any], bool]) -> int:
        """
        Masked count: Σ_r M_r where M_r = 1[condition(S^s_{rc})]
        """
        count = 0
        for r in range(r1, r2 + 1):
            if condition(self[s, r, c]):
                count += 1
        return count
    
    def VLOOKUP(self, value: Any, s: int, search_c: int, 
                return_c: int, r1: int = 0, r2: int = None) -> Any:
        """
        Indexed selection (Definition 5.3.1).
        
        VLOOKUP(v, S[s, :, c1:cn], k) = S^s_{r* c_k}
        where r* = min{r : S^s_{r c1} = v}
        """
        if r2 is None:
            r2 = self.N_r - 1
        
        for r in range(r1, r2 + 1):
            if self[s, r, search_c] == value:
                return self[s, r, return_c]
        
        return SpreadsheetError.NA
    
    def INDEX(self, s: int, r: int, c: int) -> Any:
        """Direct index access: INDEX(S, s, r, c) = S^s_{rc}"""
        return self[s, r, c]
    
    def MATCH(self, value: Any, s: int, c: int, r1: int = 0, r2: int = None) -> Union[int, SpreadsheetError]:
        """
        Find position: MATCH(v, S[s, r1:r2, c]) = r* where S^s_{r* c} = v
        Returns 1-based index for spreadsheet compatibility.
        """
        if r2 is None:
            r2 = self.N_r - 1
        
        for r in range(r1, r2 + 1):
            if self[s, r, c] == value:
                return r + 1  # 1-based
        
        return SpreadsheetError.NA
    
    # =========================================================================
    # §8. INTENT COHESION (Section 7.4)
    # =========================================================================
    
    def compute_intent_cohesion(self, 
                                 expected_pattern: Optional[Callable] = None) -> float:
        """
        Compute Intent Cohesion ratio θ_cohesion = R_Align / R_Drift.
        
        Args:
            expected_pattern: Custom function(i, j) -> bool defining expected dependencies.
                             Default: same column OR same row references.
        
        Returns:
            θ_cohesion value (higher = more aligned with expected patterns)
        """
        A = self.build_dependency_matrix()
        
        if expected_pattern is None:
            # Default: same-column (vertical) or same-row (horizontal) references
            def default_expected(i: int, j: int) -> bool:
                si, ri, ci = self._delinearize(i)
                sj, rj, cj = self._delinearize(j)
                # Same sheet AND (same column OR same row)
                return (si == sj) and (ci == cj or ri == rj)
            
            expected_pattern = default_expected
        
        R_align = 0
        R_drift = 0
        
        for i in range(self.M):
            for j in range(self.M):
                if A[i, j] == 1:
                    if expected_pattern(i, j):
                        R_align += 1
                    else:
                        R_drift += 1
        
        epsilon = 1e-6
        theta = R_align / (R_drift + epsilon)
        
        return theta
    
    def compute_dependency_metrics(self) -> Dict[str, Any]:
        """
        Compute various dependency metrics for analysis.
        """
        A = self.build_dependency_matrix()
        levels = self.compute_levels()
        
        total_refs = np.sum(A)
        cells_with_formulas = len(self._formulas)
        max_depth = self._max_level
        
        # Average dependencies per formula cell
        avg_deps = total_refs / cells_with_formulas if cells_with_formulas > 0 else 0
        
        # Sparsity of dependency matrix
        sparsity = 1 - (total_refs / (self.M * self.M))
        
        return {
            'total_cells': self.M,
            'formula_cells': cells_with_formulas,
            'total_references': int(total_refs),
            'max_dependency_depth': max_depth,
            'avg_dependencies_per_formula': avg_deps,
            'dependency_matrix_sparsity': sparsity,
            'intent_cohesion': self.compute_intent_cohesion()
        }
    
    # =========================================================================
    # §9. SERIALIZATION
    # =========================================================================
    
    def to_dict(self) -> Dict:
        """Serialize tensor to dictionary."""
        return {
            'dimensions': {'N_s': self.N_s, 'N_r': self.N_r, 'N_c': self.N_c},
            'cells': {
                f"{s},{r},{c}": {
                    'value': cell.value,
                    'type': cell.value_type.value,
                    'formula': cell.formula
                }
                for (s, r, c), cell in self._data.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SpreadsheetTensor':
        """Deserialize tensor from dictionary."""
        dims = data['dimensions']
        tensor = cls(dims['N_s'], dims['N_r'], dims['N_c'])
        
        for key, cell_data in data['cells'].items():
            s, r, c = map(int, key.split(','))
            if cell_data['formula']:
                tensor.set_formula(s, r, c, cell_data['formula'])
            else:
                tensor[s, r, c] = cell_data['value']
        
        return tensor
    
    def __repr__(self):
        return f"SpreadsheetTensor(N_s={self.N_s}, N_r={self.N_r}, N_c={self.N_c}, cells={len(self._data)})"


# =============================================================================
# §10. SHEET SLICE VIEW
# =============================================================================

class SheetSlice:
    """
    Rank-2 view into a spreadsheet (single sheet).
    
    Represents S[s, :, :] ∈ V^{N_r × N_c}
    """
    
    def __init__(self, parent: SpreadsheetTensor, sheet_index: int):
        self._parent = parent
        self._s = sheet_index
    
    def __getitem__(self, key: Tuple[int, int]) -> Any:
        r, c = key
        return self._parent[self._s, r, c]
    
    def __setitem__(self, key: Tuple[int, int], value: Any):
        r, c = key
        self._parent[self._s, r, c] = value
    
    def set_formula(self, r: int, c: int, formula: str):
        self._parent.set_formula(self._s, r, c, formula)
    
    def row(self, r: int) -> List[Any]:
        return self._parent.row(self._s, r)
    
    def column(self, c: int) -> List[Any]:
        return self._parent.column(self._s, c)
    
    @property
    def shape(self) -> Tuple[int, int]:
        return (self._parent.N_r, self._parent.N_c)


# =============================================================================
# §11. EXAMPLE USAGE AND TESTS
# =============================================================================

def example_basic():
    """Basic usage example."""
    print("=" * 60)
    print("SPREADSHEET TENSOR THEORY - Basic Example")
    print("=" * 60)
    
    # Create 1-sheet, 5-row, 4-column tensor
    S = SpreadsheetTensor(n_sheets=1, n_rows=5, n_cols=4)
    
    # Set literal values
    # Column A: Names, Column B: Q1, Column C: Q2, Column D: Total
    S[0, 0, 0] = "Product A"
    S[0, 0, 1] = 100
    S[0, 0, 2] = 150
    
    S[0, 1, 0] = "Product B"
    S[0, 1, 1] = 200
    S[0, 1, 2] = 180
    
    S[0, 2, 0] = "Product C"
    S[0, 2, 1] = 300
    S[0, 2, 2] = 320
    
    # Set formulas for totals (Column D)
    S.set_formula(0, 0, 3, "=B1+C1")
    S.set_formula(0, 1, 3, "=B2+C2")
    S.set_formula(0, 2, 3, "=B3+C3")
    
    # Set sum formulas (Row 4)
    S.set_formula(0, 3, 1, "=SUM(B1:B3)")
    S.set_formula(0, 3, 2, "=SUM(C1:C3)")
    S.set_formula(0, 3, 3, "=SUM(D1:D3)")
    
    # Check well-formedness
    print(f"\nWell-formed (acyclic): {S.is_well_formed()}")
    
    # Compute levels
    levels = S.compute_levels()
    print(f"Max dependency depth K: {S._max_level}")
    
    # Evaluate
    S.evaluate()
    
    # Print results
    print("\nEvaluated Tensor S:")
    headers = ["Name", "Q1", "Q2", "Total"]
    print(f"{'':12} " + " ".join(f"{h:>10}" for h in headers))
    print("-" * 55)
    
    for r in range(4):
        row_label = f"Row {r}:" if r < 3 else "Totals:"
        values = [S[0, r, c] for c in range(4)]
        formatted = []
        for v in values:
            if isinstance(v, (int, float)):
                formatted.append(f"{v:>10.0f}")
            elif v is None:
                formatted.append(f"{'':>10}")
            else:
                formatted.append(f"{str(v):>10}")
        print(f"{row_label:12} " + " ".join(formatted))
    
    # Metrics
    print("\nDependency Metrics:")
    metrics = S.compute_dependency_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    return S


def example_incremental():
    """Demonstrate incremental recalculation."""
    print("\n" + "=" * 60)
    print("INCREMENTAL RECALCULATION Example")
    print("=" * 60)
    
    S = SpreadsheetTensor(1, 4, 3)
    
    # Set up: A1=10, A2=20, A3=SUM(A1:A2), B1=A1*2, B2=A2*2, B3=SUM(B1:B2)
    S[0, 0, 0] = 10
    S[0, 1, 0] = 20
    S.set_formula(0, 2, 0, "=A1+A2")
    
    S.set_formula(0, 0, 1, "=A1*2")
    S.set_formula(0, 1, 1, "=A2*2")
    S.set_formula(0, 2, 1, "=B1+B2")
    
    S.evaluate()
    
    print("Initial state:")
    for r in range(3):
        print(f"  Row {r}: A={S[0,r,0]}, B={S[0,r,1]}")
    
    # Change A1
    print("\nChanging A1 from 10 to 50...")
    S[0, 0, 0] = 50
    
    affected = S.recalculate({(0, 0, 0)})
    print(f"Affected cells: {len(affected)} (instead of all {S.M})")
    
    print("\nAfter incremental recalc:")
    for r in range(3):
        print(f"  Row {r}: A={S[0,r,0]}, B={S[0,r,1]}")


def example_intent_cohesion():
    """Demonstrate Intent Cohesion metric."""
    print("\n" + "=" * 60)
    print("INTENT COHESION Example")
    print("=" * 60)
    
    # Well-structured spreadsheet (high cohesion)
    S_good = SpreadsheetTensor(1, 5, 3)
    S_good[0, 0, 0] = 10
    S_good[0, 1, 0] = 20
    S_good[0, 2, 0] = 30
    # Vertical references (same column) - aligned pattern
    S_good.set_formula(0, 3, 0, "=A1+A2+A3")
    # Horizontal reference (same row) - aligned pattern
    S_good.set_formula(0, 0, 1, "=A1*1.1")
    S_good.set_formula(0, 1, 1, "=A2*1.1")
    S_good.set_formula(0, 2, 1, "=A3*1.1")
    
    theta_good = S_good.compute_intent_cohesion()
    print(f"Well-structured spreadsheet θ: {theta_good:.2f}")
    
    # Poorly-structured spreadsheet (low cohesion)
    S_bad = SpreadsheetTensor(1, 5, 5)
    S_bad[0, 0, 0] = 10
    S_bad[0, 1, 2] = 20
    S_bad[0, 2, 4] = 30
    # Cross-references (different row AND column) - drift pattern
    S_bad.set_formula(0, 3, 1, "=A1+C2+E3")  # References across diagonals
    S_bad.set_formula(0, 4, 3, "=A1+D4")
    
    theta_bad = S_bad.compute_intent_cohesion()
    print(f"Poorly-structured spreadsheet θ: {theta_bad:.2f}")
    
    print(f"\nRatio (good/bad): {theta_good/theta_bad:.1f}x more aligned")


if __name__ == "__main__":
    example_basic()
    example_incremental()
    example_intent_cohesion()
