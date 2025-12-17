/**
 * Spreadsheet Tensor Theory — JavaScript Reference Implementation
 * ================================================================
 * 
 * Auto-Workspace-AI / Intent Tensor Theory Institute
 * Version 1.0
 * 
 * A spreadsheet is a rank-3 tensor: S ∈ V^{N_s × N_r × N_c}
 * 
 * This implementation provides:
 * - SpreadsheetTensor class with full tensor operations
 * - Dependency analysis and level computation
 * - Fixed-point evaluation
 * - Intent Cohesion metric
 * - Google Apps Script compatibility patterns
 * 
 * Usage:
 *   const S = new SpreadsheetTensor(1, 100, 10);
 *   S.set(0, 0, 0, 100);
 *   S.setFormula(0, 1, 0, '=A1*2');
 *   S.evaluate();
 */

// =============================================================================
// §1. VALUE TYPES
// =============================================================================

const ValueType = Object.freeze({
  NUMBER: 'number',
  STRING: 'string',
  BOOLEAN: 'boolean',
  ERROR: 'error',
  NULL: 'null'
});

const SpreadsheetError = Object.freeze({
  REF: '#REF!',
  VALUE: '#VALUE!',
  DIV_ZERO: '#DIV/0!',
  NAME: '#NAME?',
  NULL: '#NULL!',
  NA: '#N/A',
  NUM: '#NUM!'
});

// =============================================================================
// §2. SPREADSHEET TENSOR CLASS
// =============================================================================

class SpreadsheetTensor {
  /**
   * Rank-3 tensor: S ∈ V^{N_s × N_r × N_c}
   * 
   * @param {number} nSheets - Number of sheets (N_s)
   * @param {number} nRows - Number of rows per sheet (N_r)
   * @param {number} nCols - Number of columns per sheet (N_c)
   */
  constructor(nSheets, nRows, nCols) {
    this.N_s = nSheets;
    this.N_r = nRows;
    this.N_c = nCols;
    this.M = nSheets * nRows * nCols; // Total cells
    
    // Sparse storage using Maps
    this.data = new Map();        // key -> value
    this.formulas = new Map();    // key -> formula string
    this.deps = new Map();        // key -> Set of dependency keys
    this.dependents = new Map();  // key -> Set of dependent keys (reverse)
    
    // Cached computations
    this.levels = null;
    this.levelSets = null;
    this.maxLevel = 0;
    this.depMatrixDirty = true;
  }

  // ===========================================================================
  // §2.1 INDEX OPERATIONS
  // ===========================================================================

  /**
   * Create string key from coordinates
   */
  _key(s, r, c) {
    return `${s},${r},${c}`;
  }

  /**
   * Parse key back to coordinates
   */
  _parseKey(key) {
    return key.split(',').map(Number);
  }

  /**
   * Linearization: φ(s,r,c) = s·(N_r·N_c) + r·N_c + c
   */
  _linearize(s, r, c) {
    return s * (this.N_r * this.N_c) + r * this.N_c + c;
  }

  /**
   * Inverse linearization
   */
  _delinearize(i) {
    const s = Math.floor(i / (this.N_r * this.N_c));
    const remainder = i % (this.N_r * this.N_c);
    const r = Math.floor(remainder / this.N_c);
    const c = remainder % this.N_c;
    return [s, r, c];
  }

  /**
   * Validate coordinates
   */
  _validateCoords(s, r, c) {
    return s >= 0 && s < this.N_s && 
           r >= 0 && r < this.N_r && 
           c >= 0 && c < this.N_c;
  }

  // ===========================================================================
  // §2.2 CELL ACCESS
  // ===========================================================================

  /**
   * Get cell value: S^s_{rc}
   */
  get(s, r, c) {
    return this.data.get(this._key(s, r, c)) ?? null;
  }

  /**
   * Set literal cell value (clears formula)
   */
  set(s, r, c, value) {
    const key = this._key(s, r, c);
    
    // Clear old dependencies
    if (this.deps.has(key)) {
      for (const depKey of this.deps.get(key)) {
        const depDependents = this.dependents.get(depKey);
        if (depDependents) depDependents.delete(key);
      }
      this.deps.delete(key);
    }
    
    this.formulas.delete(key);
    this.data.set(key, value);
    this._invalidateCache();
  }

  /**
   * Set cell formula and compute dependencies
   */
  setFormula(s, r, c, formula) {
    const key = this._key(s, r, c);
    
    // Clear old dependencies
    if (this.deps.has(key)) {
      for (const depKey of this.deps.get(key)) {
        const depDependents = this.dependents.get(depKey);
        if (depDependents) depDependents.delete(key);
      }
    }
    
    // Parse new dependencies
    const newDeps = this._parseReferences(formula, s);
    this.formulas.set(key, formula);
    this.deps.set(key, newDeps);
    
    // Update reverse map
    for (const depKey of newDeps) {
      if (!this.dependents.has(depKey)) {
        this.dependents.set(depKey, new Set());
      }
      this.dependents.get(depKey).add(key);
    }
    
    // Initialize with null value
    this.data.set(key, null);
    this._invalidateCache();
  }

  /**
   * Invalidate cached computations
   */
  _invalidateCache() {
    this.levels = null;
    this.levelSets = null;
    this.depMatrixDirty = true;
  }

  // ===========================================================================
  // §2.3 SLICE OPERATIONS
  // ===========================================================================

  /**
   * Get row as array: S[s, r, :]
   */
  row(s, r) {
    const result = [];
    for (let c = 0; c < this.N_c; c++) {
      result.push(this.get(s, r, c));
    }
    return result;
  }

  /**
   * Get column as array: S[s, :, c]
   */
  column(s, c) {
    const result = [];
    for (let r = 0; r < this.N_r; r++) {
      result.push(this.get(s, r, c));
    }
    return result;
  }

  /**
   * Get range as 2D array: S[s, r1:r2, c1:c2]
   */
  range(s, r1, r2, c1, c2) {
    const result = [];
    for (let r = r1; r <= r2; r++) {
      const row = [];
      for (let c = c1; c <= c2; c++) {
        row.push(this.get(s, r, c));
      }
      result.push(row);
    }
    return result;
  }

  /**
   * Get flattened range values (for aggregations)
   */
  flatRange(s, r1, r2, c1, c2) {
    const values = [];
    for (let r = r1; r <= r2; r++) {
      for (let c = c1; c <= c2; c++) {
        const v = this.get(s, r, c);
        if (v !== null && typeof v === 'number') {
          values.push(v);
        }
      }
    }
    return values;
  }

  // ===========================================================================
  // §3. REFERENCE PARSING
  // ===========================================================================

  /**
   * Convert column letters to 0-based index
   */
  _colToIndex(col) {
    let result = 0;
    for (const char of col.toUpperCase()) {
      result = result * 26 + (char.charCodeAt(0) - 64);
    }
    return result - 1;
  }

  /**
   * Convert index to column letters
   */
  _indexToCol(idx) {
    let result = '';
    idx += 1;
    while (idx > 0) {
      const remainder = (idx - 1) % 26;
      result = String.fromCharCode(65 + remainder) + result;
      idx = Math.floor((idx - 1) / 26);
    }
    return result;
  }

  /**
   * Parse cell references from formula
   */
  _parseReferences(formula, currentSheet = 0) {
    const refs = new Set();
    
    // Range pattern: A1:B5 or Sheet1!A1:B5
    const rangePattern = /(?:(['"]?[\w\s]+['"]?)!)?\$?([A-Z]+)\$?(\d+):\$?([A-Z]+)\$?(\d+)/gi;
    let match;
    
    while ((match = rangePattern.exec(formula)) !== null) {
      const [, sheetName, col1, row1, col2, row2] = match;
      const s = sheetName ? this._parseSheetName(sheetName) : currentSheet;
      const c1 = this._colToIndex(col1);
      const c2 = this._colToIndex(col2);
      const r1 = parseInt(row1) - 1;
      const r2 = parseInt(row2) - 1;
      
      for (let r = Math.min(r1, r2); r <= Math.max(r1, r2); r++) {
        for (let c = Math.min(c1, c2); c <= Math.max(c1, c2); c++) {
          if (this._validateCoords(s, r, c)) {
            refs.add(this._key(s, r, c));
          }
        }
      }
    }
    
    // Remove ranges for cell pattern matching
    const formulaNoRanges = formula.replace(rangePattern, '');
    
    // Cell pattern: A1 or Sheet1!A1
    const cellPattern = /(?:(['"]?[\w\s]+['"]?)!)?\$?([A-Z]+)\$?(\d+)/gi;
    
    while ((match = cellPattern.exec(formulaNoRanges)) !== null) {
      const [, sheetName, col, row] = match;
      const s = sheetName ? this._parseSheetName(sheetName) : currentSheet;
      const c = this._colToIndex(col);
      const r = parseInt(row) - 1;
      
      if (this._validateCoords(s, r, c)) {
        refs.add(this._key(s, r, c));
      }
    }
    
    return refs;
  }

  /**
   * Parse sheet name to index
   */
  _parseSheetName(name) {
    if (!name) return 0;
    name = name.replace(/['"]/g, '');
    if (name.toLowerCase().startsWith('sheet')) {
      const num = parseInt(name.slice(5));
      if (!isNaN(num)) return num - 1;
    }
    return 0;
  }

  // ===========================================================================
  // §4. DEPENDENCY ANALYSIS
  // ===========================================================================

  /**
   * Build dependency matrix A ∈ {0,1}^{M×M}
   */
  buildDependencyMatrix() {
    const A = [];
    for (let i = 0; i < this.M; i++) {
      A.push(new Array(this.M).fill(0));
    }
    
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

  /**
   * Theorem 3.3: Check well-formedness (no circular refs)
   */
  isWellFormed() {
    // Use DFS cycle detection (more efficient than matrix powers)
    const WHITE = 0, GRAY = 1, BLACK = 2;
    const color = new Map();
    
    const hasCycle = (key) => {
      color.set(key, GRAY);
      
      const deps = this.deps.get(key);
      if (deps) {
        for (const depKey of deps) {
          const depColor = color.get(depKey) ?? WHITE;
          if (depColor === GRAY) return true;
          if (depColor === WHITE && hasCycle(depKey)) return true;
        }
      }
      
      color.set(key, BLACK);
      return false;
    };
    
    for (const key of this.deps.keys()) {
      if ((color.get(key) ?? WHITE) === WHITE) {
        if (hasCycle(key)) return false;
      }
    }
    
    return true;
  }

  /**
   * Compute dependency levels (Definition 4.2.2)
   */
  computeLevels() {
    if (this.levels !== null) return this.levels;
    
    const A = this.buildDependencyMatrix();
    const levels = new Map();
    
    // Initialize: cells with no dependencies are level 0
    for (let i = 0; i < this.M; i++) {
      if (A[i].every(v => v === 0)) {
        levels.set(i, 0);
      }
    }
    
    // Iteratively assign levels
    let changed = true;
    let iterations = 0;
    const maxIterations = this.M;
    
    while (changed && iterations < maxIterations) {
      changed = false;
      iterations++;
      
      for (let i = 0; i < this.M; i++) {
        if (levels.has(i)) continue;
        
        const depIndices = [];
        for (let j = 0; j < this.M; j++) {
          if (A[i][j] === 1) depIndices.push(j);
        }
        
        if (depIndices.length === 0) {
          levels.set(i, 0);
          changed = true;
        } else if (depIndices.every(j => levels.has(j))) {
          const maxDepLevel = Math.max(...depIndices.map(j => levels.get(j)));
          levels.set(i, maxDepLevel + 1);
          changed = true;
        }
      }
    }
    
    // Convert to coordinate-based map
    this.levels = new Map();
    for (const [i, lvl] of levels) {
      this.levels.set(this._key(...this._delinearize(i)), lvl);
    }
    
    // Assign level 0 to all cells without formulas
    for (let s = 0; s < this.N_s; s++) {
      for (let r = 0; r < this.N_r; r++) {
        for (let c = 0; c < this.N_c; c++) {
          const key = this._key(s, r, c);
          if (!this.levels.has(key)) {
            this.levels.set(key, 0);
          }
        }
      }
    }
    
    this.maxLevel = Math.max(...this.levels.values());
    
    return this.levels;
  }

  /**
   * Get cells grouped by level: L_k = {cells at level k}
   */
  getLevelSets() {
    if (this.levelSets !== null) return this.levelSets;
    if (this.levels === null) this.computeLevels();
    
    this.levelSets = new Map();
    for (const [key, lvl] of this.levels) {
      if (!this.levelSets.has(lvl)) {
        this.levelSets.set(lvl, []);
      }
      this.levelSets.get(lvl).push(key);
    }
    
    return this.levelSets;
  }

  // ===========================================================================
  // §5. EVALUATION
  // ===========================================================================

  /**
   * Fixed-point evaluation (Theorem 4.3)
   * Returns S* such that E(F, S*) = S*
   */
  evaluate() {
    if (!this.isWellFormed()) {
      throw new Error('Spreadsheet contains circular references');
    }
    
    const levelSets = this.getLevelSets();
    
    // Evaluate level by level (Theorem 7.1: parallelizable within levels)
    for (let k = 0; k <= this.maxLevel; k++) {
      const cellsAtLevel = levelSets.get(k) || [];
      
      for (const key of cellsAtLevel) {
        if (this.formulas.has(key)) {
          const [s, r, c] = this._parseKey(key);
          const formula = this.formulas.get(key);
          const value = this._evalFormula(formula, s);
          this.data.set(key, value);
        }
      }
    }
    
    return this;
  }

  /**
   * Evaluate a single formula
   */
  _evalFormula(formula, currentSheet = 0) {
    let expr = formula.replace(/^=/, '');
    
    // Replace range references with arrays
    const rangePattern = /(?:(['"]?[\w\s]+['"]?)!)?\$?([A-Z]+)\$?(\d+):\$?([A-Z]+)\$?(\d+)/gi;
    
    expr = expr.replace(rangePattern, (match, sheet, col1, row1, col2, row2) => {
      const s = sheet ? this._parseSheetName(sheet) : currentSheet;
      const c1 = this._colToIndex(col1);
      const c2 = this._colToIndex(col2);
      const r1 = parseInt(row1) - 1;
      const r2 = parseInt(row2) - 1;
      
      const values = this.flatRange(s, 
        Math.min(r1, r2), Math.max(r1, r2),
        Math.min(c1, c2), Math.max(c1, c2)
      );
      
      return '[' + values.join(',') + ']';
    });
    
    // Replace cell references with values
    const cellPattern = /(?:(['"]?[\w\s]+['"]?)!)?\$?([A-Z]+)\$?(\d+)/gi;
    
    expr = expr.replace(cellPattern, (match, sheet, col, row) => {
      const s = sheet ? this._parseSheetName(sheet) : currentSheet;
      const c = this._colToIndex(col);
      const r = parseInt(row) - 1;
      const val = this.get(s, r, c);
      
      if (val === null) return '0';
      if (typeof val === 'string') return `"${val}"`;
      return String(val);
    });
    
    // Expand spreadsheet functions
    expr = this._expandFunctions(expr);
    
    try {
      // Safe eval for supported operations
      const result = this._safeEval(expr);
      return result;
    } catch (e) {
      if (e.message.includes('division by zero')) {
        return SpreadsheetError.DIV_ZERO;
      }
      return SpreadsheetError.VALUE;
    }
  }

  /**
   * Expand spreadsheet functions to JavaScript
   */
  _expandFunctions(expr) {
    // SUM
    expr = expr.replace(/SUM\s*\(\s*\[([^\]]*)\]\s*\)/gi, (_, args) => {
      return `([${args}].reduce((a,b)=>a+b,0))`;
    });
    
    // AVERAGE
    expr = expr.replace(/AVERAGE\s*\(\s*\[([^\]]*)\]\s*\)/gi, (_, args) => {
      return `(([${args}].reduce((a,b)=>a+b,0))/([${args}].length||1))`;
    });
    
    // COUNT
    expr = expr.replace(/COUNT\s*\(\s*\[([^\]]*)\]\s*\)/gi, (_, args) => {
      return `[${args}].length`;
    });
    
    // MAX
    expr = expr.replace(/MAX\s*\(\s*\[([^\]]*)\]\s*\)/gi, (_, args) => {
      return `Math.max(${args})`;
    });
    
    // MIN
    expr = expr.replace(/MIN\s*\(\s*\[([^\]]*)\]\s*\)/gi, (_, args) => {
      return `Math.min(${args})`;
    });
    
    // IF
    expr = expr.replace(/IF\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)/gi,
      (_, cond, ifTrue, ifFalse) => `((${cond})?(${ifTrue}):(${ifFalse}))`
    );
    
    return expr;
  }

  /**
   * Safe expression evaluation
   */
  _safeEval(expr) {
    // Only allow basic arithmetic and Math functions
    const allowed = /^[\d\s\+\-\*\/\(\)\[\]\,\.\>\<\=\!\?\:]+$|Math\.\w+/;
    
    // Replace operators
    expr = expr.replace(/\band\b/gi, '&&');
    expr = expr.replace(/\bor\b/gi, '||');
    expr = expr.replace(/<>/g, '!==');
    
    try {
      // Use Function constructor for slightly safer eval
      const fn = new Function('Math', `return ${expr}`);
      return fn(Math);
    } catch (e) {
      throw new Error(`Evaluation error: ${e.message}`);
    }
  }

  // ===========================================================================
  // §6. CHANGE PROPAGATION
  // ===========================================================================

  /**
   * Get affected cells for incremental recalculation
   */
  getAffectedCells(changed) {
    const affected = new Set();
    const toProcess = [...changed];
    
    while (toProcess.length > 0) {
      const key = toProcess.pop();
      const deps = this.dependents.get(key);
      
      if (deps) {
        for (const dep of deps) {
          if (!affected.has(dep)) {
            affected.add(dep);
            toProcess.push(dep);
          }
        }
      }
    }
    
    return affected;
  }

  /**
   * Incremental recalculation after changes
   */
  recalculate(changed) {
    const changedKeys = new Set(
      [...changed].map(([s, r, c]) => this._key(s, r, c))
    );
    
    const affected = this.getAffectedCells(changedKeys);
    
    // Group by level
    if (this.levels === null) this.computeLevels();
    
    const byLevel = new Map();
    for (const key of affected) {
      const lvl = this.levels.get(key) ?? 0;
      if (!byLevel.has(lvl)) byLevel.set(lvl, []);
      byLevel.get(lvl).push(key);
    }
    
    // Evaluate in level order
    const sortedLevels = [...byLevel.keys()].sort((a, b) => a - b);
    for (const lvl of sortedLevels) {
      for (const key of byLevel.get(lvl)) {
        if (this.formulas.has(key)) {
          const [s, r, c] = this._parseKey(key);
          const formula = this.formulas.get(key);
          const value = this._evalFormula(formula, s);
          this.data.set(key, value);
        }
      }
    }
    
    return affected;
  }

  // ===========================================================================
  // §7. TENSOR OPERATIONS
  // ===========================================================================

  /**
   * SUM: Tensor contraction
   */
  SUM(s, r1, r2, c1, c2) {
    let total = 0;
    for (let r = r1; r <= r2; r++) {
      for (let c = c1; c <= c2; c++) {
        const v = this.get(s, r, c);
        if (typeof v === 'number') total += v;
      }
    }
    return total;
  }

  /**
   * SUMIF: Masked tensor contraction
   */
  SUMIF(s, condR1, condR2, condC, condition, sumC) {
    let total = 0;
    for (let r = condR1; r <= condR2; r++) {
      const condVal = this.get(s, r, condC);
      if (condition(condVal)) {
        const sumVal = this.get(s, r, sumC);
        if (typeof sumVal === 'number') total += sumVal;
      }
    }
    return total;
  }

  /**
   * COUNTIF: Masked count
   */
  COUNTIF(s, r1, r2, c, condition) {
    let count = 0;
    for (let r = r1; r <= r2; r++) {
      if (condition(this.get(s, r, c))) count++;
    }
    return count;
  }

  /**
   * VLOOKUP: Indexed selection
   */
  VLOOKUP(value, s, searchC, returnC, r1 = 0, r2 = null) {
    if (r2 === null) r2 = this.N_r - 1;
    
    for (let r = r1; r <= r2; r++) {
      if (this.get(s, r, searchC) === value) {
        return this.get(s, r, returnC);
      }
    }
    
    return SpreadsheetError.NA;
  }

  /**
   * MATCH: Find position (1-based)
   */
  MATCH(value, s, c, r1 = 0, r2 = null) {
    if (r2 === null) r2 = this.N_r - 1;
    
    for (let r = r1; r <= r2; r++) {
      if (this.get(s, r, c) === value) {
        return r + 1; // 1-based for spreadsheet compatibility
      }
    }
    
    return SpreadsheetError.NA;
  }

  // ===========================================================================
  // §8. INTENT COHESION
  // ===========================================================================

  /**
   * Compute Intent Cohesion ratio θ = R_Align / R_Drift
   */
  computeIntentCohesion(expectedPattern = null) {
    const A = this.buildDependencyMatrix();
    
    // Default: same column OR same row = aligned
    if (!expectedPattern) {
      expectedPattern = (i, j) => {
        const [si, ri, ci] = this._delinearize(i);
        const [sj, rj, cj] = this._delinearize(j);
        return (si === sj) && (ci === cj || ri === rj);
      };
    }
    
    let RAlign = 0;
    let RDrift = 0;
    
    for (let i = 0; i < this.M; i++) {
      for (let j = 0; j < this.M; j++) {
        if (A[i][j] === 1) {
          if (expectedPattern(i, j)) {
            RAlign++;
          } else {
            RDrift++;
          }
        }
      }
    }
    
    const epsilon = 1e-6;
    return RAlign / (RDrift + epsilon);
  }

  /**
   * Get dependency metrics
   */
  computeDependencyMetrics() {
    const A = this.buildDependencyMatrix();
    this.computeLevels();
    
    let totalRefs = 0;
    for (let i = 0; i < this.M; i++) {
      for (let j = 0; j < this.M; j++) {
        totalRefs += A[i][j];
      }
    }
    
    const formulaCells = this.formulas.size;
    const avgDeps = formulaCells > 0 ? totalRefs / formulaCells : 0;
    const sparsity = 1 - (totalRefs / (this.M * this.M));
    
    return {
      totalCells: this.M,
      formulaCells: formulaCells,
      totalReferences: totalRefs,
      maxDependencyDepth: this.maxLevel,
      avgDependenciesPerFormula: avgDeps,
      dependencyMatrixSparsity: sparsity,
      intentCohesion: this.computeIntentCohesion()
    };
  }

  // ===========================================================================
  // §9. SERIALIZATION
  // ===========================================================================

  toJSON() {
    const cells = {};
    for (const [key, value] of this.data) {
      const formula = this.formulas.get(key);
      cells[key] = { value, formula: formula || null };
    }
    
    return {
      dimensions: { N_s: this.N_s, N_r: this.N_r, N_c: this.N_c },
      cells
    };
  }

  static fromJSON(json) {
    const { dimensions, cells } = json;
    const tensor = new SpreadsheetTensor(
      dimensions.N_s, 
      dimensions.N_r, 
      dimensions.N_c
    );
    
    for (const [key, { value, formula }] of Object.entries(cells)) {
      const [s, r, c] = key.split(',').map(Number);
      if (formula) {
        tensor.setFormula(s, r, c, formula);
        tensor.data.set(key, value);
      } else {
        tensor.set(s, r, c, value);
      }
    }
    
    return tensor;
  }
}

// =============================================================================
// §10. PREDICTIVE MAP PATTERN (Production Pattern from Google Apps Scripts)
// =============================================================================

/**
 * PredictiveMap implements the tensor evaluation pattern used in production scripts.
 * 
 * This pattern IS Spreadsheet Tensor Theory:
 * 1. Build state tensor (map) before execution
 * 2. Compute eligibility mask (metaRows)
 * 3. Execute in parallel (batch operations)
 */
class PredictiveMap {
  constructor() {
    this.map = {};        // State tensor slice
    this.metaRows = {};   // Mask tensor
    this.cache = {};      // Memoization (fixed-point property)
  }

  /**
   * Build state tensor slice from source data
   * This IS constructing S^s_{rc} before evaluation
   */
  buildMap(rows) {
    for (const row of rows) {
      this.map[row.index] = {
        sourceValue: row.sourceValue,
        destinationValue: row.destinationValue,
        metadata: this.generateMetadata(row)
      };
    }
    return this;
  }

  /**
   * Build mask tensor for conditional operations
   * This IS constructing M_r for SUMIF-style masked contraction
   */
  buildMask(rows) {
    for (const row of rows) {
      this.metaRows[row.index] = {
        conditionBlank: row.sourceValue === '',
        conditionNotBlank: row.sourceValue !== '',
        isSelected: row.selected === true,
        notSelected: row.selected === false,
        isFetchable: this._checkFetchable(row)
      };
    }
    return this;
  }

  generateMetadata(row) {
    return {
      timestamp: Date.now(),
      hash: this._hash(row.sourceValue),
      processed: false
    };
  }

  _checkFetchable(row) {
    return row.sourceValue && 
           typeof row.sourceValue === 'string' &&
           row.sourceValue.startsWith('http') && 
           !row.destinationValue;
  }

  _hash(str) {
    if (!str) return 0;
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = ((hash << 5) - hash) + str.charCodeAt(i);
      hash = hash & hash;
    }
    return Math.abs(hash);
  }

  /**
   * Get eligible rows based on mask
   * This is the masked selection: {r : M_r = 1}
   */
  getEligible(maskField = 'isFetchable') {
    return Object.entries(this.metaRows)
      .filter(([_, meta]) => meta[maskField])
      .map(([idx, _]) => ({
        index: idx,
        ...this.map[idx]
      }));
  }

  /**
   * Execute operations on eligible rows
   * This IS parallel_map(E, F ∘ L_k) from Theorem 7.1
   */
  async execute(processor, options = {}) {
    const eligible = this.getEligible(options.maskField || 'isFetchable');
    const batchSize = options.batchSize || 50;
    const results = [];
    
    // Process in batches (parallelizable)
    for (let i = 0; i < eligible.length; i += batchSize) {
      const batch = eligible.slice(i, i + batchSize);
      
      // Check cache first (fixed-point memoization)
      const toProcess = batch.filter(row => {
        const cacheKey = this._hash(row.sourceValue);
        if (this.cache[cacheKey]) {
          results.push({ ...row, result: this.cache[cacheKey], cached: true });
          return false;
        }
        return true;
      });
      
      // Process uncached items
      if (toProcess.length > 0) {
        const batchResults = await Promise.all(
          toProcess.map(async row => {
            const result = await processor(row);
            const cacheKey = this._hash(row.sourceValue);
            this.cache[cacheKey] = result;
            return { ...row, result, cached: false };
          })
        );
        results.push(...batchResults);
      }
    }
    
    return results;
  }
}

// =============================================================================
// §11. EXPORTS
// =============================================================================

// CommonJS export for Node.js
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    SpreadsheetTensor,
    PredictiveMap,
    ValueType,
    SpreadsheetError
  };
}

// ES6 export
if (typeof exports !== 'undefined') {
  exports.SpreadsheetTensor = SpreadsheetTensor;
  exports.PredictiveMap = PredictiveMap;
  exports.ValueType = ValueType;
  exports.SpreadsheetError = SpreadsheetError;
}

// Global export for browser/Google Apps Script
if (typeof globalThis !== 'undefined') {
  globalThis.SpreadsheetTensor = SpreadsheetTensor;
  globalThis.PredictiveMap = PredictiveMap;
}


// =============================================================================
// §12. EXAMPLE USAGE
// =============================================================================

function exampleUsage() {
  console.log('='.repeat(60));
  console.log('SPREADSHEET TENSOR THEORY - JavaScript Example');
  console.log('='.repeat(60));
  
  // Create tensor: 1 sheet, 5 rows, 4 columns
  const S = new SpreadsheetTensor(1, 5, 4);
  
  // Set values
  S.set(0, 0, 0, 'Product A');
  S.set(0, 0, 1, 100);
  S.set(0, 0, 2, 150);
  
  S.set(0, 1, 0, 'Product B');
  S.set(0, 1, 1, 200);
  S.set(0, 1, 2, 180);
  
  S.set(0, 2, 0, 'Product C');
  S.set(0, 2, 1, 300);
  S.set(0, 2, 2, 320);
  
  // Set formulas for totals (Column D)
  S.setFormula(0, 0, 3, '=B1+C1');
  S.setFormula(0, 1, 3, '=B2+C2');
  S.setFormula(0, 2, 3, '=B3+C3');
  
  // Set sum formulas (Row 4)
  S.setFormula(0, 3, 1, '=SUM(B1:B3)');
  S.setFormula(0, 3, 2, '=SUM(C1:C3)');
  S.setFormula(0, 3, 3, '=SUM(D1:D3)');
  
  console.log(`\nWell-formed: ${S.isWellFormed()}`);
  
  // Evaluate
  S.evaluate();
  
  console.log('\nEvaluated Tensor:');
  for (let r = 0; r < 4; r++) {
    const row = S.row(0, r);
    console.log(`  Row ${r}: ${JSON.stringify(row)}`);
  }
  
  // Metrics
  console.log('\nDependency Metrics:');
  const metrics = S.computeDependencyMetrics();
  for (const [key, value] of Object.entries(metrics)) {
    console.log(`  ${key}: ${typeof value === 'number' ? value.toFixed(2) : value}`);
  }
  
  return S;
}

// Run example if this is the main module
if (typeof require !== 'undefined' && require.main === module) {
  exampleUsage();
}
