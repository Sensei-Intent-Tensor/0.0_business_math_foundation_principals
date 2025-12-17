"""
Spreadsheet Tensor Theory - Basic Usage Example
Auto-Workspace-AI / Intent Tensor Theory Institute
"""

import sys
sys.path.append('..')
from implementations.spreadsheet_tensor import SpreadsheetTensor

def main():
    # ═══════════════════════════════════════════════════════════════
    # Example 1: Simple Financial Model
    # ═══════════════════════════════════════════════════════════════
    
    print("=" * 60)
    print("Example 1: Simple Financial Model")
    print("=" * 60)
    
    # Create 1-sheet, 6-row, 4-column tensor
    # Columns: A=Item, B=Q1, C=Q2, D=Total
    S = SpreadsheetTensor(n_sheets=1, n_rows=6, n_cols=4)
    
    # Set headers (row 0)
    S[0, 0, 0] = "Item"
    S[0, 0, 1] = "Q1"
    S[0, 0, 2] = "Q2"
    S[0, 0, 3] = "Total"
    
    # Set data (rows 1-3)
    S[0, 1, 0] = "Revenue"
    S[0, 1, 1] = 100000
    S[0, 1, 2] = 120000
    
    S[0, 2, 0] = "Costs"
    S[0, 2, 1] = 60000
    S[0, 2, 2] = 65000
    
    S[0, 3, 0] = "Profit"
    # Profit formulas reference Revenue - Costs
    S.set_formula((0, 3, 1), "=B2-B3")  # Q1 Profit
    S.set_formula((0, 3, 2), "=C2-C3")  # Q2 Profit
    
    # Total formulas
    S.set_formula((0, 1, 3), "=B2+C2")  # Total Revenue
    S.set_formula((0, 2, 3), "=B3+C3")  # Total Costs
    S.set_formula((0, 3, 3), "=D2-D3")  # Total Profit
    
    # Verify well-formedness (no circular refs)
    print(f"\nWell-formed: {S.is_well_formed()}")
    
    # Compute dependency levels
    levels = S.compute_levels()
    print(f"Dependency levels: {S._max_level + 1} levels")
    
    # Evaluate to fixed point
    S.evaluate()
    
    # Print results
    print("\nEvaluated Spreadsheet:")
    print("-" * 50)
    for r in range(4):
        row = [S[0, r, c] for c in range(4)]
        print(f"  Row {r}: {row}")
    
    # Compute Intent Cohesion
    theta = S.compute_intent_cohesion()
    print(f"\nIntent Cohesion θ: {theta:.2f}")
    print("  (High θ = references follow structural patterns)")
    
    # ═══════════════════════════════════════════════════════════════
    # Example 2: Tensor Operations
    # ═══════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 60)
    print("Example 2: Tensor Operations (SUM, SUMIF, VLOOKUP)")
    print("=" * 60)
    
    # Create product sales tensor
    T = SpreadsheetTensor(n_sheets=1, n_rows=6, n_cols=3)
    
    # Column A: Product, Column B: Region, Column C: Sales
    products = ["Widget", "Gadget", "Widget", "Gizmo", "Gadget"]
    regions = ["North", "South", "South", "North", "North"]
    sales = [1000, 1500, 800, 2000, 1200]
    
    for i, (prod, reg, sale) in enumerate(zip(products, regions, sales)):
        T[0, i, 0] = prod
        T[0, i, 1] = reg
        T[0, i, 2] = sale
    
    # SUM: Total sales (contraction over row dimension)
    total = T.SUM(s=0, r1=0, r2=4, c=2)
    print(f"\nSUM(Sales): {total}")
    
    # SUMIF: Sales where Region = "North" (masked contraction)
    north_sales = T.SUMIF(s=0, cond_col=1, cond=lambda x: x == "North", sum_col=2)
    print(f"SUMIF(Region='North', Sales): {north_sales}")
    
    # VLOOKUP: Find sales for "Gadget" (indexed selection)
    gadget_sales = T.VLOOKUP(value="Gadget", s=0, c_search=0, c_return=2)
    print(f"VLOOKUP('Gadget', Sales): {gadget_sales}")
    
    # ═══════════════════════════════════════════════════════════════
    # Example 3: Change Propagation
    # ═══════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 60)
    print("Example 3: Change Propagation (Theorem 4.4)")
    print("=" * 60)
    
    # Using the financial model from Example 1
    # If we change Q1 Revenue, what cells are affected?
    
    changed_cells = {(0, 1, 1)}  # Q1 Revenue
    affected = S.get_affected_cells(changed_cells)
    
    print(f"\nChanged cell: (0, 1, 1) = Q1 Revenue")
    print(f"Affected cells requiring recalculation:")
    for cell in sorted(affected):
        print(f"  {cell}")
    
    print(f"\nTotal affected: {len(affected)} cells")
    print(f"Total cells: {S.M}")
    print(f"Recalculation savings: {100 * (1 - len(affected)/S.M):.1f}%")


if __name__ == "__main__":
    main()
