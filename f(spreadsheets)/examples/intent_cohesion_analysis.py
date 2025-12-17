"""
Spreadsheet Tensor Theory - Intent Cohesion Analysis
Auto-Workspace-AI / Intent Tensor Theory Institute

This example demonstrates how to compute and interpret the Intent Cohesion
metric (θ_cohesion) from Intent Tensor Theory applied to spreadsheets.
"""

import sys
sys.path.append('..')
from implementations.spreadsheet_tensor import SpreadsheetTensor

def create_well_structured_spreadsheet():
    """
    Create a spreadsheet following structural conventions:
    - Vertical references (same column) for rollups
    - Horizontal references (same row) for calculations
    """
    S = SpreadsheetTensor(n_sheets=1, n_rows=5, n_cols=4)
    
    # Headers
    S[0, 0, 0] = "Category"
    S[0, 0, 1] = "Jan"
    S[0, 0, 2] = "Feb"
    S[0, 0, 3] = "Total"
    
    # Data rows
    S[0, 1, 0] = "Sales"
    S[0, 1, 1] = 1000
    S[0, 1, 2] = 1200
    S.set_formula((0, 1, 3), "=B2+C2")  # Horizontal: same row
    
    S[0, 2, 0] = "Expenses"
    S[0, 2, 1] = 600
    S[0, 2, 2] = 650
    S.set_formula((0, 2, 3), "=B3+C3")  # Horizontal: same row
    
    S[0, 3, 0] = "Net"
    S.set_formula((0, 3, 1), "=B2-B3")  # Vertical: same column
    S.set_formula((0, 3, 2), "=C2-C3")  # Vertical: same column
    S.set_formula((0, 3, 3), "=D2-D3")  # Vertical: same column
    
    return S


def create_poorly_structured_spreadsheet():
    """
    Create a spreadsheet with ad-hoc cross-references:
    - Diagonal references
    - Cross-sheet patterns (if multi-sheet)
    - Non-standard reference directions
    """
    S = SpreadsheetTensor(n_sheets=1, n_rows=5, n_cols=4)
    
    # Same data setup
    S[0, 0, 0] = "Category"
    S[0, 0, 1] = "Jan"
    S[0, 0, 2] = "Feb"
    S[0, 0, 3] = "Total"
    
    S[0, 1, 0] = "Sales"
    S[0, 1, 1] = 1000
    S[0, 1, 2] = 1200
    
    S[0, 2, 0] = "Expenses"
    S[0, 2, 1] = 600
    S[0, 2, 2] = 650
    
    S[0, 3, 0] = "Net"
    
    # BAD: Diagonal/cross references (violates structural convention)
    S.set_formula((0, 1, 3), "=B2+C3")    # Mixes rows - DRIFT
    S.set_formula((0, 2, 3), "=C2+B3")    # Mixes rows - DRIFT
    S.set_formula((0, 3, 1), "=B2-C3")    # Diagonal - DRIFT
    S.set_formula((0, 3, 2), "=C2-B3")    # Diagonal - DRIFT
    S.set_formula((0, 3, 3), "=B2-C3+D2") # Chaos - DRIFT
    
    return S


def analyze_spreadsheet(S, name):
    """Analyze a spreadsheet's structural quality."""
    print(f"\n{'='*60}")
    print(f"Analysis: {name}")
    print(f"{'='*60}")
    
    # Verify well-formedness
    is_wf = S.is_well_formed()
    print(f"\n1. Well-Formed (no circular refs): {is_wf}")
    
    if not is_wf:
        print("   ERROR: Cannot analyze - contains circular references")
        return
    
    # Compute levels
    levels = S.compute_levels()
    print(f"\n2. Dependency Structure:")
    print(f"   Max dependency depth (K): {S._max_level}")
    print(f"   Parallelization potential: {S._max_level + 1} sequential steps")
    
    # Compute Intent Cohesion
    theta = S.compute_intent_cohesion()
    print(f"\n3. Intent Cohesion Analysis:")
    print(f"   θ_cohesion = {theta:.4f}")
    
    # Interpret the metric
    if theta > 10:
        quality = "EXCELLENT"
        interpretation = "References strongly follow structural patterns"
    elif theta > 1:
        quality = "GOOD"
        interpretation = "More aligned than drifting references"
    elif theta > 0.5:
        quality = "MODERATE"
        interpretation = "Roughly equal aligned and drifting references"
    else:
        quality = "POOR"
        interpretation = "Significant structural drift - consider refactoring"
    
    print(f"   Quality: {quality}")
    print(f"   Interpretation: {interpretation}")
    
    # Evaluate and show results
    S.evaluate()
    print(f"\n4. Evaluated Values:")
    for r in range(4):
        row = [S[0, r, c] for c in range(4)]
        print(f"   Row {r}: {row}")


def main():
    print("=" * 60)
    print("INTENT COHESION ANALYSIS")
    print("From Intent Tensor Theory → Spreadsheet Tensor Theory")
    print("=" * 60)
    
    print("""
The Intent Cohesion ratio θ measures structural alignment:

    θ_cohesion = R_Align / R_Drift

Where:
  R_Align = references following expected patterns
            (same-column for vertical rollups,
             same-row for horizontal calculations)
  R_Drift = references violating expected patterns
            (diagonal, cross-references)

High θ → Well-structured, maintainable spreadsheet
Low θ  → Ad-hoc structure, potential technical debt
""")
    
    # Analyze both spreadsheets
    well_structured = create_well_structured_spreadsheet()
    analyze_spreadsheet(well_structured, "Well-Structured Spreadsheet")
    
    poorly_structured = create_poorly_structured_spreadsheet()
    analyze_spreadsheet(poorly_structured, "Poorly-Structured Spreadsheet")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    theta_good = well_structured.compute_intent_cohesion()
    theta_bad = poorly_structured.compute_intent_cohesion()
    
    print(f"""
Well-Structured θ:   {theta_good:.4f}
Poorly-Structured θ: {theta_bad:.4f}
Ratio:               {theta_good/theta_bad:.1f}x better

The Intent Cohesion metric makes structural quality COMPUTABLE.
This enables automated spreadsheet quality assessment.
""")


if __name__ == "__main__":
    main()
