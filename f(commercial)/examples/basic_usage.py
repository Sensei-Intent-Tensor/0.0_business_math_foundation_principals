"""
Basic Usage Example - Commercial Collapse Geometry

Demonstrates core CCG concepts with simple examples.
"""

import sys
sys.path.append('..')
from implementations.commercial_tensor import (
    IntentVector, TransactionCollapse, ComplexIntentTensor,
    AdaptiveFitnessCalculator, MarketFieldEvolution, SurvivalAnalyzer,
    quick_fitness_check, check_survival
)

print("=" * 60)
print("COMMERCIAL COLLAPSE GEOMETRY - BASIC USAGE")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# EXAMPLE 1: Transaction Collapse Analysis
# ─────────────────────────────────────────────────────────────
print("\n1. TRANSACTION COLLAPSE ANALYSIS")
print("-" * 40)

# Define buyer and seller intents
seller = IntentVector(
    price=0.8,      # Asking price (normalized)
    trust=0.7,      # Trust offered
    timing=0.6,     # Timing flexibility
    relevance=0.9   # Product relevance
)

buyer = IntentVector(
    price=0.75,     # Willing to pay
    trust=0.65,     # Trust required
    timing=0.7,     # Timing need
    relevance=0.85  # Relevance sought
)

# Analyze collapse potential
collapse = TransactionCollapse(epsilon_market=0.15)
analysis = collapse.gap_analysis(buyer, seller)

print(f"Intent Gap: {analysis['total_gap']:.4f}")
print(f"Threshold: {analysis['threshold']:.4f}")
print(f"Will Collapse (Transact): {analysis['will_collapse']}")
print(f"Collapse Probability: {analysis['collapse_probability']:.2%}")

if analysis['blocking_components']:
    print(f"Blocking Components: {analysis['blocking_components']}")

# ─────────────────────────────────────────────────────────────
# EXAMPLE 2: Complex Intent Tensor
# ─────────────────────────────────────────────────────────────
print("\n2. COMPLEX INTENT TENSOR")
print("-" * 40)

# Create entity with current yield and adaptation capacity
entity = ComplexIntentTensor(I_site=1.0, S_imag=0.6)

print(f"I_site (Current Yield): {entity.I_site}")
print(f"S_imag (Adaptation Capacity): {entity.S_imag}")
print(f"Total Energy |I_C|: {entity.magnitude:.4f}")
print(f"Phase: {entity.phase_degrees:.1f}° (0°=Fruit, 90°=Roots)")

# Optimal allocation for 30% transition probability
allocation = entity.allocation_recommendation(
    transition_probability=0.3,
    total_resources=100
)
print(f"\nOptimal Allocation (p=0.3):")
print(f"  I_site: {allocation['I_site']:.1f}")
print(f"  S_imag: {allocation['S_imag']:.1f}")
print(f"  Ratio: {allocation['ratio']:.3f}")

# ─────────────────────────────────────────────────────────────
# EXAMPLE 3: Adaptive Fitness Score
# ─────────────────────────────────────────────────────────────
print("\n3. ADAPTIVE FITNESS SCORE")
print("-" * 40)

result = quick_fitness_check(
    I_site=1.0,
    S_imag=0.6,
    coherence=1.2,    # Slightly specific
    potential=0.9,    # Good flexibility
    authority=0.8     # Moderate authority
)

print(f"Adaptive Fitness: {result['fitness']:.4f}")
print(f"Components:")
print(f"  |I_C|: {result['components']['magnitude']:.4f}")
print(f"  Ψ (ICWHE): {result['components']['psi']:.4f}")
print(f"  Authority: {result['components']['authority']:.4f}")
print(f"ICWHE Status: {result['icwhe']['status']}")
print(f"Recommendation: {result['icwhe']['recommendation']}")

# ─────────────────────────────────────────────────────────────
# EXAMPLE 4: Survival Analysis
# ─────────────────────────────────────────────────────────────
print("\n4. SURVIVAL ANALYSIS")
print("-" * 40)

for s_imag in [0.3, 0.5, 0.7]:
    survival = check_survival(s_imag, years=10)
    print(f"S_imag={s_imag}: {survival['survival_probability']:.1%} (10yr), "
          f"Median={survival['median_survival']:.1f}yr")

# ─────────────────────────────────────────────────────────────
# EXAMPLE 5: Market Evolution with Transitions
# ─────────────────────────────────────────────────────────────
print("\n5. MARKET EVOLUTION SIMULATION")
print("-" * 40)

from implementations.commercial_tensor import PhaseTransition

simulator = MarketFieldEvolution(
    r=0.05,           # 5% base growth
    delta_base=0.07,  # 7% obsolescence
    k=1.0             # Adaptability scaling
)

# Simulate 20 years with two phase transitions
transitions = [
    PhaseTransition(time=5, delta=0.3, name="Digital Disruption"),
    PhaseTransition(time=15, delta=0.4, name="AI Revolution")
]

result = simulator.evolve(I_0=1.0, S_0=0.6, T=20, transitions=transitions)

print(f"Initial: I_site=1.0, S_imag=0.6")
print(f"Transitions: {len(transitions)}")
print(f"Final State:")
print(f"  I_site: {result['final_state']['I_site']:.4f}")
print(f"  S_imag: {result['final_state']['S_imag']:.4f}")
print(f"  |I_C|: {result['final_state']['magnitude']:.4f}")
print(f"Survived: {result['survival']}")

print("\n" + "=" * 60)
print("HAIL MATH.")
print("=" * 60)
