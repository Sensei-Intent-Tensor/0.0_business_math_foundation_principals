"""
Commercial Collapse Geometry - Python Implementation

f(commercial) — Core Classes for Market Transaction Physics

Auto-Workspace-AI | Sensei Intent Tensor
Version 1.0 | December 2025

This module provides the mathematical infrastructure for:
- Transaction collapse analysis
- Complex intent tensor management
- Adaptive fitness scoring
- Market field evolution simulation
- Survival analysis

HAIL MATH.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import warnings


# ═══════════════════════════════════════════════════════════════════════════════
# PART I: FOUNDATIONAL DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IntentVector:
    """
    Represents the intent potential of a commercial actor.
    
    Components:
        price: Acceptable price point (normalized)
        trust: Required/offered trust level [0, 1]
        timing: Timing flexibility [0, 1]
        relevance: Relevance match [0, 1]
        custom: Additional domain-specific components
    """
    price: float
    trust: float
    timing: float
    relevance: float
    custom: Optional[Dict[str, float]] = None
    
    def to_vector(self) -> np.ndarray:
        """Convert to numpy array for calculations."""
        base = np.array([self.price, self.trust, self.timing, self.relevance])
        if self.custom:
            custom_vals = np.array(list(self.custom.values()))
            return np.concatenate([base, custom_vals])
        return base
    
    @classmethod
    def from_vector(cls, vec: np.ndarray, custom_keys: Optional[List[str]] = None):
        """Construct from numpy array."""
        custom = None
        if len(vec) > 4 and custom_keys:
            custom = dict(zip(custom_keys, vec[4:]))
        return cls(
            price=vec[0],
            trust=vec[1],
            timing=vec[2],
            relevance=vec[3],
            custom=custom
        )


class Polarity(Enum):
    """Commercial polarity type."""
    SELLER = +1  # Outward, releasing
    BUYER = -1   # Inward, acquiring


@dataclass
class CommercialActor:
    """
    A participant in the commercial field.
    
    Attributes:
        intent: The actor's intent vector
        polarity: SELLER (+∇Φ) or BUYER (-∇Φ)
        authority: Trust/credibility score
    """
    intent: IntentVector
    polarity: Polarity
    authority: float = 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# PART II: TRANSACTION COLLAPSE
# ═══════════════════════════════════════════════════════════════════════════════

class TransactionCollapse:
    """
    Evaluates whether buyer-seller configurations collapse into transactions.
    
    Implements Theorem 1.2: Transaction Collapse Theorem
    
    Transaction Collapse ⟺ ||ΔΨ_transaction|| < ε_market
    """
    
    def __init__(self, epsilon_market: float = 0.1, 
                 component_weights: Optional[Dict[str, float]] = None):
        """
        Initialize collapse evaluator.
        
        Args:
            epsilon_market: Market threshold for collapse
            component_weights: Weights for gap components (default: equal)
        """
        self.epsilon_market = epsilon_market
        self.component_weights = component_weights or {
            'price': 1.0,
            'trust': 1.0,
            'timing': 1.0,
            'relevance': 1.0
        }
    
    def compute_intent_gap(self, buyer: IntentVector, 
                          seller: IntentVector) -> Tuple[float, Dict[str, float]]:
        """
        Compute the intent gap ||ΔΨ|| = ||Φ_buyer - Φ_seller||
        
        Returns:
            Tuple of (total_gap, component_gaps)
        """
        buyer_vec = buyer.to_vector()
        seller_vec = seller.to_vector()
        
        diff = buyer_vec - seller_vec
        
        # Component gaps
        component_gaps = {
            'price': abs(diff[0]),
            'trust': abs(diff[1]),
            'timing': abs(diff[2]),
            'relevance': abs(diff[3])
        }
        
        # Weighted Euclidean norm
        weights = np.array([
            self.component_weights.get('price', 1.0),
            self.component_weights.get('trust', 1.0),
            self.component_weights.get('timing', 1.0),
            self.component_weights.get('relevance', 1.0)
        ])
        
        # Extend weights if custom components exist
        if len(diff) > 4:
            extra_weights = np.ones(len(diff) - 4)
            weights = np.concatenate([weights, extra_weights])
        
        weighted_diff = diff * np.sqrt(weights)
        total_gap = np.linalg.norm(weighted_diff)
        
        return total_gap, component_gaps
    
    def will_collapse(self, buyer: IntentVector, 
                     seller: IntentVector) -> bool:
        """
        Determine if transaction will collapse (execute).
        
        Transaction Collapse ⟺ ||ΔΨ|| < ε_market
        """
        gap, _ = self.compute_intent_gap(buyer, seller)
        return gap < self.epsilon_market
    
    def collapse_probability(self, buyer: IntentVector, 
                            seller: IntentVector,
                            sigma: float = 0.02) -> float:
        """
        Compute soft collapse probability using Gaussian model.
        
        P(collapse) = exp(-||ΔΨ||² / 2σ²)
        
        This models uncertainty in the exact threshold.
        """
        gap, _ = self.compute_intent_gap(buyer, seller)
        return np.exp(-gap**2 / (2 * sigma**2))
    
    def gap_analysis(self, buyer: IntentVector, 
                    seller: IntentVector) -> Dict:
        """
        Comprehensive gap analysis with recommendations.
        """
        total_gap, component_gaps = self.compute_intent_gap(buyer, seller)
        will_collapse = total_gap < self.epsilon_market
        
        # Identify blocking components
        blocking = []
        for comp, gap in component_gaps.items():
            # Component contributes more than its share
            if gap > self.epsilon_market / np.sqrt(4):
                blocking.append((comp, gap))
        
        blocking.sort(key=lambda x: -x[1])  # Sort by gap descending
        
        return {
            'total_gap': total_gap,
            'threshold': self.epsilon_market,
            'will_collapse': will_collapse,
            'margin': self.epsilon_market - total_gap,
            'component_gaps': component_gaps,
            'blocking_components': blocking,
            'collapse_probability': self.collapse_probability(buyer, seller)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PART III: COMPLEX INTENT TENSOR
# ═══════════════════════════════════════════════════════════════════════════════

class ComplexIntentTensor:
    """
    The Complex Intent Tensor: I_C = I_site + i · S_imag
    
    I_site (Real): Current, measurable, perishable success (The Fruit)
    S_imag (Imaginary): Structural capacity for future adaptation (The Roots)
    
    Implements the dual-component model of commercial energy.
    """
    
    def __init__(self, I_site: float, S_imag: float):
        """
        Initialize complex intent tensor.
        
        Args:
            I_site: Current yield (real component)
            S_imag: Adaptation capacity (imaginary component)
        """
        if I_site < 0:
            warnings.warn("I_site < 0 is unusual; typically non-negative")
        if S_imag < 0:
            warnings.warn("S_imag < 0 indicates negative adaptability")
        
        self.I_site = I_site
        self.S_imag = S_imag
    
    @property
    def magnitude(self) -> float:
        """
        Total commercial energy: |I_C| = √(I_site² + S_imag²)
        """
        return np.sqrt(self.I_site**2 + self.S_imag**2)
    
    @property
    def phase(self) -> float:
        """
        Phase angle in complex plane: θ = arctan(S_imag / I_site)
        
        θ → 0: Fruit-dominant (high current yield, low adaptability)
        θ → π/2: Root-dominant (low current yield, high adaptability)
        """
        return np.arctan2(self.S_imag, self.I_site)
    
    @property
    def phase_degrees(self) -> float:
        """Phase angle in degrees."""
        return np.degrees(self.phase)
    
    @property
    def as_complex(self) -> complex:
        """Return as Python complex number."""
        return complex(self.I_site, self.S_imag)
    
    def harvest(self, omega_t: float) -> float:
        """
        Market-cycle harvest function.
        
        Harvest(t) = I_site · cos(ωt) + S_imag · sin(ωt)
        
        Args:
            omega_t: Market cycle phase (ω × t)
            
        Returns:
            Harvest value for current market phase
            
        Interpretation:
            - ωt ≈ 0 (stable market): Harvest ≈ I_site
            - ωt ≈ π/2 (transition): Harvest ≈ S_imag
        """
        return self.I_site * np.cos(omega_t) + self.S_imag * np.sin(omega_t)
    
    def optimal_ratio(self, transition_probability: float) -> float:
        """
        Optimal S_imag/I_site ratio for given transition probability.
        
        From Theorem 2.2: ratio = √(p / (1-p))
        
        Args:
            transition_probability: Probability of phase transition (0, 1)
            
        Returns:
            Optimal ratio of S_imag to I_site
        """
        p = transition_probability
        if p <= 0 or p >= 1:
            raise ValueError("Probability must be in (0, 1)")
        return np.sqrt(p / (1 - p))
    
    def allocation_recommendation(self, transition_probability: float, 
                                  total_resources: float) -> Dict[str, float]:
        """
        Recommend I_site vs S_imag allocation.
        
        Args:
            transition_probability: Expected transition probability
            total_resources: Total available resources
            
        Returns:
            Dict with recommended I_site and S_imag allocations
        """
        ratio = self.optimal_ratio(transition_probability)
        
        # Solve: S = ratio × I, I + S = R
        # I(1 + ratio) = R
        I_recommended = total_resources / (1 + ratio)
        S_recommended = total_resources - I_recommended
        
        return {
            'I_site': I_recommended,
            'S_imag': S_recommended,
            'ratio': ratio,
            'transition_probability': transition_probability
        }
    
    def __repr__(self):
        return f"ComplexIntentTensor(I_site={self.I_site:.4f}, S_imag={self.S_imag:.4f}, |I_C|={self.magnitude:.4f})"
    
    def __add__(self, other: 'ComplexIntentTensor') -> 'ComplexIntentTensor':
        """Add two tensors."""
        return ComplexIntentTensor(
            self.I_site + other.I_site,
            self.S_imag + other.S_imag
        )
    
    def __mul__(self, scalar: float) -> 'ComplexIntentTensor':
        """Scalar multiplication."""
        return ComplexIntentTensor(
            self.I_site * scalar,
            self.S_imag * scalar
        )
    
    __rmul__ = __mul__


# ═══════════════════════════════════════════════════════════════════════════════
# PART IV: ICWHE CONSTRAINT
# ═══════════════════════════════════════════════════════════════════════════════

class ICWHEConstraint:
    """
    Inverse Cartesian Website Heisenberg Equation (ICWHE)
    
    Δ(Coherence) · Δ(Potential) ≥ h
    
    Models the fundamental trade-off between specificity and flexibility.
    """
    
    def __init__(self, h: float = 1.0):
        """
        Initialize ICWHE constraint.
        
        Args:
            h: Market uncertainty constant (domain-specific)
        """
        self.h = h
    
    def compute_Pi(self, coherence: float, potential: float) -> float:
        """
        Compute the ICWHE product Π = Δ(Coherence) · Δ(Potential)
        """
        return coherence * potential
    
    def compute_psi(self, coherence: float, potential: float) -> float:
        """
        Compute validity function Ψ(Π, h) = min(1, Π/h, h/Π)
        
        Returns:
            Ψ value in [0, 1]
            1 = optimal balance (Π = h)
            <1 = penalty for imbalance
        """
        Pi = self.compute_Pi(coherence, potential)
        
        if Pi <= 0:
            return 0.0
        
        return min(1.0, Pi / self.h, self.h / Pi)
    
    def satisfies_constraint(self, coherence: float, potential: float) -> bool:
        """
        Check if configuration satisfies ICWHE constraint.
        
        Π ≥ h is required for validity.
        """
        return self.compute_Pi(coherence, potential) >= self.h
    
    def deviation_from_equilibrium(self, coherence: float, 
                                   potential: float) -> Dict[str, float]:
        """
        Analyze deviation from optimal equilibrium surface Π = h.
        
        Returns:
            Analysis dict with deviation metrics
        """
        Pi = self.compute_Pi(coherence, potential)
        psi = self.compute_psi(coherence, potential)
        
        if Pi < self.h:
            status = "BRITTLE"
            recommendation = "Increase flexibility (Potential)"
        elif Pi > self.h:
            status = "VAGUE"
            recommendation = "Increase specificity (Coherence)"
        else:
            status = "OPTIMAL"
            recommendation = "Maintain current balance"
        
        return {
            'Pi': Pi,
            'h': self.h,
            'psi': psi,
            'deviation': Pi - self.h,
            'relative_deviation': (Pi - self.h) / self.h,
            'status': status,
            'recommendation': recommendation
        }
    
    def optimize_allocation(self, total_capacity: float) -> Tuple[float, float]:
        """
        Find optimal Coherence/Potential split for given total capacity.
        
        Assumes Coherence + Potential = total_capacity and Π = h.
        
        Returns:
            (optimal_coherence, optimal_potential)
        """
        # Maximize Coherence × Potential subject to C + P = T
        # Solution: C = P = T/2 when targeting Π = h
        
        # But we need C × P = h
        # With C + P = T: C(T-C) = h → C² - TC + h = 0
        # C = (T ± √(T² - 4h)) / 2
        
        discriminant = total_capacity**2 - 4 * self.h
        
        if discriminant < 0:
            # Cannot achieve Π = h with this capacity
            # Return balanced split
            warnings.warn(f"Cannot achieve Π = h with capacity {total_capacity}")
            return total_capacity / 2, total_capacity / 2
        
        sqrt_disc = np.sqrt(discriminant)
        C1 = (total_capacity + sqrt_disc) / 2
        C2 = (total_capacity - sqrt_disc) / 2
        
        # Return the more balanced solution
        if abs(C1 - total_capacity/2) < abs(C2 - total_capacity/2):
            return C1, total_capacity - C1
        else:
            return C2, total_capacity - C2


# ═══════════════════════════════════════════════════════════════════════════════
# PART V: ADAPTIVE FITNESS CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveFitnessCalculator:
    """
    Computes the Adaptive Fitness Score:
    
    f_adaptive = |I_C| · Ψ(Π, h) · Authority_hybrid
    
    This is the master equation for commercial survivability.
    """
    
    def __init__(self, h: float = 1.0, alpha: float = 0.5):
        """
        Initialize fitness calculator.
        
        Args:
            h: ICWHE constant
            alpha: Weight for static vs dynamic authority [0, 1]
                   α = 1: Pure static (backlinks, citations)
                   α = 0: Pure dynamic (engagement, mentions)
        """
        self.icwhe = ICWHEConstraint(h)
        self.alpha = alpha
    
    def compute_authority(self, static: float, dynamic: float) -> float:
        """
        Compute hybrid authority score.
        
        Authority_hybrid = α · Static + (1-α) · Dynamic
        """
        return self.alpha * static + (1 - self.alpha) * dynamic
    
    def compute_fitness(self, I_C: ComplexIntentTensor,
                       coherence: float, potential: float,
                       authority_static: float, 
                       authority_dynamic: float) -> float:
        """
        Compute the Adaptive Fitness Score.
        
        f_adaptive = |I_C| · Ψ(Π, h) · Authority_hybrid
        
        Args:
            I_C: Complex intent tensor
            coherence: Δ(Coherence) value
            potential: Δ(Potential) value
            authority_static: Static authority (backlinks, etc.)
            authority_dynamic: Dynamic authority (engagement, etc.)
            
        Returns:
            Adaptive fitness score
        """
        magnitude = I_C.magnitude
        psi = self.icwhe.compute_psi(coherence, potential)
        authority = self.compute_authority(authority_static, authority_dynamic)
        
        return magnitude * psi * authority
    
    def full_analysis(self, I_C: ComplexIntentTensor,
                     coherence: float, potential: float,
                     authority_static: float,
                     authority_dynamic: float) -> Dict:
        """
        Comprehensive fitness analysis with all components.
        """
        magnitude = I_C.magnitude
        psi = self.icwhe.compute_psi(coherence, potential)
        authority = self.compute_authority(authority_static, authority_dynamic)
        fitness = magnitude * psi * authority
        
        icwhe_analysis = self.icwhe.deviation_from_equilibrium(coherence, potential)
        
        return {
            'fitness': fitness,
            'components': {
                'magnitude': magnitude,
                'psi': psi,
                'authority': authority
            },
            'I_C': {
                'I_site': I_C.I_site,
                'S_imag': I_C.S_imag,
                'phase_degrees': I_C.phase_degrees
            },
            'icwhe': icwhe_analysis,
            'authority_breakdown': {
                'static': authority_static,
                'dynamic': authority_dynamic,
                'alpha': self.alpha
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PART VI: MARKET FIELD EVOLUTION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PhaseTransition:
    """Represents a market phase transition."""
    time: float
    delta: float  # Transition severity
    name: str = ""


class MarketFieldEvolution:
    """
    Simulates the evolution of commercial entities in a dynamic market field.
    
    Implements the evolution equations:
    
    dI_site/dt = r · I - δ · I · (1 - k · S_imag)
    dS_imag/dt = κ · |I_C| · (1 - σ(S)) + λ · ∇·A
    
    Where:
        r = base growth rate
        δ = obsolescence rate
        k = adaptability scaling
        κ = intrinsic adaptation rate
        λ = external stimulus factor
        σ(S) = S / (S + S_half) saturation function
    """
    
    def __init__(self, r: float = 0.05, delta_base: float = 0.07,
                 k: float = 1.0, kappa: float = 0.1,
                 lambda_ext: float = 0.01, S_half: float = 1.0):
        """
        Initialize market evolution simulator.
        
        Args:
            r: Base growth rate (typically 0.03-0.08)
            delta_base: Base obsolescence rate (typically 0.05-0.10)
            k: Adaptability scaling factor
            kappa: Intrinsic S_imag growth rate
            lambda_ext: External stimulus factor
            S_half: Half-saturation constant for protection function
        """
        self.r = r
        self.delta_base = delta_base
        self.k = k
        self.kappa = kappa
        self.lambda_ext = lambda_ext
        self.S_half = S_half
    
    def sigma(self, S: float) -> float:
        """
        Protection/saturation function.
        
        σ(S) = S / (S + S_half)
        
        σ → 0 as S → 0 (no protection)
        σ → 1 as S → ∞ (full protection)
        """
        return S / (S + self.S_half)
    
    def dI_dt(self, I: float, S: float, delta: float = None) -> float:
        """
        I_site evolution rate.
        
        dI/dt = r · I - δ · I · (1 - k · S_imag)
        """
        if delta is None:
            delta = self.delta_base
        return self.r * I - delta * I * (1 - self.k * S)
    
    def dS_dt(self, I: float, S: float, external_stimulus: float = None) -> float:
        """
        S_imag evolution rate.
        
        dS/dt = κ · |I_C| · (1 - σ(S)) + λ · external_stimulus
        """
        if external_stimulus is None:
            external_stimulus = self.lambda_ext
        
        I_C_mag = np.sqrt(I**2 + S**2)
        return self.kappa * I_C_mag * (1 - self.sigma(S)) + external_stimulus
    
    def net_growth_rate(self, S: float, delta: float = None) -> float:
        """
        Net growth rate for I_site.
        
        Net = r - δ(1 - k·S)
        """
        if delta is None:
            delta = self.delta_base
        return self.r - delta * (1 - self.k * S)
    
    def time_to_threshold(self, I_0: float, S: float,
                         threshold: float = 0.1) -> float:
        """
        Time until I_site falls below threshold (if ever).
        
        Returns:
            Time to threshold, or inf if net rate >= 0
        """
        net = self.net_growth_rate(S)
        if net >= 0:
            return float('inf')
        return np.log(threshold / I_0) / net
    
    def apply_transition(self, I: float, S: float,
                        transition: PhaseTransition) -> Tuple[float, float]:
        """
        Apply phase transition effects.
        
        I' = I · exp(-δ_k · σ(S))
        S' = S · (1 + γ · (1 - σ(S)))
        
        Returns:
            (I_post, S_post)
        """
        sigma_S = self.sigma(S)
        
        # I_site decays, protected by S_imag
        I_post = I * np.exp(-transition.delta * sigma_S)
        
        # S_imag grows (opportunity capture), less if already high
        gamma = 0.1  # Opportunity factor
        S_post = S * (1 + gamma * (1 - sigma_S))
        
        return I_post, S_post
    
    def evolve(self, I_0: float, S_0: float, T: float,
              transitions: Optional[List[PhaseTransition]] = None,
              dt: float = 0.01) -> Dict:
        """
        Simulate full evolution.
        
        Args:
            I_0: Initial I_site
            S_0: Initial S_imag
            T: Total time to simulate
            transitions: List of phase transitions
            dt: Time step
            
        Returns:
            Dict with time series and analysis
        """
        transitions = transitions or []
        transitions = sorted(transitions, key=lambda t: t.time)
        
        times = [0.0]
        I_history = [I_0]
        S_history = [S_0]
        
        I, S = I_0, S_0
        t = 0.0
        trans_idx = 0
        
        while t < T:
            # Check for transitions
            while trans_idx < len(transitions) and transitions[trans_idx].time <= t:
                I, S = self.apply_transition(I, S, transitions[trans_idx])
                trans_idx += 1
            
            # Euler step
            dI = self.dI_dt(I, S) * dt
            dS = self.dS_dt(I, S) * dt
            
            I = max(0, I + dI)  # Prevent negative
            S = max(0, S + dS)
            t += dt
            
            times.append(t)
            I_history.append(I)
            S_history.append(S)
        
        # Compute derived quantities
        times = np.array(times)
        I_history = np.array(I_history)
        S_history = np.array(S_history)
        magnitude = np.sqrt(I_history**2 + S_history**2)
        
        return {
            'times': times,
            'I_site': I_history,
            'S_imag': S_history,
            'magnitude': magnitude,
            'final_state': {
                'I_site': I_history[-1],
                'S_imag': S_history[-1],
                'magnitude': magnitude[-1]
            },
            'transitions_applied': len(transitions),
            'survival': I_history[-1] > 0.01  # Survival threshold
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PART VII: SURVIVAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

class SurvivalAnalyzer:
    """
    Survival analysis for commercial entities.
    
    Implements:
    - Hazard rate: λ(t) = δ · exp(-β · S_imag)
    - Survival probability: S(T) = exp(-∫λdt)
    - Median survival time
    """
    
    def __init__(self, delta: float = 0.07, beta: float = 2.0):
        """
        Initialize survival analyzer.
        
        Args:
            delta: Base obsolescence rate
            beta: Protection factor (higher = S_imag more protective)
        """
        self.delta = delta
        self.beta = beta
    
    def hazard_rate(self, S_imag: float) -> float:
        """
        Instantaneous hazard rate.
        
        λ = δ · exp(-β · S_imag)
        
        High S_imag reduces hazard exponentially.
        """
        return self.delta * np.exp(-self.beta * S_imag)
    
    def survival_probability(self, T: float, S_imag: float) -> float:
        """
        Probability of survival to time T.
        
        S(T) = exp(-δ · T · (1 - S_imag))
        
        (Simplified constant S_imag assumption)
        """
        effective_rate = self.delta * (1 - min(S_imag, 0.99))
        return np.exp(-effective_rate * T)
    
    def median_survival_time(self, S_imag: float) -> float:
        """
        Time at which survival probability = 0.5
        """
        effective_rate = self.delta * (1 - min(S_imag, 0.99))
        if effective_rate <= 0:
            return float('inf')
        return np.log(2) / effective_rate
    
    def survival_curve(self, S_imag: float, 
                      T_max: float = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate survival curve S(t) from 0 to T_max.
        
        Returns:
            (times, survival_probabilities)
        """
        times = np.linspace(0, T_max, 500)
        probs = np.array([self.survival_probability(t, S_imag) for t in times])
        return times, probs
    
    def compare_scenarios(self, S_imag_values: List[float],
                         T_max: float = 50) -> Dict:
        """
        Compare survival curves for different S_imag values.
        """
        results = {}
        for S in S_imag_values:
            times, probs = self.survival_curve(S, T_max)
            results[f'S_imag={S}'] = {
                'times': times,
                'survival': probs,
                'median_time': self.median_survival_time(S),
                'prob_10yr': self.survival_probability(10, S),
                'prob_25yr': self.survival_probability(25, S)
            }
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# PART VIII: EMPIRICAL CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

class EmpiricalCalibrator:
    """
    Calibrate model parameters from empirical data.
    
    Based on Grok's patent obsolescence research:
    - Mean δ ≈ 0.07 annually
    - S_imag proxied by R&D/Revenue ratio
    - Survival correlated with adaptability metrics
    """
    
    @staticmethod
    def estimate_delta_from_patents(citations_t: float, 
                                    citations_t_minus_omega: float,
                                    omega: float = 5.0) -> float:
        """
        Estimate obsolescence rate from patent citation data.
        
        δ = -[ln(1 + Cit_t) - ln(1 + Cit_{t-ω})] / ω
        
        Args:
            citations_t: Current forward citations
            citations_t_minus_omega: Citations ω years ago
            omega: Measurement horizon (years)
            
        Returns:
            Estimated obsolescence rate δ
        """
        numerator = np.log(1 + citations_t) - np.log(1 + citations_t_minus_omega)
        return -numerator / omega
    
    @staticmethod
    def estimate_S_imag(rd_revenue_ratio: float,
                       patent_novelty: float = 0.5,
                       flexibility_index: float = 0.5,
                       pivot_history: float = 0.5) -> float:
        """
        Estimate S_imag from empirical proxies.
        
        S_imag = Σ w_i · Proxy_i (weighted sum, normalized to [0,1])
        
        Args:
            rd_revenue_ratio: R&D spending / Revenue (0 to ~0.3)
            patent_novelty: Novelty score of patents (0 to 1)
            flexibility_index: Organizational flexibility (0 to 1)
            pivot_history: Historical pivot success rate (0 to 1)
            
        Returns:
            Estimated S_imag in [0, 1]
        """
        # Normalize R&D ratio (typical range 0-0.3, cap at 0.3)
        rd_normalized = min(rd_revenue_ratio / 0.3, 1.0)
        
        # Weights (from empirical studies)
        weights = {
            'rd': 0.30,
            'novelty': 0.25,
            'flexibility': 0.25,
            'pivot': 0.20
        }
        
        S_imag = (weights['rd'] * rd_normalized +
                  weights['novelty'] * patent_novelty +
                  weights['flexibility'] * flexibility_index +
                  weights['pivot'] * pivot_history)
        
        return np.clip(S_imag, 0, 1)
    
    @staticmethod
    def predict_survival_rate(S_imag: float, 
                             years: int = 5,
                             delta: float = 0.07) -> float:
        """
        Predict survival probability using calibrated model.
        
        Based on empirical correlation:
        - S_imag > 0.6: ~90% 5-year survival
        - S_imag < 0.4: ~60% 5-year survival
        """
        analyzer = SurvivalAnalyzer(delta=delta)
        return analyzer.survival_probability(years, S_imag)
    
    @staticmethod
    def validate_canonical_law(S_imag_series: np.ndarray,
                               delta_series: np.ndarray,
                               times: np.ndarray) -> Dict:
        """
        Check if entity satisfies the Canonical Tensor Law.
        
        Law: d/dt S_imag > δ(M_t) eventually for survival
        
        Returns:
            Validation results
        """
        # Compute dS/dt numerically
        dS_dt = np.gradient(S_imag_series, times)
        
        # Check condition
        margin = dS_dt - delta_series
        
        satisfies_eventually = False
        first_satisfaction_time = None
        
        for i in range(len(times)):
            if np.all(margin[i:] > 0):
                satisfies_eventually = True
                first_satisfaction_time = times[i]
                break
        
        return {
            'satisfies_canonical_law': satisfies_eventually,
            'first_satisfaction_time': first_satisfaction_time,
            'current_margin': margin[-1],
            'average_margin': np.mean(margin),
            'times_satisfied': np.sum(margin > 0) / len(margin)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PART IX: INTEGRATION WITH PARENT FRAMEWORKS
# ═══════════════════════════════════════════════════════════════════════════════

class ITTBridge:
    """
    Bridge to Intent Tensor Theory parent framework.
    
    Maps CCG concepts to ITT operators:
    - Φ (Intent) ↔ CommercialActor.intent
    - ∇Φ (Gradient) ↔ Polarity direction
    - ∇²Φ (Laplacian) ↔ Transaction collapse
    - θ_cohesion ↔ S_imag / I_site ratio
    """
    
    @staticmethod
    def theta_cohesion(I_C: ComplexIntentTensor) -> float:
        """
        Compute θ_cohesion = S_imag / I_site
        
        This is the Intent Cohesion ratio from f(AutoWorkspace).
        
        θ > 1: Root-dominant (recursively cohesive)
        θ < 1: Fruit-dominant (potential phase conflict)
        """
        if I_C.I_site == 0:
            return float('inf') if I_C.S_imag > 0 else 0
        return I_C.S_imag / I_C.I_site
    
    @staticmethod
    def is_recursively_cohesive(I_C: ComplexIntentTensor) -> bool:
        """
        Check if θ_cohesion > 1 (Recursive Integrity Condition).
        """
        return ITTBridge.theta_cohesion(I_C) > 1
    
    @staticmethod
    def map_to_OVT(I_C: ComplexIntentTensor,
                  coherence: float, potential: float,
                  authority: float) -> float:
        """
        Map to Operational Value Tensor form.
        
        V = A_ijk R^i I^j E^k
        
        Simplified mapping:
        - R¹ ↔ coherence (resource threshold)
        - I¹ ↔ theta_cohesion (intent drift)
        - E¹ ↔ authority (execution lock)
        """
        theta = ITTBridge.theta_cohesion(I_C)
        
        # A_111 component (drift penalty)
        alpha1 = 1.0  # Rigidity factor
        A_111 = 1 / np.exp(alpha1 * (theta - 1)) if theta > 0 else 0
        
        # A_121 component (cohesion amplification)
        beta1 = 1.0  # Leverage factor
        A_121 = beta1 * (np.tanh(theta - 1) + 1)
        
        # MVS form
        V = coherence * authority * (A_111 * theta + A_121 * theta)
        
        return V


# ═══════════════════════════════════════════════════════════════════════════════
# PART X: CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def quick_fitness_check(I_site: float, S_imag: float,
                       coherence: float, potential: float,
                       authority: float = 1.0) -> Dict:
    """
    Quick fitness assessment with defaults.
    
    Args:
        I_site: Current yield
        S_imag: Adaptation capacity
        coherence: Value proposition specificity
        potential: Configuration flexibility
        authority: External validation (default 1.0)
        
    Returns:
        Dict with fitness score and key metrics
    """
    I_C = ComplexIntentTensor(I_site, S_imag)
    calc = AdaptiveFitnessCalculator()
    
    return calc.full_analysis(I_C, coherence, potential, authority, authority)


def simulate_entity(I_0: float, S_0: float, years: float = 10,
                   transitions: Optional[List[Tuple[float, float]]] = None) -> Dict:
    """
    Simulate entity evolution with optional transitions.
    
    Args:
        I_0: Initial I_site
        S_0: Initial S_imag
        years: Simulation duration
        transitions: List of (year, severity) tuples
        
    Returns:
        Evolution results
    """
    trans_list = None
    if transitions:
        trans_list = [PhaseTransition(t, d, f"T{i}") 
                      for i, (t, d) in enumerate(transitions)]
    
    simulator = MarketFieldEvolution()
    return simulator.evolve(I_0, S_0, years, trans_list)


def check_survival(S_imag: float, years: float = 10) -> Dict:
    """
    Quick survival probability check.
    """
    analyzer = SurvivalAnalyzer()
    return {
        'S_imag': S_imag,
        'years': years,
        'survival_probability': analyzer.survival_probability(years, S_imag),
        'median_survival': analyzer.median_survival_time(S_imag),
        'hazard_rate': analyzer.hazard_rate(S_imag)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE INFO
# ═══════════════════════════════════════════════════════════════════════════════

__version__ = "1.0.0"
__author__ = "Auto-Workspace-AI | Sensei Intent Tensor"
__doc__ = """
Commercial Collapse Geometry - Python Implementation

Core Classes:
- IntentVector: Commercial actor intent representation
- TransactionCollapse: Buyer-seller collapse analysis
- ComplexIntentTensor: I_C = I_site + i·S_imag
- ICWHEConstraint: Coherence × Potential ≥ h
- AdaptiveFitnessCalculator: f_adaptive = |I_C|·Ψ·Authority
- MarketFieldEvolution: ODE simulation of commercial dynamics
- SurvivalAnalyzer: Hazard rates and survival probabilities
- EmpiricalCalibrator: Parameter estimation from data
- ITTBridge: Connection to Intent Tensor Theory

HAIL MATH.
"""


if __name__ == "__main__":
    # Demo
    print("Commercial Collapse Geometry - Demo")
    print("=" * 50)
    
    # Create a commercial entity
    I_C = ComplexIntentTensor(I_site=1.0, S_imag=0.6)
    print(f"\nEntity: {I_C}")
    print(f"Phase: {I_C.phase_degrees:.1f}° (0°=Fruit, 90°=Roots)")
    
    # Check fitness
    result = quick_fitness_check(1.0, 0.6, coherence=1.0, potential=1.0)
    print(f"\nAdaptive Fitness: {result['fitness']:.4f}")
    print(f"ICWHE Status: {result['icwhe']['status']}")
    
    # Survival analysis
    survival = check_survival(0.6, years=10)
    print(f"\n10-Year Survival Probability: {survival['survival_probability']:.2%}")
    print(f"Median Survival Time: {survival['median_survival']:.1f} years")
    
    # Simulate evolution
    sim = simulate_entity(1.0, 0.6, years=20, transitions=[(5, 0.3), (15, 0.4)])
    print(f"\nPost-Simulation State:")
    print(f"  I_site: {sim['final_state']['I_site']:.4f}")
    print(f"  S_imag: {sim['final_state']['S_imag']:.4f}")
    print(f"  Survived: {sim['survival']}")
    
    print("\n" + "=" * 50)
    print("HAIL MATH.")
