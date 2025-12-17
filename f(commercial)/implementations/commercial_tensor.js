/**
 * Commercial Collapse Geometry - JavaScript Implementation
 * 
 * f(commercial) — Core Classes for Market Transaction Physics
 * 
 * Auto-Workspace-AI | Sensei Intent Tensor
 * Version 1.0 | December 2025
 * 
 * Compatible with:
 * - Browser (ES6+)
 * - Node.js
 * - Google Apps Script
 * 
 * HAIL MATH.
 */

// ═══════════════════════════════════════════════════════════════════════════════
// PART I: FOUNDATIONAL DATA STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Intent Vector - represents commercial actor intent
 */
class IntentVector {
  /**
   * @param {number} price - Acceptable price point (normalized)
   * @param {number} trust - Required/offered trust level [0, 1]
   * @param {number} timing - Timing flexibility [0, 1]
   * @param {number} relevance - Relevance match [0, 1]
   * @param {Object} custom - Additional domain-specific components
   */
  constructor(price, trust, timing, relevance, custom = null) {
    this.price = price;
    this.trust = trust;
    this.timing = timing;
    this.relevance = relevance;
    this.custom = custom;
  }

  /**
   * Convert to array for calculations
   * @returns {number[]}
   */
  toArray() {
    const base = [this.price, this.trust, this.timing, this.relevance];
    if (this.custom) {
      return base.concat(Object.values(this.custom));
    }
    return base;
  }

  /**
   * Create from array
   * @param {number[]} arr
   * @param {string[]} customKeys
   * @returns {IntentVector}
   */
  static fromArray(arr, customKeys = null) {
    let custom = null;
    if (arr.length > 4 && customKeys) {
      custom = {};
      customKeys.forEach((key, i) => {
        custom[key] = arr[4 + i];
      });
    }
    return new IntentVector(arr[0], arr[1], arr[2], arr[3], custom);
  }
}

/**
 * Polarity enum
 */
const Polarity = {
  SELLER: 1,   // Outward, releasing (+∇Φ)
  BUYER: -1    // Inward, acquiring (-∇Φ)
};

/**
 * Commercial Actor
 */
class CommercialActor {
  /**
   * @param {IntentVector} intent
   * @param {number} polarity - Polarity.SELLER or Polarity.BUYER
   * @param {number} authority - Trust/credibility score
   */
  constructor(intent, polarity, authority = 1.0) {
    this.intent = intent;
    this.polarity = polarity;
    this.authority = authority;
  }
}


// ═══════════════════════════════════════════════════════════════════════════════
// PART II: TRANSACTION COLLAPSE
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Transaction Collapse Evaluator
 * 
 * Implements Theorem 1.2: Transaction Collapse Theorem
 * Transaction Collapse ⟺ ||ΔΨ_transaction|| < ε_market
 */
class TransactionCollapse {
  /**
   * @param {number} epsilonMarket - Market threshold for collapse
   * @param {Object} componentWeights - Weights for gap components
   */
  constructor(epsilonMarket = 0.1, componentWeights = null) {
    this.epsilonMarket = epsilonMarket;
    this.componentWeights = componentWeights || {
      price: 1.0,
      trust: 1.0,
      timing: 1.0,
      relevance: 1.0
    };
  }

  /**
   * Compute intent gap ||ΔΨ|| = ||Φ_buyer - Φ_seller||
   * @param {IntentVector} buyer
   * @param {IntentVector} seller
   * @returns {{totalGap: number, componentGaps: Object}}
   */
  computeIntentGap(buyer, seller) {
    const buyerVec = buyer.toArray();
    const sellerVec = seller.toArray();
    
    const diff = buyerVec.map((b, i) => b - sellerVec[i]);
    
    const componentGaps = {
      price: Math.abs(diff[0]),
      trust: Math.abs(diff[1]),
      timing: Math.abs(diff[2]),
      relevance: Math.abs(diff[3])
    };
    
    // Weighted Euclidean norm
    const weights = [
      this.componentWeights.price || 1.0,
      this.componentWeights.trust || 1.0,
      this.componentWeights.timing || 1.0,
      this.componentWeights.relevance || 1.0
    ];
    
    let sumSquared = 0;
    for (let i = 0; i < Math.min(diff.length, weights.length); i++) {
      sumSquared += (diff[i] * Math.sqrt(weights[i])) ** 2;
    }
    
    return {
      totalGap: Math.sqrt(sumSquared),
      componentGaps: componentGaps
    };
  }

  /**
   * Determine if transaction will collapse
   * @param {IntentVector} buyer
   * @param {IntentVector} seller
   * @returns {boolean}
   */
  willCollapse(buyer, seller) {
    const { totalGap } = this.computeIntentGap(buyer, seller);
    return totalGap < this.epsilonMarket;
  }

  /**
   * Compute soft collapse probability using Gaussian
   * @param {IntentVector} buyer
   * @param {IntentVector} seller
   * @param {number} sigma
   * @returns {number}
   */
  collapseProbability(buyer, seller, sigma = 0.02) {
    const { totalGap } = this.computeIntentGap(buyer, seller);
    return Math.exp(-totalGap ** 2 / (2 * sigma ** 2));
  }

  /**
   * Comprehensive gap analysis
   * @param {IntentVector} buyer
   * @param {IntentVector} seller
   * @returns {Object}
   */
  gapAnalysis(buyer, seller) {
    const { totalGap, componentGaps } = this.computeIntentGap(buyer, seller);
    const willCollapse = totalGap < this.epsilonMarket;
    
    // Identify blocking components
    const threshold = this.epsilonMarket / Math.sqrt(4);
    const blocking = Object.entries(componentGaps)
      .filter(([_, gap]) => gap > threshold)
      .sort((a, b) => b[1] - a[1]);
    
    return {
      totalGap,
      threshold: this.epsilonMarket,
      willCollapse,
      margin: this.epsilonMarket - totalGap,
      componentGaps,
      blockingComponents: blocking,
      collapseProbability: this.collapseProbability(buyer, seller)
    };
  }
}


// ═══════════════════════════════════════════════════════════════════════════════
// PART III: COMPLEX INTENT TENSOR
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Complex Intent Tensor: I_C = I_site + i · S_imag
 * 
 * I_site (Real): Current, measurable, perishable success (The Fruit)
 * S_imag (Imaginary): Structural capacity for future adaptation (The Roots)
 */
class ComplexIntentTensor {
  /**
   * @param {number} ISite - Current yield (real component)
   * @param {number} SImag - Adaptation capacity (imaginary component)
   */
  constructor(ISite, SImag) {
    this.ISite = ISite;
    this.SImag = SImag;
  }

  /**
   * Total commercial energy: |I_C| = √(I_site² + S_imag²)
   * @returns {number}
   */
  get magnitude() {
    return Math.sqrt(this.ISite ** 2 + this.SImag ** 2);
  }

  /**
   * Phase angle in complex plane
   * @returns {number} radians
   */
  get phase() {
    return Math.atan2(this.SImag, this.ISite);
  }

  /**
   * Phase angle in degrees
   * @returns {number}
   */
  get phaseDegrees() {
    return this.phase * (180 / Math.PI);
  }

  /**
   * Market-cycle harvest function
   * Harvest(t) = I_site · cos(ωt) + S_imag · sin(ωt)
   * @param {number} omegaT - Market cycle phase (ω × t)
   * @returns {number}
   */
  harvest(omegaT) {
    return this.ISite * Math.cos(omegaT) + this.SImag * Math.sin(omegaT);
  }

  /**
   * Optimal S_imag/I_site ratio for given transition probability
   * ratio = √(p / (1-p))
   * @param {number} transitionProbability - Probability in (0, 1)
   * @returns {number}
   */
  optimalRatio(transitionProbability) {
    const p = transitionProbability;
    if (p <= 0 || p >= 1) {
      throw new Error("Probability must be in (0, 1)");
    }
    return Math.sqrt(p / (1 - p));
  }

  /**
   * Recommend allocation
   * @param {number} transitionProbability
   * @param {number} totalResources
   * @returns {Object}
   */
  allocationRecommendation(transitionProbability, totalResources) {
    const ratio = this.optimalRatio(transitionProbability);
    const IRecommended = totalResources / (1 + ratio);
    const SRecommended = totalResources - IRecommended;
    
    return {
      ISite: IRecommended,
      SImag: SRecommended,
      ratio: ratio,
      transitionProbability: transitionProbability
    };
  }

  /**
   * Add two tensors
   * @param {ComplexIntentTensor} other
   * @returns {ComplexIntentTensor}
   */
  add(other) {
    return new ComplexIntentTensor(
      this.ISite + other.ISite,
      this.SImag + other.SImag
    );
  }

  /**
   * Scalar multiplication
   * @param {number} scalar
   * @returns {ComplexIntentTensor}
   */
  multiply(scalar) {
    return new ComplexIntentTensor(
      this.ISite * scalar,
      this.SImag * scalar
    );
  }

  toString() {
    return `ComplexIntentTensor(ISite=${this.ISite.toFixed(4)}, SImag=${this.SImag.toFixed(4)}, |IC|=${this.magnitude.toFixed(4)})`;
  }
}


// ═══════════════════════════════════════════════════════════════════════════════
// PART IV: ICWHE CONSTRAINT
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * ICWHE Constraint
 * Δ(Coherence) · Δ(Potential) ≥ h
 */
class ICWHEConstraint {
  /**
   * @param {number} h - Market uncertainty constant
   */
  constructor(h = 1.0) {
    this.h = h;
  }

  /**
   * Compute Π = Δ(Coherence) · Δ(Potential)
   * @param {number} coherence
   * @param {number} potential
   * @returns {number}
   */
  computePi(coherence, potential) {
    return coherence * potential;
  }

  /**
   * Compute validity function Ψ(Π, h) = min(1, Π/h, h/Π)
   * @param {number} coherence
   * @param {number} potential
   * @returns {number}
   */
  computePsi(coherence, potential) {
    const Pi = this.computePi(coherence, potential);
    if (Pi <= 0) return 0;
    return Math.min(1, Pi / this.h, this.h / Pi);
  }

  /**
   * Check if configuration satisfies constraint
   * @param {number} coherence
   * @param {number} potential
   * @returns {boolean}
   */
  satisfiesConstraint(coherence, potential) {
    return this.computePi(coherence, potential) >= this.h;
  }

  /**
   * Analyze deviation from equilibrium
   * @param {number} coherence
   * @param {number} potential
   * @returns {Object}
   */
  deviationFromEquilibrium(coherence, potential) {
    const Pi = this.computePi(coherence, potential);
    const psi = this.computePsi(coherence, potential);
    
    let status, recommendation;
    if (Pi < this.h) {
      status = "BRITTLE";
      recommendation = "Increase flexibility (Potential)";
    } else if (Pi > this.h) {
      status = "VAGUE";
      recommendation = "Increase specificity (Coherence)";
    } else {
      status = "OPTIMAL";
      recommendation = "Maintain current balance";
    }
    
    return {
      Pi,
      h: this.h,
      psi,
      deviation: Pi - this.h,
      relativeDeviation: (Pi - this.h) / this.h,
      status,
      recommendation
    };
  }
}


// ═══════════════════════════════════════════════════════════════════════════════
// PART V: ADAPTIVE FITNESS CALCULATOR
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Adaptive Fitness Calculator
 * f_adaptive = |I_C| · Ψ(Π, h) · Authority_hybrid
 */
class AdaptiveFitnessCalculator {
  /**
   * @param {number} h - ICWHE constant
   * @param {number} alpha - Weight for static vs dynamic authority [0, 1]
   */
  constructor(h = 1.0, alpha = 0.5) {
    this.icwhe = new ICWHEConstraint(h);
    this.alpha = alpha;
  }

  /**
   * Compute hybrid authority
   * @param {number} staticAuth
   * @param {number} dynamicAuth
   * @returns {number}
   */
  computeAuthority(staticAuth, dynamicAuth) {
    return this.alpha * staticAuth + (1 - this.alpha) * dynamicAuth;
  }

  /**
   * Compute adaptive fitness score
   * @param {ComplexIntentTensor} IC
   * @param {number} coherence
   * @param {number} potential
   * @param {number} authorityStatic
   * @param {number} authorityDynamic
   * @returns {number}
   */
  computeFitness(IC, coherence, potential, authorityStatic, authorityDynamic) {
    const magnitude = IC.magnitude;
    const psi = this.icwhe.computePsi(coherence, potential);
    const authority = this.computeAuthority(authorityStatic, authorityDynamic);
    
    return magnitude * psi * authority;
  }

  /**
   * Full analysis
   * @param {ComplexIntentTensor} IC
   * @param {number} coherence
   * @param {number} potential
   * @param {number} authorityStatic
   * @param {number} authorityDynamic
   * @returns {Object}
   */
  fullAnalysis(IC, coherence, potential, authorityStatic, authorityDynamic) {
    const magnitude = IC.magnitude;
    const psi = this.icwhe.computePsi(coherence, potential);
    const authority = this.computeAuthority(authorityStatic, authorityDynamic);
    const fitness = magnitude * psi * authority;
    
    return {
      fitness,
      components: { magnitude, psi, authority },
      IC: {
        ISite: IC.ISite,
        SImag: IC.SImag,
        phaseDegrees: IC.phaseDegrees
      },
      icwhe: this.icwhe.deviationFromEquilibrium(coherence, potential),
      authorityBreakdown: {
        static: authorityStatic,
        dynamic: authorityDynamic,
        alpha: this.alpha
      }
    };
  }
}


// ═══════════════════════════════════════════════════════════════════════════════
// PART VI: MARKET FIELD EVOLUTION
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Phase Transition
 */
class PhaseTransition {
  /**
   * @param {number} time
   * @param {number} delta - Transition severity
   * @param {string} name
   */
  constructor(time, delta, name = "") {
    this.time = time;
    this.delta = delta;
    this.name = name;
  }
}

/**
 * Market Field Evolution Simulator
 */
class MarketFieldEvolution {
  /**
   * @param {Object} params
   */
  constructor(params = {}) {
    this.r = params.r || 0.05;           // Base growth rate
    this.deltaBase = params.deltaBase || 0.07;  // Base obsolescence
    this.k = params.k || 1.0;            // Adaptability scaling
    this.kappa = params.kappa || 0.1;    // Intrinsic S growth
    this.lambdaExt = params.lambdaExt || 0.01;  // External stimulus
    this.SHalf = params.SHalf || 1.0;    // Half-saturation
  }

  /**
   * Protection function σ(S) = S / (S + S_half)
   * @param {number} S
   * @returns {number}
   */
  sigma(S) {
    return S / (S + this.SHalf);
  }

  /**
   * I_site evolution rate
   * @param {number} I
   * @param {number} S
   * @param {number} delta
   * @returns {number}
   */
  dIdt(I, S, delta = null) {
    delta = delta || this.deltaBase;
    return this.r * I - delta * I * (1 - this.k * S);
  }

  /**
   * S_imag evolution rate
   * @param {number} I
   * @param {number} S
   * @returns {number}
   */
  dSdt(I, S) {
    const ICMag = Math.sqrt(I ** 2 + S ** 2);
    return this.kappa * ICMag * (1 - this.sigma(S)) + this.lambdaExt;
  }

  /**
   * Net growth rate
   * @param {number} S
   * @param {number} delta
   * @returns {number}
   */
  netGrowthRate(S, delta = null) {
    delta = delta || this.deltaBase;
    return this.r - delta * (1 - this.k * S);
  }

  /**
   * Apply phase transition
   * @param {number} I
   * @param {number} S
   * @param {PhaseTransition} transition
   * @returns {{I: number, S: number}}
   */
  applyTransition(I, S, transition) {
    const sigmaS = this.sigma(S);
    const IPost = I * Math.exp(-transition.delta * sigmaS);
    const gamma = 0.1;
    const SPost = S * (1 + gamma * (1 - sigmaS));
    return { I: IPost, S: SPost };
  }

  /**
   * Simulate evolution
   * @param {number} I0
   * @param {number} S0
   * @param {number} T
   * @param {PhaseTransition[]} transitions
   * @param {number} dt
   * @returns {Object}
   */
  evolve(I0, S0, T, transitions = [], dt = 0.01) {
    transitions = transitions.sort((a, b) => a.time - b.time);
    
    const times = [0];
    const IHistory = [I0];
    const SHistory = [S0];
    
    let I = I0, S = S0, t = 0;
    let transIdx = 0;
    
    while (t < T) {
      // Check transitions
      while (transIdx < transitions.length && transitions[transIdx].time <= t) {
        const result = this.applyTransition(I, S, transitions[transIdx]);
        I = result.I;
        S = result.S;
        transIdx++;
      }
      
      // Euler step
      const dI = this.dIdt(I, S) * dt;
      const dS = this.dSdt(I, S) * dt;
      
      I = Math.max(0, I + dI);
      S = Math.max(0, S + dS);
      t += dt;
      
      times.push(t);
      IHistory.push(I);
      SHistory.push(S);
    }
    
    const magnitude = IHistory.map((I, i) => Math.sqrt(I ** 2 + SHistory[i] ** 2));
    
    return {
      times,
      ISite: IHistory,
      SImag: SHistory,
      magnitude,
      finalState: {
        ISite: IHistory[IHistory.length - 1],
        SImag: SHistory[SHistory.length - 1],
        magnitude: magnitude[magnitude.length - 1]
      },
      transitionsApplied: transitions.length,
      survival: IHistory[IHistory.length - 1] > 0.01
    };
  }
}


// ═══════════════════════════════════════════════════════════════════════════════
// PART VII: SURVIVAL ANALYSIS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Survival Analyzer
 */
class SurvivalAnalyzer {
  /**
   * @param {number} delta - Base obsolescence rate
   * @param {number} beta - Protection factor
   */
  constructor(delta = 0.07, beta = 2.0) {
    this.delta = delta;
    this.beta = beta;
  }

  /**
   * Hazard rate λ = δ · exp(-β · S_imag)
   * @param {number} SImag
   * @returns {number}
   */
  hazardRate(SImag) {
    return this.delta * Math.exp(-this.beta * SImag);
  }

  /**
   * Survival probability S(T) = exp(-δ · T · (1 - S_imag))
   * @param {number} T
   * @param {number} SImag
   * @returns {number}
   */
  survivalProbability(T, SImag) {
    const effectiveRate = this.delta * (1 - Math.min(SImag, 0.99));
    return Math.exp(-effectiveRate * T);
  }

  /**
   * Median survival time
   * @param {number} SImag
   * @returns {number}
   */
  medianSurvivalTime(SImag) {
    const effectiveRate = this.delta * (1 - Math.min(SImag, 0.99));
    if (effectiveRate <= 0) return Infinity;
    return Math.log(2) / effectiveRate;
  }

  /**
   * Compare scenarios
   * @param {number[]} SImagValues
   * @param {number} TMax
   * @returns {Object}
   */
  compareScenarios(SImagValues, TMax = 50) {
    const results = {};
    for (const S of SImagValues) {
      results[`SImag=${S}`] = {
        medianTime: this.medianSurvivalTime(S),
        prob10yr: this.survivalProbability(10, S),
        prob25yr: this.survivalProbability(25, S)
      };
    }
    return results;
  }
}


// ═══════════════════════════════════════════════════════════════════════════════
// PART VIII: EMPIRICAL CALIBRATION
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Empirical Calibrator
 */
class EmpiricalCalibrator {
  /**
   * Estimate δ from patent citations
   * @param {number} citationsT
   * @param {number} citationsTMinusOmega
   * @param {number} omega
   * @returns {number}
   */
  static estimateDeltaFromPatents(citationsT, citationsTMinusOmega, omega = 5.0) {
    const numerator = Math.log(1 + citationsT) - Math.log(1 + citationsTMinusOmega);
    return -numerator / omega;
  }

  /**
   * Estimate S_imag from proxies
   * @param {number} rdRevenueRatio
   * @param {number} patentNovelty
   * @param {number} flexibilityIndex
   * @param {number} pivotHistory
   * @returns {number}
   */
  static estimateSImag(rdRevenueRatio, patentNovelty = 0.5, 
                       flexibilityIndex = 0.5, pivotHistory = 0.5) {
    const rdNormalized = Math.min(rdRevenueRatio / 0.3, 1.0);
    
    const weights = { rd: 0.30, novelty: 0.25, flexibility: 0.25, pivot: 0.20 };
    
    const SImag = weights.rd * rdNormalized +
                  weights.novelty * patentNovelty +
                  weights.flexibility * flexibilityIndex +
                  weights.pivot * pivotHistory;
    
    return Math.max(0, Math.min(1, SImag));
  }
}


// ═══════════════════════════════════════════════════════════════════════════════
// PART IX: ITT BRIDGE
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Bridge to Intent Tensor Theory
 */
class ITTBridge {
  /**
   * Compute θ_cohesion = S_imag / I_site
   * @param {ComplexIntentTensor} IC
   * @returns {number}
   */
  static thetaCohesion(IC) {
    if (IC.ISite === 0) {
      return IC.SImag > 0 ? Infinity : 0;
    }
    return IC.SImag / IC.ISite;
  }

  /**
   * Check recursive cohesion (θ > 1)
   * @param {ComplexIntentTensor} IC
   * @returns {boolean}
   */
  static isRecursivelyCohesive(IC) {
    return ITTBridge.thetaCohesion(IC) > 1;
  }

  /**
   * Map to OVT form
   * @param {ComplexIntentTensor} IC
   * @param {number} coherence
   * @param {number} potential
   * @param {number} authority
   * @returns {number}
   */
  static mapToOVT(IC, coherence, potential, authority) {
    const theta = ITTBridge.thetaCohesion(IC);
    
    const alpha1 = 1.0;
    const A111 = theta > 0 ? 1 / Math.exp(alpha1 * (theta - 1)) : 0;
    
    const beta1 = 1.0;
    const A121 = beta1 * (Math.tanh(theta - 1) + 1);
    
    return coherence * authority * (A111 * theta + A121 * theta);
  }
}


// ═══════════════════════════════════════════════════════════════════════════════
// PART X: CONVENIENCE FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Quick fitness check
 * @param {number} ISite
 * @param {number} SImag
 * @param {number} coherence
 * @param {number} potential
 * @param {number} authority
 * @returns {Object}
 */
function quickFitnessCheck(ISite, SImag, coherence, potential, authority = 1.0) {
  const IC = new ComplexIntentTensor(ISite, SImag);
  const calc = new AdaptiveFitnessCalculator();
  return calc.fullAnalysis(IC, coherence, potential, authority, authority);
}

/**
 * Simulate entity
 * @param {number} I0
 * @param {number} S0
 * @param {number} years
 * @param {Array} transitions - Array of [year, severity] pairs
 * @returns {Object}
 */
function simulateEntity(I0, S0, years = 10, transitions = []) {
  const transList = transitions.map((t, i) => 
    new PhaseTransition(t[0], t[1], `T${i}`)
  );
  const sim = new MarketFieldEvolution();
  return sim.evolve(I0, S0, years, transList);
}

/**
 * Check survival
 * @param {number} SImag
 * @param {number} years
 * @returns {Object}
 */
function checkSurvival(SImag, years = 10) {
  const analyzer = new SurvivalAnalyzer();
  return {
    SImag,
    years,
    survivalProbability: analyzer.survivalProbability(years, SImag),
    medianSurvival: analyzer.medianSurvivalTime(SImag),
    hazardRate: analyzer.hazardRate(SImag)
  };
}


// ═══════════════════════════════════════════════════════════════════════════════
// EXPORTS (for different environments)
// ═══════════════════════════════════════════════════════════════════════════════

// Node.js / CommonJS
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    // Classes
    IntentVector,
    Polarity,
    CommercialActor,
    TransactionCollapse,
    ComplexIntentTensor,
    ICWHEConstraint,
    AdaptiveFitnessCalculator,
    PhaseTransition,
    MarketFieldEvolution,
    SurvivalAnalyzer,
    EmpiricalCalibrator,
    ITTBridge,
    // Functions
    quickFitnessCheck,
    simulateEntity,
    checkSurvival
  };
}

// ES6 Modules (if using bundler)
// export { ... }

// Browser global
if (typeof window !== 'undefined') {
  window.CCG = {
    IntentVector,
    Polarity,
    CommercialActor,
    TransactionCollapse,
    ComplexIntentTensor,
    ICWHEConstraint,
    AdaptiveFitnessCalculator,
    PhaseTransition,
    MarketFieldEvolution,
    SurvivalAnalyzer,
    EmpiricalCalibrator,
    ITTBridge,
    quickFitnessCheck,
    simulateEntity,
    checkSurvival
  };
}

// Google Apps Script compatibility
// (Functions are available globally)


// ═══════════════════════════════════════════════════════════════════════════════
// DEMO
// ═══════════════════════════════════════════════════════════════════════════════

function demo() {
  console.log("Commercial Collapse Geometry - JavaScript Demo");
  console.log("=".repeat(50));
  
  // Create entity
  const IC = new ComplexIntentTensor(1.0, 0.6);
  console.log(`\nEntity: ${IC.toString()}`);
  console.log(`Phase: ${IC.phaseDegrees.toFixed(1)}° (0°=Fruit, 90°=Roots)`);
  
  // Fitness check
  const result = quickFitnessCheck(1.0, 0.6, 1.0, 1.0);
  console.log(`\nAdaptive Fitness: ${result.fitness.toFixed(4)}`);
  console.log(`ICWHE Status: ${result.icwhe.status}`);
  
  // Survival
  const survival = checkSurvival(0.6, 10);
  console.log(`\n10-Year Survival: ${(survival.survivalProbability * 100).toFixed(1)}%`);
  console.log(`Median Survival: ${survival.medianSurvival.toFixed(1)} years`);
  
  // Simulation
  const sim = simulateEntity(1.0, 0.6, 20, [[5, 0.3], [15, 0.4]]);
  console.log(`\nPost-Simulation:`);
  console.log(`  I_site: ${sim.finalState.ISite.toFixed(4)}`);
  console.log(`  S_imag: ${sim.finalState.SImag.toFixed(4)}`);
  console.log(`  Survived: ${sim.survival}`);
  
  console.log("\n" + "=".repeat(50));
  console.log("HAIL MATH.");
}

// Run demo if main
if (typeof require !== 'undefined' && require.main === module) {
  demo();
}
