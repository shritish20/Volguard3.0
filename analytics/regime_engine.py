"""
World-Class Regime Engine for VOLGUARD
"""

import numpy as np
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from config.settings import ANALYTICS_CONFIG
from analytics.models import (
    VolMetrics, StructMetrics, EdgeMetrics, ExternalMetrics,
    TimeMetrics, RegimeScore, TradingMandate
)


class RegimeEngine:
    """World-Class Regime Detection Engine"""
    
    def __init__(self):
        self.regime_history = []
        self.transition_matrix = {}
        self.score_history = []
        
    def calculate_scores(self, vol: VolMetrics, struct: StructMetrics, 
                        edge: EdgeMetrics, external: ExternalMetrics,
                        time: TimeMetrics, expiry_type: str) -> RegimeScore:
        """Calculate regime scores with sensitivity analysis"""
        weighted_vrp = edge.weighted_vrp_weekly if expiry_type == "WEEKLY" else edge.weighted_vrp_monthly
        
        # 1. EDGE SCORE
        edge_score = 5.0
        if weighted_vrp > 4.0:
            edge_score += 3.0
        elif weighted_vrp > 2.0:
            edge_score += 2.0
        elif weighted_vrp > 1.0:
            edge_score += 1.0
        elif weighted_vrp < 0:
            edge_score -= 3.0
        
        if edge.term_regime == "BACKWARDATION" and edge.term_spread < -2.0:
            edge_score += 1.0
        elif edge.term_regime == "CONTANGO":
            edge_score += 0.5
        
        edge_score = max(0, min(10, edge_score))
        
        # 2. VOL SCORE
        vol_score = 5.0
        if vol.vov_zscore > ANALYTICS_CONFIG.VOV_CRASH_ZSCORE:
            vol_score = 0.0
        elif vol.vov_zscore > ANALYTICS_CONFIG.VOV_WARNING_ZSCORE:
            vol_score -= 3.0
        elif vol.vov_zscore < 1.5:
            vol_score += 1.5
        
        if vol.ivp_1yr > ANALYTICS_CONFIG.HIGH_VOL_IVP:
            if vol.vix_momentum == "FALLING":
                vol_score += 1.5
            elif vol.vix_momentum == "RISING":
                vol_score -= 1.0
            else:
                vol_score += 0.5
        elif vol.ivp_1yr < ANALYTICS_CONFIG.LOW_VOL_IVP:
            vol_score -= 2.5
        else:
            vol_score += 1.0
        
        vol_score = max(0, min(10, vol_score))
        
        # 3. STRUCT SCORE
        struct_score = 5.0
        if struct.gex_regime == "STICKY":
            if expiry_type == "WEEKLY" and time.dte_weekly <= 1:
                struct_score += 2.5
            else:
                struct_score += 1.0
        elif struct.gex_regime == "SLIPPERY":
            struct_score -= 1.0
        
        if 0.9 < struct.pcr_atm < 1.1:
            struct_score += 1.5
        elif struct.pcr_atm > 1.3 or struct.pcr_atm < 0.7:
            struct_score -= 0.5
        
        if struct.skew_regime == "CRASH_FEAR":
            struct_score -= 1.0
        elif struct.skew_regime == "MELT_UP":
            struct_score -= 0.5
        else:
            struct_score += 0.5
        
        struct_score = max(0, min(10, struct_score))
        
        # 4. RISK SCORE
        risk_score = 10.0
        if external.event_risk == "HIGH":
            risk_score -= 3.0
        if external.fast_vol:
            risk_score -= 2.0
        
        # Flow regime adjustments
        if external.flow_regime == "AGGRESSIVE_BEAR":
            penalty = 4.0 if external.fii_conviction == "VERY_HIGH" else 3.0
            risk_score -= penalty
        elif external.flow_regime == "AGGRESSIVE_BULL":
            bonus = 1.5 if external.fii_conviction == "VERY_HIGH" else 1.0
            risk_score += bonus
        elif external.flow_regime == "CONTRARIAN_TRAP":
            risk_score -= 2.0
        elif external.flow_regime == "GUARDED_BULL":
            risk_score -= 0.5
        
        # Time risk adjustments
        if expiry_type == "WEEKLY" and time.is_gamma_week:
            risk_score -= 2.0
        elif expiry_type == "MONTHLY" and time.is_gamma_month:
            risk_score -= 2.5
        
        risk_score = max(0, min(10, risk_score))
        
        # COMPOSITE SCORE
        composite = (
            vol_score * ANALYTICS_CONFIG.WEIGHT_VOL +
            struct_score * ANALYTICS_CONFIG.WEIGHT_STRUCT +
            edge_score * ANALYTICS_CONFIG.WEIGHT_EDGE +
            risk_score * ANALYTICS_CONFIG.WEIGHT_RISK
        )
        
        # Sensitivity analysis
        alt_weights = [
            (0.30, 0.30, 0.20, 0.20),  # More balanced
            (0.50, 0.20, 0.20, 0.10),  # Vol heavy
            (0.30, 0.40, 0.20, 0.10)   # Structure heavy
        ]
        
        alt_scores = []
        for wv, ws, we, wr in alt_weights:
            alt_scores.append(
                vol_score * wv + struct_score * ws + 
                edge_score * we + risk_score * wr
            )
        
        score_stability = 1.0 - (np.std(alt_scores) / np.mean(alt_scores)) if np.mean(alt_scores) > 0 else 0.5
        
        # Confidence calculation
        if composite >= 8.0 and score_stability > 0.85:
            confidence = "VERY_HIGH"
        elif composite >= 6.5 and score_stability > 0.75:
            confidence = "HIGH"
        elif composite >= 4.0:
            confidence = "MODERATE"
        else:
            confidence = "LOW"
        
        # Transition probabilities
        transition_probability = self.get_transition_probabilities(vol.vol_regime)
        
        score = RegimeScore(
            vol_score=vol_score,
            struct_score=struct_score,
            edge_score=edge_score,
            risk_score=risk_score,
            composite=composite,
            confidence=confidence,
            score_stability=score_stability,
            transition_probability=transition_probability
        )
        
        self.score_history.append(score)
        return score
    
    def calculate_regime_persistence(self, current_regime: str, lookback: int = 5) -> float:
        """Calculate regime persistence over lookback period"""
        if len(self.regime_history) < lookback:
            return 0.5
        
        recent = self.regime_history[-lookback:]
        return sum(1 for r in recent if r == current_regime) / lookback
    
    def update_transition_matrix(self, from_regime: str, to_regime: str):
        """Update transition probability matrix"""
        if from_regime not in self.transition_matrix:
            self.transition_matrix[from_regime] = {}
        
        if to_regime not in self.transition_matrix[from_regime]:
            self.transition_matrix[from_regime][to_regime] = 0
        
        self.transition_matrix[from_regime][to_regime] += 1
    
    def get_transition_probabilities(self, current_regime: str) -> Dict[str, float]:
        """Get transition probabilities from current regime"""
        if current_regime not in self.transition_matrix:
            return {}
        
        transitions = self.transition_matrix[current_regime]
        total = sum(transitions.values())
        
        if total == 0:
            return {}
        
        return {regime: count / total for regime, count in transitions.items()}
    
    def generate_mandate(self, score: RegimeScore, vol: VolMetrics, 
                        struct: StructMetrics, edge: EdgeMetrics,
                        external: ExternalMetrics, time: TimeMetrics,
                        expiry_type: str, expiry_date: date, dte: int) -> TradingMandate:
        """Generate trading mandate based on regime analysis"""
        rationale = []
        warnings = []
        
        weighted_vrp = edge.weighted_vrp_weekly if expiry_type == "WEEKLY" else edge.weighted_vrp_monthly
        
        # Regime classification
        if score.composite >= 7.5:
            regime_name = "AGGRESSIVE_SHORT"
            allocation = 60.0
            strategy = "AGGRESSIVE_SHORT"
            suggested = "STRANGLE"
            rationale.append(f"High Confidence ({score.confidence}): Weighted VRP {weighted_vrp:.2f}% is strong")
            rationale.append(f"Score Stability: {score.score_stability:.2%} (robust across weight variations)")
        elif score.composite >= 6.0:
            regime_name = "MODERATE_SHORT"
            allocation = 40.0
            strategy = "MODERATE_SHORT"
            suggested = "IRON_CONDOR" if dte > 1 else "IRON_FLY"
            rationale.append(f"Moderate Confidence: VRP {weighted_vrp:.2f}% is positive")
        elif score.composite >= 4.0:
            regime_name = "DEFENSIVE"
            allocation = 20.0
            strategy = "DEFENSIVE"
            suggested = "CREDIT_SPREAD"
            rationale.append("Defensive Posture: Focus on defined risk only")
        else:
            regime_name = "CASH"
            allocation = 0.0
            strategy = "CASH"
            suggested = "NONE"
            rationale.append("Regime Unfavorable: Cash is a position")
        
        # Warnings based on risk factors
        if vol.vov_zscore > ANALYTICS_CONFIG.VOV_WARNING_ZSCORE:
            warnings.append(f"⚠️ HIGH VOL-OF-VOL ({vol.vov_zscore:.2f}σ): Market unstable")
        
        if vol.vix_momentum == "RISING" and vol.ivp_1yr > 50:
            warnings.append(f"⚠️ VIX MOMENTUM: Rising {vol.vix_change_5d:+.1f}% in 5d – vol breakout risk")
        
        if struct.skew_regime == "CRASH_FEAR":
            warnings.append(f"⚠️ SKEW: Crash fear (25Δ skew {struct.skew_25d:+.2f}%) – puts expensive")
        
        # Flow regime adjustments
        if external.flow_regime == "AGGRESSIVE_BEAR":
            warnings.append(f"⚠️ FII AGGRESSIVE BEAR ({external.fii_conviction} conviction): Short Fut & Calls")
            if allocation > 0:
                allocation = min(allocation, 20.0)
                rationale.append(f"Allocation capped to 20% due to {external.fii_conviction} conviction bearish flow")
        elif external.flow_regime == "CONTRARIAN_TRAP":
            warnings.append(f"⚠️ FII TRAP SIGNAL: Short Fut but Buying Calls ({external.fii_conviction} conviction)")
            allocation = min(allocation, 30.0)
        
        # Gamma risk adjustments
        if dte <= ANALYTICS_CONFIG.GAMMA_DANGER_DTE and expiry_type == "WEEKLY":
            warnings.append(f"⚠️ GAMMA RISK: {dte} DTE – theta acceleration but pin risk high")
            allocation *= 0.5
        
        # Regime persistence check
        regime_persistence = self.calculate_regime_persistence(regime_name)
        if regime_persistence < ANALYTICS_CONFIG.REGIME_STABILITY_THRESHOLD:
            warnings.append(f"⚠️ REGIME UNSTABLE: Only {regime_persistence:.0%} persistence over 5 days")
            allocation *= 0.7
        else:
            rationale.append(f"Regime Persistence: {regime_persistence:.0%} stability over 5 days")
        
        # Update history and transitions
        self.regime_history.append(regime_name)
        if len(self.regime_history) > 1:
            self.update_transition_matrix(self.regime_history[-2], regime_name)
        
        # Position sizing
        deployable = ANALYTICS_CONFIG.BASE_CAPITAL * (allocation / 100.0)
        if strategy in ["AGGRESSIVE_SHORT", "MODERATE_SHORT"]:
            risk_per_lot = ANALYTICS_CONFIG.MARGIN_SELL_BASE
        else:
            risk_per_lot = ANALYTICS_CONFIG.MARGIN_SELL_BASE * 0.6
        
        max_lots = int(deployable / risk_per_lot) if risk_per_lot > 0 else 0
        
        return TradingMandate(
            expiry_type=expiry_type,
            expiry_date=expiry_date,
            dte=dte,
            regime_name=regime_name,
            strategy_type=strategy,
            allocation_pct=allocation,
            max_lots=max_lots,
            risk_per_lot=risk_per_lot,
            score=score,
            rationale=rationale,
            warnings=warnings,
            suggested_structure=suggested,
            regime_persistence=regime_persistence
        )
