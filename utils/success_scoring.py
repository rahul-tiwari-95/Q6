"""
Composite Success Scoring System
=================================
Version: 0.5
Date: December 12, 2025

Multi-dimensional episode evaluation beyond binary win/loss.

PHILOSOPHY:
-----------
Binary win/loss (0 or 1) is too simplistic. It doesn't capture:
- HOW WELL the agent performed
- WHAT STRATEGIES it used
- WHERE it can improve

Real-world missions are graded on multiple dimensions:
- Did you complete the objective? (Primary)
- Did you survive? (Important)
- Were you efficient? (Desirable)
- Did you show intelligence? (Bonus)

This scoring system measures agent performance across these dimensions,
providing richer feedback for analysis and more nuanced success metrics.

COMPONENT WEIGHTS:
------------------
- Objectives (40%): Pellet collection - the PRIMARY goal
- Survival (30%): Damage management - IMPORTANT for mission success
- Efficiency (20%): Time/steps management - DESIRABLE for optimization
- Strategy (10%): Intelligent patterns - BONUS for emergent behavior

SUCCESS TIERS:
--------------
S-Rank (0.95-1.00): 🏆 PERFECT   - Flawless execution
A-Rank (0.85-0.95): ⭐ EXCELLENT - Very strong performance
B-Rank (0.70-0.85): ✅ GOOD      - Solid performance
C-Rank (0.50-0.70): ✓ ACCEPTABLE - Minimum passing
D-Rank (0.30-0.50): ⚠️ POOR      - Significant issues
F-Rank (0.00-0.30): ❌ FAILURE    - Did not accomplish mission
"""

from typing import Dict, Any, Tuple
from enum import Enum


class SuccessTier(Enum):
    """
    Episode performance classification
    
    Maps continuous success scores to discrete tiers for easy understanding.
    Like academic grades: A/B/C/D/F but with more granularity.
    """
    
    PERFECT = ("S", 0.95, 1.00, "🏆", "PERFECT")
    EXCELLENT = ("A", 0.85, 0.95, "⭐", "EXCELLENT")
    GOOD = ("B", 0.70, 0.85, "✅", "GOOD")
    ACCEPTABLE = ("C", 0.50, 0.70, "✓", "ACCEPTABLE")
    POOR = ("D", 0.30, 0.50, "⚠️", "POOR")
    FAILURE = ("F", 0.00, 0.30, "❌", "FAILURE")
    
    def __init__(self, grade, min_score, max_score, emoji, label):
        self.grade = grade
        self.min_score = min_score
        self.max_score = max_score
        self.emoji = emoji
        self.label = label
    
    @classmethod
    def classify(cls, score: float) -> 'SuccessTier':
        """
        Classify score into appropriate tier
        
        Args:
            score: Success score (0.0 to 1.0)
        
        Returns:
            Corresponding SuccessTier enum
        """
        for tier in cls:
            if tier.min_score <= score < tier.max_score:
                return tier
        # Handle edge case: score >= 1.0
        return cls.PERFECT if score >= 0.95 else cls.FAILURE


class EpisodeSuccessScorer:
    """
    Calculate composite success score for episodes
    
    Evaluates agent performance across 4 dimensions:
    1. Objective Completion (40%)
    2. Survival Quality (30%)
    3. Efficiency (20%)
    4. Strategic Behavior (10%)
    
    Usage:
        scorer = EpisodeSuccessScorer()
        total_score, components = scorer.calculate_score(episode_metrics)
        tier = scorer.get_tier(total_score)
    """
    
    def __init__(self, 
                 objective_weight: float = 0.4,
                 survival_weight: float = 0.3,
                 efficiency_weight: float = 0.2,
                 strategy_weight: float = 0.1):
        """
        Initialize scorer with component weights
        
        Args:
            objective_weight: Weight for objective completion (default 40%)
            survival_weight: Weight for survival quality (default 30%)
            efficiency_weight: Weight for efficiency (default 20%)
            strategy_weight: Weight for strategic behavior (default 10%)
        
        Raises:
            ValueError: If weights don't sum to 1.0
        """
        self.weights = {
            'objective': objective_weight,
            'survival': survival_weight,
            'efficiency': efficiency_weight,
            'strategy': strategy_weight
        }
        
        # Validate weights sum to 1.0
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
    
    def calculate_score(self, metrics: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Calculate composite success score
        
        Args:
            metrics: Episode metrics dictionary containing:
                - pellets_collected: int
                - times_caught: int
                - steps_taken: int
                - walls_hit: int (optional)
                - phase: int (optional)
        
        Returns:
            Tuple of (total_score, component_scores_dict)
            - total_score: float (0.0 to 1.0)
            - component_scores: dict with individual component scores
        """
        # Calculate component scores
        objective_score = self._score_objectives(metrics)
        survival_score = self._score_survival(metrics)
        efficiency_score = self._score_efficiency(metrics)
        strategy_score = self._score_strategy(metrics)
        
        # Weighted sum
        total_score = (
            objective_score * self.weights['objective'] +
            survival_score * self.weights['survival'] +
            efficiency_score * self.weights['efficiency'] +
            strategy_score * self.weights['strategy']
        )
        
        component_scores = {
            'objective': objective_score,
            'survival': survival_score,
            'efficiency': efficiency_score,
            'strategy': strategy_score,
            'total': total_score
        }
        
        return total_score, component_scores
    
    def _score_objectives(self, metrics: Dict[str, Any]) -> float:
        """
        Score objective completion (40% of total)
        
        Measures how many pellets were collected out of 4 total.
        Linear scaling with completion bonus.
        
        Args:
            metrics: Must contain 'pellets_collected'
        
        Returns:
            Score from 0.0 (no pellets) to 1.0 (all pellets + bonus)
        
        Examples:
            0 pellets: 0.00
            2 pellets: 0.50
            4 pellets: 1.00 (with 10% bonus)
        """
        pellets_collected = metrics.get('pellets_collected', 0)
        max_pellets = 4
        
        # Linear scaling
        base_score = min(1.0, pellets_collected / max_pellets)
        
        # Completion bonus: +10% for getting ALL pellets
        if pellets_collected >= max_pellets:
            base_score = min(1.0, base_score * 1.1)
        
        return base_score
    
    def _score_survival(self, metrics: Dict[str, Any]) -> float:
        """
        Score survival quality (30% of total)
        
        Measures how well agent avoided getting caught.
        Inverse relationship: fewer catches = higher score.
        
        Args:
            metrics: Must contain 'times_caught'
        
        Returns:
            Score from 0.0 (died) to 1.0 (perfect survival)
        
        Examples:
            0 catches: 1.00 (flawless)
            1 catch:   0.67 (good)
            2 catches: 0.33 (risky)
            3+ catches: 0.00 (dead)
        """
        times_caught = metrics.get('times_caught', 0)
        max_lives = 3
        
        if times_caught == 0:
            return 1.0  # Perfect survival - never caught
        elif times_caught >= max_lives:
            return 0.0  # Death - mission failed
        else:
            # Gradual penalty for each catch
            return 1.0 - (times_caught / max_lives)
    
    def _score_efficiency(self, metrics: Dict[str, Any]) -> float:
        """
        Score efficiency (20% of total)
        
        Measures steps taken per pellet collected.
        Phase-adaptive: harder phases have higher ideal step counts.
        
        Args:
            metrics: Must contain 'steps_taken', 'pellets_collected', 'phase'
        
        Returns:
            Score from 0.0 (very inefficient) to 1.0 (perfect efficiency)
        
        Ideal Steps Per Pellet (by phase):
            Phase 1 (Random Hunter):      50 steps/pellet
            Phase 2 (Greedy Hunter):      75 steps/pellet
            Phase 3 (Smart Greedy):       90 steps/pellet
            Phase 4 (A* Hunter):         100 steps/pellet
        
        Examples:
            Phase 4, 4 pellets in 400 steps: 1.00 (perfect)
            Phase 4, 4 pellets in 800 steps: 0.50 (acceptable)
            Phase 4, 1 pellet in 500 steps:  0.20 (poor)
        """
        steps_taken = metrics.get('steps_taken', 0)
        pellets_collected = metrics.get('pellets_collected', 0)
        
        if pellets_collected == 0:
            return 0.0  # No objectives = no efficiency to measure
        
        # Phase-adaptive ideal steps per pellet
        phase = metrics.get('phase', 4)
        ideal_steps_per_pellet = {
            1: 50,   # Phase 1: Random Hunter (easiest)
            2: 75,   # Phase 2: Greedy Hunter
            3: 90,   # Phase 3: Smart Greedy
            4: 100   # Phase 4: A* Hunter (hardest)
        }.get(phase, 100)
        
        actual_steps_per_pellet = steps_taken / pellets_collected
        
        # Score inversely proportional to steps
        if actual_steps_per_pellet <= ideal_steps_per_pellet:
            return 1.0  # At or better than ideal
        else:
            # Gradually decrease score as steps increase
            efficiency_ratio = ideal_steps_per_pellet / actual_steps_per_pellet
            return max(0.0, efficiency_ratio)
    
    def _score_strategy(self, metrics: Dict[str, Any]) -> float:
        """
        Score strategic behavior (10% of total)
        
        Bonus points for intelligent patterns that suggest planning/awareness:
        - Collecting pellets in challenging phases (risk-taking with purpose)
        - Good risk management (pellets with minimal catches)
        - Clean navigation (no wall collisions)
        - High efficiency (quick pellet collection)
        
        Args:
            metrics: Contains pellets_collected, times_caught, walls_hit, etc.
        
        Returns:
            Score from 0.0 (no strategic behavior) to 1.0 (genius level)
        
        Scoring Breakdown:
            +0.3: Collected pellets in Phase 3+ (faced hard Hunter)
            +0.3: Risk management (3+ pellets with ≤1 catch)
            +0.2: Clean navigation (0 wall hits)
            +0.2: Efficiency (2+ pellets with <100 steps/pellet)
        """
        score = 0.0
        
        pellets = metrics.get('pellets_collected', 0)
        caught = metrics.get('times_caught', 0)
        walls = metrics.get('walls_hit', 0)
        phase = metrics.get('phase', 1)
        steps = metrics.get('steps_taken', 999)
        
        # Bonus 1: Collected pellets in challenging phases (0.3 points)
        # Shows agent isn't just surviving, but accomplishing objectives under pressure
        if phase >= 3 and pellets > 0:
            score += 0.3
        
        # Bonus 2: Excellent risk management (0.3 points)
        # Collected most/all pellets with minimal damage = smart play
        if pellets >= 3 and caught <= 1:
            score += 0.3
        
        # Bonus 3: Clean navigation (0.2 points)
        # No wall hits = good spatial awareness and pathfinding
        if walls == 0:
            score += 0.2
        
        # Bonus 4: High efficiency (0.2 points)
        # Collected multiple pellets quickly = optimal strategy
        if pellets >= 2 and (steps / pellets) < 100:
            score += 0.2
        
        return min(1.0, score)
    
    def get_tier(self, score: float) -> SuccessTier:
        """
        Get success tier classification for score
        
        Args:
            score: Success score (0.0 to 1.0)
        
        Returns:
            SuccessTier enum
        """
        return SuccessTier.classify(score)
    
    def format_score_report(self, metrics: Dict[str, Any]) -> str:
        """
        Generate beautifully formatted score report
        
        Args:
            metrics: Episode metrics
        
        Returns:
            Multi-line formatted string with detailed breakdown
        """
        total_score, components = self.calculate_score(metrics)
        tier = self.get_tier(total_score)
        
        report = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                        EPISODE SUCCESS ANALYSIS                            ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  Overall Grade:  {tier.emoji} {tier.grade}-RANK ({tier.label})                                ║
║  Total Score:    {total_score:.3f} / 1.000                                      ║
║                                                                            ║
║  ┌─ Component Breakdown ─────────────────────────────────────────────┐   ║
║  │                                                                    │   ║
║  │  🎯 Objectives ({self.weights['objective']*100:.0f}%):    {components['objective']:.3f}  [{self._bar(components['objective'])}]  │   ║
║  │  💚 Survival ({self.weights['survival']*100:.0f}%):      {components['survival']:.3f}  [{self._bar(components['survival'])}]  │   ║
║  │  ⚡ Efficiency ({self.weights['efficiency']*100:.0f}%):   {components['efficiency']:.3f}  [{self._bar(components['efficiency'])}]  │   ║
║  │  🧠 Strategy ({self.weights['strategy']*100:.0f}%):     {components['strategy']:.3f}  [{self._bar(components['strategy'])}]  │   ║
║  │                                                                    │   ║
║  └────────────────────────────────────────────────────────────────────┘   ║
║                                                                            ║
║  📊 Metrics:                                                               ║
║    • Pellets Collected:  {metrics.get('pellets_collected', 0)}/4                                      ║
║    • Times Caught:       {metrics.get('times_caught', 0)}/3                                      ║
║    • Steps Taken:        {metrics.get('steps_taken', 0):4d}                                      ║
║    • Walls Hit:          {metrics.get('walls_hit', 0):4d}                                      ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
"""
        return report
    
    @staticmethod
    def _bar(value: float, width: int = 10) -> str:
        """
        Generate ASCII progress bar
        
        Args:
            value: Value between 0.0 and 1.0
            width: Character width of bar
        
        Returns:
            ASCII bar like "████████░░"
        """
        filled = int(value * width)
        return '█' * filled + '░' * (width - filled)
