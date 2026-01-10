"""
Enhanced Terminal Display System
=================================
Version: 0.5
Date: December 12, 2025

Beautiful, informative terminal output for training monitoring.

PHILOSOPHY:
-----------
Training runs can take hours. The terminal should be:
- INFORMATIVE: Show what matters (success scores, tiers, metrics)
- BEAUTIFUL: Use colors, emojis, boxes for visual appeal
- MOTIVATING: Celebrate progress, show improvements
- SCANNABLE: Easy to glance and understand status

This isn't just logging - it's the user experience of watching the AI learn.
"""

import sys
from typing import Dict, List, Any, Optional
from datetime import datetime


class Color:
    """ANSI color codes for terminal styling"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    GRAY = '\033[90m'


class TerminalDisplay:
    """Enhanced terminal display manager with beautiful formatting"""
    
    @staticmethod
    def print_v05_header():
        """Print V0.5 welcome header"""
        print("\n")
        print("╔" + "═" * 78 + "╗")
        print("║" + Color.BOLD + Color.CYAN + " Q6 PROJECT V0.5 - STRATEGIC CURRICULUM LEARNING ".center(88) + Color.END + "║")
        print("║" + " The Progressive Hunter Approach ".center(78) + "║")
        print("╚" + "═" * 78 + "╝")
        print()
        
        print(Color.YELLOW + "🎯 V0.5 KEY IMPROVEMENTS:" + Color.END)
        print("   ✅ Progressive Hunter Difficulty (Random → Greedy → Smart → A*)")
        print("   ✅ Composite Success Scoring (Beyond Binary Win/Loss)")
        print("   ✅ Success Tier Classification (S/A/B/C/D/F Ranks)")
        print("   ✅ Phase-Adaptive Epsilon Reset (Fresh Exploration Per Phase)")
        print("   ✅ Reduced Catch Penalty (-10 vs -20, Softer Learning)")
        print("   ✅ Optimized Training (15K episodes, Strategic Allocation)")
        print()
        print("─" * 80)
        print()
    
    @staticmethod
    def print_phase_header(phase: int, episode_start: int, episode_end: int, hunter_type: str):
        """
        Print phase transition header with visual flair
        
        Args:
            phase: Phase number (1-4)
            episode_start: Starting episode number
            episode_end: Ending episode number
            hunter_type: Name of Hunter for this phase
        """
        phase_colors = {
            1: Color.GREEN,
            2: Color.CYAN,
            3: Color.YELLOW,
            4: Color.RED
        }
        
        phase_emoji = {
            1: "🟢",
            2: "🔵",
            3: "🟡",
            4: "🔴"
        }
        
        phase_desc = {
            1: "EASY - Learn fundamentals with unpredictable Hunter",
            2: "MEDIUM - Face pursuing Hunter, learn evasion",
            3: "HARD - Smart Hunter that doesn't get trapped",
            4: "EXPERT - Optimal A* Hunter, ultimate challenge"
        }
        
        color = phase_colors.get(phase, Color.BLUE)
        emoji = phase_emoji.get(phase, "⚪")
        desc = phase_desc.get(phase, "")
        
        print("\n")
        print("╔" + "═" * 78 + "╗")
        print("║" + color + f"{emoji}     CURRICULUM PHASE {phase}: {hunter_type.upper()}     {emoji}".center(88) + Color.END + "║")
        print("╠" + "═" * 78 + "╣")
        print(f"║  Episodes:    {episode_start}-{episode_end}".ljust(79) + "║")
        print(f"║  Description: {desc}".ljust(79) + "║")
        print(f"║  Hunter Type: {hunter_type}".ljust(79) + "║")
        print("╚" + "═" * 78 + "╝")
        print()
        print("┌" + "─" * 14 + "┬" + "─" * 10 + "┬" + "─" * 20 + "┬" + "─" * 5 + "┬" + "─" * 5 + "┬" + "─" * 6 + "┬" + "─" * 12 + "┐")
        print("│   Episode    │   Tier   │    Success (Reward)  │ 🎯  │ 💀  │  👣  │   Epsilon  │")
        print("│              │          │   Score  (Raw DQN)   │Pills│Dths │Steps │ Explore %  │")
        print("├" + "─" * 14 + "┼" + "─" * 10 + "┼" + "─" * 20 + "┼" + "─" * 5 + "┼" + "─" * 5 + "┼" + "─" * 6 + "┼" + "─" * 12 + "┤")
    
    @staticmethod
    def print_episode_progress(episode: int, total: int, metrics: Dict[str, Any]):
        """
        Print single-line episode progress with tier and metrics
        
        Args:
            episode: Current episode number
            total: Total episodes
            metrics: Episode metrics dict
        """
        score = metrics.get('total_reward', 0)
        pellets = metrics.get('pellets_collected', 0)
        caught = metrics.get('times_caught', 0)
        steps = metrics.get('steps_taken', 0)
        epsilon = metrics.get('epsilon', 0)
        success_score = metrics.get('success_score', 0)
        tier_emoji = metrics.get('success_tier_emoji', '❓')
        tier = metrics.get('success_tier', 'F')
        
        # Color code based on success tier
        if success_score >= 0.85:
            score_color = Color.GREEN
        elif success_score >= 0.70:
            score_color = Color.CYAN
        elif success_score >= 0.50:
            score_color = Color.YELLOW
        else:
            score_color = Color.RED
        
        # Format line with better spacing and alignment
        print(f"│ {episode:6d}/{total:<6d} │ "
              f"{tier_emoji} {tier:^6s} │ "
              f"{score_color}{success_score:>5.3f}{Color.END} ({score:>7.1f}) │ "
              f" {pellets:^3d} │ "
              f" {caught:^3d} │ "
              f" {steps:^4d} │ "
              f" {epsilon:>8.6f} │")
    
    @staticmethod
    def print_phase_footer():
        """Print footer after phase episodes"""
        print("└" + "─" * 14 + "┴" + "─" * 10 + "┴" + "─" * 20 + "┴" + "─" * 5 + "┴" + "─" * 5 + "┴" + "─" * 6 + "┴" + "─" * 12 + "┘")

    
    @staticmethod
    def print_phase_summary(phase_data: List[Dict], phase: int, hunter_type: str):
        """
        Print detailed phase summary with statistics and tier distribution
        
        Args:
            phase_data: List of episode dicts for this phase
            phase: Phase number
            hunter_type: Hunter name
        """
        print("\n")
        print("╔" + "═" * 78 + "╗")
        print("║" + f" PHASE {phase} COMPLETE - {hunter_type.upper()} ".center(78) + "║")
        print("╠" + "═" * 78 + "╣")
        
        # Calculate statistics
        total_eps = len(phase_data)
        avg_score = sum(d.get('total_reward', 0) for d in phase_data) / total_eps
        avg_success = sum(d.get('success_score', 0) for d in phase_data) / total_eps
        avg_pellets = sum(d.get('pellets_collected', 0) for d in phase_data) / total_eps
        avg_caught = sum(d.get('times_caught', 0) for d in phase_data) / total_eps
        avg_steps = sum(d.get('steps_taken', 0) for d in phase_data) / total_eps
        
        # Tier distribution
        tier_counts = {'S': 0, 'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
        for d in phase_data:
            tier = d.get('success_tier', 'F')
            if tier in tier_counts:
                tier_counts[tier] += 1
        
        print("║" + "                                                                              ║")
        print(f"║  📊 PERFORMANCE METRICS:".ljust(79) + "║")
        print(f"║     • Average Success Score:     {avg_success:.3f} / 1.000".ljust(79) + "║")
        print(f"║     • Average Episode Score:     {avg_score:>8.2f}".ljust(79) + "║")
        print(f"║     • Average Pellets/Episode:   {avg_pellets:.2f} / 4".ljust(79) + "║")
        print(f"║     • Average Catches/Episode:   {avg_caught:.2f} / 3".ljust(79) + "║")
        print(f"║     • Average Steps/Episode:     {avg_steps:.1f}".ljust(79) + "║")
        print("║" + "                                                                              ║")
        print("║  🏆 SUCCESS TIER DISTRIBUTION:".ljust(79) + "║")
        
        tier_order = ['S', 'A', 'B', 'C', 'D', 'F']
        tier_labels = {
            'S': '🏆 PERFECT',
            'A': '⭐ EXCELLENT',
            'B': '✅ GOOD',
            'C': '✓  ACCEPTABLE',
            'D': '⚠️  POOR',
            'F': '❌ FAILURE'
        }
        
        for tier in tier_order:
            count = tier_counts.get(tier, 0)
            pct = (count / total_eps) * 100 if total_eps > 0 else 0
            bar_width = int(pct / 2)  # 50 chars max
            bar = "█" * bar_width + "░" * (50 - bar_width)
            print(f"║     {tier_labels[tier]}: {pct:5.1f}%  [{bar}]".ljust(79) + "║")
        
        # Highlight best episodes
        best_episodes = sorted(phase_data, key=lambda x: x.get('success_score', 0), reverse=True)[:3]
        print("║" + "                                                                              ║")
        print("║  ⭐ TOP 3 EPISODES:".ljust(79) + "║")
        for i, ep in enumerate(best_episodes, 1):
            ep_num = ep.get('episode', 0)
            ep_score = ep.get('success_score', 0)
            ep_tier = ep.get('success_tier_emoji', '?')
            print(f"║     {i}. Episode {ep_num:5d}: {ep_tier} {ep_score:.3f}".ljust(79) + "║")
        
        print("║" + "                                                                              ║")
        print("╚" + "═" * 78 + "╝")
        print()
    
    @staticmethod
    def print_epsilon_reset(old_epsilon: float, new_epsilon: float, phase: int):
        """
        Print epsilon reset notification
        
        Args:
            old_epsilon: Previous epsilon value
            new_epsilon: New epsilon value
            phase: Phase number
        """
        print()
        print("┌" + "─" * 78 + "┐")
        print("│" + Color.YELLOW + " 🔄 EPSILON RESET TRIGGERED ".center(78) + Color.END + "│")
        print("├" + "─" * 78 + "┤")
        print(f"│  Phase {phase} Transition:".ljust(79) + "│")
        print(f"│    Old Epsilon: {old_epsilon:.6f}".ljust(79) + "│")
        print(f"│    New Epsilon: {new_epsilon:.6f}".ljust(79) + "│")
        print(f"│    Reason: New Hunter difficulty requires fresh exploration".ljust(79) + "│")
        print("└" + "─" * 78 + "┘")
        print()
    
    @staticmethod
    def print_final_summary(all_episodes_data: List[Dict], training_time: float, run_dir: str):
        """
        Print final training summary with overall statistics
        
        Args:
            all_episodes_data: All episode metrics
            training_time: Total training time in seconds
            run_dir: Directory where results are saved
        """
        print("\n")
        print("╔" + "═" * 78 + "╗")
        print("║" + Color.BOLD + Color.GREEN + " 🎉 V0.5 TRAINING COMPLETE! 🎉 ".center(88) + Color.END + "║")
        print("╠" + "═" * 78 + "╣")
        
        total_eps = len(all_episodes_data)
        
        # Calculate overall metrics
        final_100_success = sum(d.get('success_score', 0) for d in all_episodes_data[-100:]) / min(100, len(all_episodes_data))
        final_100_score = sum(d.get('total_reward', 0) for d in all_episodes_data[-100:]) / min(100, len(all_episodes_data))
        best_success = max(d.get('success_score', 0) for d in all_episodes_data)
        best_score = max(d.get('total_reward', 0) for d in all_episodes_data)
        avg_pellets = sum(d.get('pellets_collected', 0) for d in all_episodes_data) / total_eps
        
        # Tier distribution
        tier_counts = {'S': 0, 'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
        for d in all_episodes_data:
            tier = d.get('success_tier', 'F')
            if tier in tier_counts:
                tier_counts[tier] += 1
        
        print("║" + "                                                                              ║")
        print(f"║  📊 TRAINING STATISTICS:".ljust(79) + "║")
        print(f"║     • Total Episodes:            {total_eps:5d}".ljust(79) + "║")
        print(f"║     • Training Time:             {training_time/60:>6.1f} minutes".ljust(79) + "║")
        print(f"║     • Final 100-Ep Avg Success:  {final_100_success:.3f}".ljust(79) + "║")
        print(f"║     • Final 100-Ep Avg Score:    {final_100_score:>8.2f}".ljust(79) + "║")
        print(f"║     • Best Success Score:        {best_success:.3f}".ljust(79) + "║")
        print(f"║     • Best Episode Score:        {best_score:>8.2f}".ljust(79) + "║")
        print(f"║     • Average Pellets Collected: {avg_pellets:.2f} / 4".ljust(79) + "║")
        print("║" + "                                                                              ║")
        print("║  🏆 OVERALL TIER DISTRIBUTION:".ljust(79) + "║")
        
        for tier in ['S', 'A', 'B', 'C', 'D', 'F']:
            count = tier_counts.get(tier, 0)
            pct = (count / total_eps) * 100 if total_eps > 0 else 0
            tier_labels = {
                'S': '🏆 PERFECT',
                'A': '⭐ EXCELLENT',
                'B': '✅ GOOD',
                'C': '✓  ACCEPTABLE',
                'D': '⚠️  POOR',
                'F': '❌ FAILURE'
            }
            print(f"║     {tier_labels[tier]}: {pct:5.1f}% ({count} episodes)".ljust(79) + "║")
        
        print("║" + "                                                                              ║")
        print("║  📁 RESULTS SAVED TO:".ljust(79) + "║")
        print(f"║     {run_dir}".ljust(79) + "║")
        print("║" + "                                                                              ║")
        print("╚" + "═" * 78 + "╝")
        print()
        print(Color.GREEN + "✅ Training complete! Check the results directory for detailed logs and plots." + Color.END)
        print()
    
    @staticmethod
    def print_comparison_v04_v05(v04_stats: Optional[Dict] = None):
        """
        Print comparison between v0.4 and v0.5 if v0.4 data available
        
        Args:
            v04_stats: Optional v0.4 statistics dict
        """
        if not v04_stats:
            return
        
        print("\n")
        print("╔" + "═" * 78 + "╗")
        print("║" + " 📊 V0.4 vs V0.5 COMPARISON ".center(78) + "║")
        print("╠" + "═" * 78 + "╣")
        print("║" + "  (Coming soon after v0.5 completes)".center(78) + "║")
        print("╚" + "═" * 78 + "╝")
        print()
