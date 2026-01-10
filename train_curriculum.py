"""
Curriculum Learning Training Script
====================================
This script implements progressive difficulty training:
- Phase 1 (Episodes 1-500): Only 2 Greedy bots
- Phase 2 (Episodes 501-1000): 2 Greedy + 1 Patroller
- Phase 3 (Episodes 1001-1500): 2 Greedy + 2 Patrollers
- Phase 4 (Episodes 1501-2000): Full game (2 Greedy + 2 Patrollers + 1 Hunter)

This approach allows the agent to learn fundamentals before facing harder challenges.
"""

import os
import sys
import numpy as np
import torch
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any

# Import our custom modules
from environment.hunter_gridworld import HunterGridworld
from agent.dq_agent import DQNAgent
from utils.logger import TrainingLogger
from utils.environment_wrapper import EnvironmentMetricsWrapper


def action_to_name(action: int) -> str:
    """Convert action integer to human-readable name."""
    actions = {0: "UP   ", 1: "DOWN ", 2: "LEFT ", 3: "RIGHT"}
    return actions.get(action, "UNKNOWN")


def create_training_directories() -> Tuple[str, str, str]:
    """Create directories for saving training artifacts."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.join('training_runs', timestamp)
    
    checkpoint_dir = os.path.join(base_dir, 'checkpoints')
    log_dir = os.path.join(base_dir, 'logs')
    plot_dir = os.path.join(base_dir, 'plots')
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("TRAINING DIRECTORIES CREATED")
    print(f"{'='*80}")
    print(f"Checkpoint Dir: {checkpoint_dir}")
    print(f"Log Dir:        {log_dir}")
    print(f"Plot Dir:       {plot_dir}")
    print(f"{'='*80}\n")
    
    return checkpoint_dir, log_dir, plot_dir


def plot_scores(scores: List[float], 
                avg_scores: List[float], 
                save_path: str,
                episode: int,
                phase_boundaries: List[int] = None) -> None:
    """
    Plot training scores with phase boundaries marked.
    
    Args:
        scores: List of episode scores
        avg_scores: List of moving average scores
        save_path: Path to save the plot
        episode: Current episode number
        phase_boundaries: List of episode numbers where difficulty changed
    """
    plt.figure(figsize=(14, 7))
    plt.plot(scores, label='Episode Score', alpha=0.5, linewidth=1)
    plt.plot(avg_scores, label='100-Episode Moving Average', linewidth=2, color='red')
    
    # Mark phase boundaries
    if phase_boundaries:
        for boundary in phase_boundaries:
            if boundary <= episode:
                plt.axvline(x=boundary, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'Curriculum Learning Progress (Episodes 1-{episode})', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def generate_comprehensive_analysis(scores: List[float],
                                   episodes_data: List[Dict],
                                   phase_boundaries: List[int],
                                   save_dir: str) -> None:
    """
    Generate comprehensive post-training analysis with multiple visualizations.
    
    This creates a deep analytical view of the training run to understand:
    - How the agent learned over time
    - Performance differences across phases
    - Whether Phase 4 showed deep pattern emergence
    - Action strategies and exploration patterns
    
    Args:
        scores: List of all episode scores
        episodes_data: List of dicts with episode metrics
        phase_boundaries: Episode numbers where phases changed
        save_dir: Directory to save analysis plots
    """
    # Create analysis subdirectory
    analysis_dir = os.path.join(save_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("GENERATING COMPREHENSIVE POST-TRAINING ANALYSIS")
    print(f"{'='*80}\n")
    
    # Extract metrics from episodes_data
    pellets = [ep.get('pellets_collected', 0) for ep in episodes_data]
    caught = [ep.get('times_caught', 0) for ep in episodes_data]
    steps = [ep.get('steps', 0) for ep in episodes_data]
    epsilons = [ep.get('epsilon', 1.0) for ep in episodes_data]
    
    # ===== FIGURE 1: Training Overview =====
    print("📊 Creating Figure 1: Training Overview with Multiple Metrics...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Subplot 1: Scores with moving averages
    ax1 = axes[0, 0]
    ax1.plot(scores, alpha=0.3, linewidth=0.5, label='Episode Score', color='blue')
    
    # Multiple moving averages
    ma_50 = [np.mean(scores[max(0, i-50):i+1]) for i in range(len(scores))]
    ma_100 = [np.mean(scores[max(0, i-100):i+1]) for i in range(len(scores))]
    ma_200 = [np.mean(scores[max(0, i-200):i+1]) for i in range(len(scores))]
    
    ax1.plot(ma_50, label='MA(50)', linewidth=1.5, color='orange')
    ax1.plot(ma_100, label='MA(100)', linewidth=2, color='red')
    ax1.plot(ma_200, label='MA(200)', linewidth=2.5, color='darkred')
    
    # Phase boundaries
    for i, boundary in enumerate(phase_boundaries):
        ax1.axvline(x=boundary, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
        ax1.text(boundary, ax1.get_ylim()[1]*0.9, f'Phase {i+1}', rotation=90, fontsize=9)
    
    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('Training Scores with Multiple Moving Averages', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Pellets Collected Over Time
    ax2 = axes[0, 1]
    ax2.plot(pellets, alpha=0.3, linewidth=0.5, color='green')
    ma_pellets = [np.mean(pellets[max(0, i-100):i+1]) for i in range(len(pellets))]
    ax2.plot(ma_pellets, linewidth=2, color='darkgreen', label='MA(100)')
    
    for boundary in phase_boundaries:
        ax2.axvline(x=boundary, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
    
    ax2.set_xlabel('Episode', fontsize=11)
    ax2.set_ylabel('Pellets Collected', fontsize=11)
    ax2.set_title('Pellets Collected Over Time', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    # Subplot 3: Times Caught Over Time
    ax3 = axes[1, 0]
    ax3.plot(caught, alpha=0.3, linewidth=0.5, color='red')
    ma_caught = [np.mean(caught[max(0, i-100):i+1]) for i in range(len(caught))]
    ax3.plot(ma_caught, linewidth=2, color='darkred', label='MA(100)')
    
    for boundary in phase_boundaries:
        ax3.axvline(x=boundary, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
    
    ax3.set_xlabel('Episode', fontsize=11)
    ax3.set_ylabel('Times Caught', fontsize=11)
    ax3.set_title('Times Caught Over Time (Lower is Better)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)
    
    # Subplot 4: Episode Length (Survival Time)
    ax4 = axes[1, 1]
    ax4.plot(steps, alpha=0.3, linewidth=0.5, color='purple')
    ma_steps = [np.mean(steps[max(0, i-100):i+1]) for i in range(len(steps))]
    ax4.plot(ma_steps, linewidth=2, color='darkviolet', label='MA(100)')
    
    for boundary in phase_boundaries:
        ax4.axvline(x=boundary, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
    
    ax4.set_xlabel('Episode', fontsize=11)
    ax4.set_ylabel('Steps Taken', fontsize=11)
    ax4.set_title('Episode Length Over Time (Survival)', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, '01_training_overview.png'), dpi=150)
    plt.close()
    print("   ✓ Saved: 01_training_overview.png")
    
    # ===== FIGURE 2: Phase Comparison =====
    print("📊 Creating Figure 2: Phase-by-Phase Comparison...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Calculate phase statistics
    phase_stats = []
    for i in range(len(phase_boundaries)):
        start = phase_boundaries[i-1] if i > 0 else 0
        end = phase_boundaries[i]
        
        phase_scores = scores[start:end]
        phase_pellets = pellets[start:end]
        phase_caught = caught[start:end]
        phase_steps = steps[start:end]
        
        phase_stats.append({
            'phase': i+1,
            'avg_score': np.mean(phase_scores),
            'avg_pellets': np.mean(phase_pellets),
            'avg_caught': np.mean(phase_caught),
            'avg_steps': np.mean(phase_steps),
            'max_score': np.max(phase_scores),
            'min_score': np.min(phase_scores)
        })
    
    # Bar charts for each metric
    phases = [s['phase'] for s in phase_stats]
    
    ax1 = axes[0, 0]
    ax1.bar(phases, [s['avg_score'] for s in phase_stats], color=['lightblue', 'lightgreen', 'lightyellow', 'lightcoral'])
    ax1.set_xlabel('Phase', fontsize=11)
    ax1.set_ylabel('Average Score', fontsize=11)
    ax1.set_title('Average Score by Phase', fontsize=12, fontweight='bold')
    ax1.set_xticks(phases)
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2 = axes[0, 1]
    ax2.bar(phases, [s['avg_pellets'] for s in phase_stats], color=['lightblue', 'lightgreen', 'lightyellow', 'lightcoral'])
    ax2.set_xlabel('Phase', fontsize=11)
    ax2.set_ylabel('Average Pellets', fontsize=11)
    ax2.set_title('Average Pellets Collected by Phase', fontsize=12, fontweight='bold')
    ax2.set_xticks(phases)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(bottom=0)
    
    ax3 = axes[1, 0]
    ax3.bar(phases, [s['avg_caught'] for s in phase_stats], color=['lightblue', 'lightgreen', 'lightyellow', 'lightcoral'])
    ax3.set_xlabel('Phase', fontsize=11)
    ax3.set_ylabel('Average Times Caught', fontsize=11)
    ax3.set_title('Average Times Caught by Phase', fontsize=12, fontweight='bold')
    ax3.set_xticks(phases)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(bottom=0)
    
    ax4 = axes[1, 1]
    ax4.bar(phases, [s['avg_steps'] for s in phase_stats], color=['lightblue', 'lightgreen', 'lightyellow', 'lightcoral'])
    ax4.set_xlabel('Phase', fontsize=11)
    ax4.set_ylabel('Average Episode Length', fontsize=11)
    ax4.set_title('Average Survival Time by Phase', fontsize=12, fontweight='bold')
    ax4.set_xticks(phases)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, '02_phase_comparison.png'), dpi=150)
    plt.close()
    print("   ✓ Saved: 02_phase_comparison.png")
    
    # ===== FIGURE 3: Phase 4 Deep Dive =====
    print("📊 Creating Figure 3: Phase 4 Deep Dive (Extended Training Analysis)...")
    
    phase4_start = phase_boundaries[2]
    phase4_scores = scores[phase4_start:]
    phase4_pellets = pellets[phase4_start:]
    phase4_caught = caught[phase4_start:]
    phase4_steps = steps[phase4_start:]
    
    # Split Phase 4 into thirds for progression analysis
    third = len(phase4_scores) // 3
    early = phase4_scores[:third]
    mid = phase4_scores[third:2*third]
    late = phase4_scores[2*third:]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Phase 4 score progression
    ax1 = axes[0, 0]
    ax1.plot(range(phase4_start, phase4_start + len(phase4_scores)), phase4_scores, 
             alpha=0.3, linewidth=0.5, color='red')
    ma_p4 = [np.mean(phase4_scores[max(0, i-100):i+1]) for i in range(len(phase4_scores))]
    ax1.plot(range(phase4_start, phase4_start + len(ma_p4)), ma_p4, 
             linewidth=2, color='darkred', label='MA(100)')
    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('Phase 4: Score Progression (Full Difficulty)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Phase 4 progression comparison (Early vs Mid vs Late)
    ax2 = axes[0, 1]
    progression_data = [np.mean(early), np.mean(mid), np.mean(late)]
    ax2.bar(['Early\n(First 1/3)', 'Mid\n(Second 1/3)', 'Late\n(Final 1/3)'], 
            progression_data, color=['lightcoral', 'coral', 'darkred'])
    ax2.set_ylabel('Average Score', fontsize=11)
    ax2.set_title('Phase 4: Learning Progression', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Check if still improving
    improvement = progression_data[2] - progression_data[0]
    color = 'green' if improvement > 0 else 'red'
    ax2.text(1, max(progression_data)*0.9, 
             f'Change: {improvement:+.2f}', 
             ha='center', fontsize=10, color=color, fontweight='bold')
    
    # Score distribution histogram
    ax3 = axes[1, 0]
    ax3.hist(phase4_scores, bins=50, color='darkred', alpha=0.7, edgecolor='black')
    ax3.axvline(np.mean(phase4_scores), color='blue', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(phase4_scores):.2f}')
    ax3.axvline(np.median(phase4_scores), color='green', linestyle='--', 
                linewidth=2, label=f'Median: {np.median(phase4_scores):.2f}')
    ax3.set_xlabel('Score', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Phase 4: Score Distribution', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Performance metrics over Phase 4
    ax4 = axes[1, 1]
    p4_x = range(len(phase4_pellets))
    ax4_twin = ax4.twinx()
    
    ma_p4_pellets = [np.mean(phase4_pellets[max(0, i-50):i+1]) for i in range(len(phase4_pellets))]
    ma_p4_caught = [np.mean(phase4_caught[max(0, i-50):i+1]) for i in range(len(phase4_caught))]
    
    line1 = ax4.plot(p4_x, ma_p4_pellets, linewidth=2, color='green', label='Pellets (MA50)')
    line2 = ax4_twin.plot(p4_x, ma_p4_caught, linewidth=2, color='red', label='Caught (MA50)')
    
    ax4.set_xlabel('Episodes into Phase 4', fontsize=11)
    ax4.set_ylabel('Pellets Collected', fontsize=11, color='green')
    ax4_twin.set_ylabel('Times Caught', fontsize=11, color='red')
    ax4.set_title('Phase 4: Pellets vs Caught Over Time', fontsize=12, fontweight='bold')
    ax4.tick_params(axis='y', labelcolor='green')
    ax4_twin.tick_params(axis='y', labelcolor='red')
    ax4.grid(True, alpha=0.3)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, fontsize=9, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, '03_phase4_deep_dive.png'), dpi=150)
    plt.close()
    print("   ✓ Saved: 03_phase4_deep_dive.png")
    
    # ===== FIGURE 4: Exploration Analysis =====
    print("📊 Creating Figure 4: Exploration Strategy Analysis...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Epsilon decay curve
    ax1 = axes[0]
    ax1.plot(epsilons, linewidth=2, color='purple')
    for boundary in phase_boundaries:
        ax1.axvline(x=boundary, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('Epsilon (Exploration Rate)', fontsize=11)
    ax1.set_title('Epsilon Decay Over Training', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # Score vs Epsilon scatter
    ax2 = axes[1]
    scatter = ax2.scatter(epsilons, scores, alpha=0.3, c=range(len(scores)), 
                         cmap='viridis', s=10)
    ax2.set_xlabel('Epsilon', fontsize=11)
    ax2.set_ylabel('Score', fontsize=11)
    ax2.set_title('Score vs Exploration Rate', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Colorbar to show episode progression
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Episode', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, '04_exploration_analysis.png'), dpi=150)
    plt.close()
    print("   ✓ Saved: 04_exploration_analysis.png")
    
    # ===== FIGURE 5: Statistical Summary =====
    print("📊 Creating Figure 5: Statistical Summary Dashboard...")
    fig = plt.figure(figsize=(16, 10))
    
    # Create text summary
    summary_text = f"""
COMPREHENSIVE TRAINING SUMMARY
{'='*80}

OVERALL STATISTICS:
  Total Episodes:           {len(scores)}
  Total Training Time:      [Check logs for duration]
  
SCORE METRICS:
  Final MA(100) Score:      {np.mean(scores[-100:]):.2f}
  Best MA(100) Score:       {max([np.mean(scores[max(0,i-100):i+1]) for i in range(len(scores))]):.2f}
  Overall Average:          {np.mean(scores):.2f}
  Overall Std Dev:          {np.std(scores):.2f}
  Max Single Episode:       {max(scores):.2f}
  Min Single Episode:       {min(scores):.2f}

PERFORMANCE METRICS:
  Avg Pellets/Episode:      {np.mean(pellets):.2f}
  Avg Times Caught/Episode: {np.mean(caught):.2f}
  Avg Episode Length:       {np.mean(steps):.1f} steps
  
PHASE 1 (Episodes 1-{phase_boundaries[0]}):
  Average Score:            {phase_stats[0]['avg_score']:.2f}
  Avg Pellets:              {phase_stats[0]['avg_pellets']:.2f}
  Avg Caught:               {phase_stats[0]['avg_caught']:.2f}

PHASE 2 (Episodes {phase_boundaries[0]+1}-{phase_boundaries[1]}):
  Average Score:            {phase_stats[1]['avg_score']:.2f}
  Avg Pellets:              {phase_stats[1]['avg_pellets']:.2f}
  Avg Caught:               {phase_stats[1]['avg_caught']:.2f}

PHASE 3 (Episodes {phase_boundaries[1]+1}-{phase_boundaries[2]}):
  Average Score:            {phase_stats[2]['avg_score']:.2f}
  Avg Pellets:              {phase_stats[2]['avg_pellets']:.2f}
  Avg Caught:               {phase_stats[2]['avg_caught']:.2f}

PHASE 4 (Episodes {phase_boundaries[2]+1}-{phase_boundaries[3]}) - EXTENDED:
  Average Score:            {phase_stats[3]['avg_score']:.2f}
  Avg Pellets:              {phase_stats[3]['avg_pellets']:.2f}
  Avg Caught:               {phase_stats[3]['avg_caught']:.2f}
  
  Early Third Score:        {np.mean(early):.2f}
  Middle Third Score:       {np.mean(mid):.2f}
  Late Third Score:         {np.mean(late):.2f}
  
  Improvement (Late-Early): {np.mean(late) - np.mean(early):+.2f}
  {'  ✓ PATTERNS EMERGING!' if np.mean(late) > np.mean(early) else '  ⚠ Still struggling with full difficulty'}

EXPLORATION:
  Starting Epsilon:         1.00
  Final Epsilon:            {epsilons[-1]:.4f}
  Episodes to reach 0.1:    {next((i for i, e in enumerate(epsilons) if e <= 0.1), 'Never')}

{'='*80}
"""
    
    plt.text(0.05, 0.95, summary_text, transform=fig.transFigure,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, '05_statistical_summary.png'), dpi=150)
    plt.close()
    print("   ✓ Saved: 05_statistical_summary.png")
    
    print(f"\n{'='*80}")
    print(f"✅ ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"All visualizations saved to: {analysis_dir}")
    print(f"\nGenerated files:")
    print(f"  1. 01_training_overview.png     - Multi-metric training curves")
    print(f"  2. 02_phase_comparison.png      - Performance across phases")
    print(f"  3. 03_phase4_deep_dive.png      - Extended Phase 4 analysis")
    print(f"  4. 04_exploration_analysis.png  - Epsilon decay and correlation")
    print(f"  5. 05_statistical_summary.png   - Complete numerical summary")
    print(f"{'='*80}\n")


def plot_scores(scores: List[float],


def print_hyperparameters(agent: DQNAgent) -> None:
    """Print agent hyperparameters."""
    print(f"{'='*80}")
    print("AGENT HYPERPARAMETERS")
    print(f"{'='*80}")
    print(f"Learning Rate (α):       {agent.learning_rate}")
    print(f"Discount Factor (γ):     {agent.gamma}")
    print(f"Batch Size:              {agent.batch_size}")
    print(f"Replay Buffer Size:      {len(agent.memory)}/{agent.memory.buffer.maxlen}")
    print(f"Target Update (τ):       {agent.tau}")
    print(f"Network Architecture:    {agent.state_size} → {agent.hidden_sizes} → {agent.action_size}")
    print(f"Device:                  {agent.device}")
    print(f"\nEXPLORATION STRATEGY (IMPROVED):")
    print(f"Initial Epsilon:         {agent.epsilon}")
    print(f"Minimum Epsilon:         {agent.epsilon_min} (was 0.01, now 0.05 for better exploration)")
    print(f"Decay Rate:              {agent.epsilon_decay} (was 0.995, now 0.9995 for slower decay)")
    print(f"{'='*80}\n")


def print_environment_info(env) -> None:
    """Print environment configuration."""
    print(f"{'='*80}")
    print(f"ENVIRONMENT CONFIGURATION")
    print(f"{'='*80}")
    print(f"Grid Size:               25x25")
    print(f"State Space Size:        {env.state_size} (flattened)")
    print(f"Actions:                 4 (Up, Down, Left, Right)")
    print(f"\nREWARD STRUCTURE:")
    print(f"  Collect Pellet:        +50")
    print(f"  Get Caught:            -20")
    print(f"  Hit Wall:              -5")
    print(f"  Per Step Penalty:      -0.01 (CRITICAL FIX: was -0.1)")
    print(f"  Win (Score >= 200):    +100")
    print(f"\nWin Condition:           Collect 4 pellets (4 × 50 = 200 points)")
    print(f"Loss Condition:          Lose all 3 lives")
    print(f"{'='*80}\n")


def print_curriculum_phase(phase: int, episode: int, total_episodes: int) -> None:
    """Print curriculum learning phase information."""
    phases = {
        1: {
            'name': 'EASY',
            'enemies': '2 Greedy Bots only',
            'description': 'Learn basic navigation and pellet collection'
        },
        2: {
            'name': 'MEDIUM',
            'enemies': '2 Greedy Bots + 1 Patroller',
            'description': 'Learn to avoid predictable patrol patterns'
        },
        3: {
            'name': 'HARD',
            'enemies': '2 Greedy Bots + 2 Patrollers',
            'description': 'Navigate through multiple patrol routes'
        },
        4: {
            'name': 'EXPERT',
            'enemies': '2 Greedy Bots + 2 Patrollers + 1 Hunter (FULL GAME)',
            'description': 'Face intelligent A* pathfinding enemy'
        }
    }
    
    info = phases.get(phase, phases[1])
    print(f"\n{'='*80}")
    print(f"🎓 CURRICULUM PHASE {phase}: {info['name']}")
    print(f"{'='*80}")
    print(f"Episodes: {episode}/{total_episodes}")
    print(f"Enemies:  {info['enemies']}")
    print(f"Goal:     {info['description']}")
    print(f"{'='*80}\n")


def train_curriculum(n_episodes: int = 10000,
                     max_steps: int = 1000,
                     verbose_every: int = 10,
                     phase_episodes: List[int] = [1500, 1500, 1500, 5500]) -> None:
    """
    Main curriculum training function with extended Phase 4.
    
    Phase 4 is intentionally MUCH LONGER (5500 episodes vs 1500 for others).
    This allows deep pattern learning against full difficulty.
    
    Args:
        n_episodes (int): Total number of episodes (default: 10000)
        max_steps (int): Maximum steps per episode
        verbose_every (int): Print detailed logs every N episodes
        phase_episodes (list): Episodes per phase [phase1, phase2, phase3, phase4]
                               Default: [1500, 1500, 1500, 5500] = 10,000 total
    """
    
    print("\n" + "="*80)
    print("CURRICULUM LEARNING TRAINING")
    print("="*80)
    print("\n📚 TRAINING PHILOSOPHY:")
    print("Instead of throwing agent into full difficulty immediately,")
    print("we progressively increase challenge as agent learns.")
    print("\nThis mimics how humans learn:")
    print("  1. Learn basics in safe environment")
    print("  2. Gradually add complexity")
    print("  3. Master each level before advancing")
    print("  4. Face full challenge only when ready\n")
    
    # Create directories for artifacts
    checkpoint_dir, log_dir, plot_dir = create_training_directories()
    
    # Initialize logger
    logger = TrainingLogger(log_dir)
    
    # Initialize agent (same for all phases)
    agent = DQNAgent(
        state_size=625,
        action_size=4,
        hidden_sizes=(256, 128),
        learning_rate=1e-4,
        gamma=0.99,
        batch_size=64
    )
    
    # Print configuration
    print_hyperparameters(agent)
    
    # Calculate phase boundaries
    phase_boundaries = []
    cumulative = 0
    for episodes_in_phase in phase_episodes:
        cumulative += episodes_in_phase
        phase_boundaries.append(cumulative)
    
    print(f"\n{'='*80}")
    print("CURRICULUM PHASES")
    print(f"{'='*80}")
    print(f"Phase 1 (Easy):    Episodes 1-{phase_boundaries[0]} ({phase_episodes[0]} episodes)")
    print(f"Phase 2 (Medium):  Episodes {phase_boundaries[0]+1}-{phase_boundaries[1]} ({phase_episodes[1]} episodes)")
    print(f"Phase 3 (Hard):    Episodes {phase_boundaries[1]+1}-{phase_boundaries[2]} ({phase_episodes[2]} episodes)")
    print(f"Phase 4 (Expert):  Episodes {phase_boundaries[2]+1}-{phase_boundaries[3]} ({phase_episodes[3]} episodes)")
    print(f"\n⚠️  NOTE: Phase 4 is INTENTIONALLY LONGER ({phase_episodes[3]} episodes)")
    print(f"    Goal: See if deep patterns emerge with extended training on full difficulty")
    print(f"{'='*80}\n")
    
    # Training tracking variables
    scores = []
    scores_window = deque(maxlen=100)
    episodes_data = []  # Store detailed episode data for analysis
    current_phase = 1
    env = None
    
    print(f"\n{'='*80}")
    print("STARTING CURRICULUM TRAINING")
    print(f"{'='*80}\n")
    
    # ===== MAIN TRAINING LOOP =====
    for i_episode in range(1, n_episodes + 1):
        
        # Determine difficulty level based on episode
        if i_episode <= phase_boundaries[0]:
            difficulty = 1
        elif i_episode <= phase_boundaries[1]:
            difficulty = 2
        elif i_episode <= phase_boundaries[2]:
            difficulty = 3
        else:
            difficulty = 4
        
        # Create new environment if phase changed
        if difficulty != current_phase:
            current_phase = difficulty
            print_curriculum_phase(current_phase, i_episode, n_episodes)
            base_env = HunterGridworld(difficulty_level=current_phase)
            env = EnvironmentMetricsWrapper(base_env)
            if current_phase == 1:
                print_environment_info(base_env)
        
        # Initialize environment on first episode
        if env is None:
            print_curriculum_phase(1, i_episode, n_episodes)
            base_env = HunterGridworld(difficulty_level=1)
            env = EnvironmentMetricsWrapper(base_env)
            print_environment_info(base_env)
        
        # Reset environment for new episode
        state, info = env.reset()
        score = 0
        done = False
        
        # Track action distribution for this episode
        actions_taken = {0: 0, 1: 0, 2: 0, 3: 0}
        
        # ===== EPISODE LOOP =====
        for t in range(max_steps):
            # Agent chooses action
            action = agent.act(state, training=True)
            actions_taken[action] += 1
            
            # Take action in environment
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store experience and learn
            agent.step(state, action, reward, next_state, done or truncated)
            
            score += reward
            state = next_state
            
            # Print step-by-step info if verbose
            if (i_episode % verbose_every == 0 and t < 5):
                print(f"  Step {t+1}: Action={action_to_name(action)} "
                      f"Position={info.get('krishna_position', 'N/A')} "
                      f"Reward={reward:6.2f} Score={info.get('score', 0)} "
                      f"Lives={info.get('lives', 3)}")
            
            if done or truncated:
                break
        
        # ===== END OF EPISODE =====
        
        # Get episode metrics
        metrics = env.get_episode_metrics()
        scores_window.append(score)
        scores.append(score)
        
        # Store detailed episode data for post-training analysis
        episodes_data.append({
            'episode': i_episode,
            'phase': current_phase,
            'score': score,
            'pellets_collected': metrics['pellets_collected'],
            'times_caught': metrics['times_caught'],
            'walls_hit': metrics['walls_hit'],
            'steps': metrics['total_steps'],
            'epsilon': agent.epsilon,
            'won': score >= 100
        })
        
        # Decay epsilon
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        
        # Log episode
        logger.log_episode_end(
            episode=i_episode,
            total_reward=score,
            total_steps=metrics['total_steps'],
            pellets_collected=metrics['pellets_collected'],
            times_caught=metrics['times_caught'],
            walls_hit=metrics['walls_hit'],
            won=score >= 100,  # Won if score is positive and high
            epsilon=agent.epsilon,
            avg_reward_per_step=metrics['avg_reward_per_step']
        )
        
        # Print progress indicator
        print(f'\r[Episode {i_episode:4d}/{n_episodes}] [Phase {current_phase}] '
              f'Score: {score:8.2f} | '
              f'Avg(100): {np.mean(scores_window):8.2f} | '
              f'ε: {agent.epsilon:.4f} | '
              f'Collected: {metrics["pellets_collected"]} | '
              f'Caught: {metrics["times_caught"]}', end='')
        
        # Detailed logging every verbose_every episodes
        if i_episode % verbose_every == 0:
            print()  # New line
            logger.print_episode_summary(i_episode, list(scores_window), agent.epsilon)
            
            # Print action distribution
            print(f"\nAction Distribution in Episode {i_episode}:")
            action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
            for action_idx, count in actions_taken.items():
                pct = (count / max(1, metrics['total_steps'])) * 100
                print(f"  {action_names[action_idx]}: {count:4d} times ({pct:5.1f}%)")
        
        # Save checkpoint and plot every 100 episodes
        if i_episode % 100 == 0:
            # Save agent checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{i_episode}.pth')
            agent.save(checkpoint_path)
            
            # Create and save plot
            avg_scores = [np.mean(scores[max(0, i-100):i+1]) for i in range(len(scores))]
            plot_path = os.path.join(plot_dir, f'training_progress_{i_episode}.png')
            plot_scores(scores, avg_scores, plot_path, i_episode, phase_boundaries)
            
            print(f"\n✓ Checkpoint saved: {checkpoint_path}")
            print(f"✓ Plot saved: {plot_path}\n")
    
    # ===== TRAINING COMPLETE =====
    
    # Create final plot
    print("\nGenerating final training curve...")
    avg_scores = [np.mean(scores[max(0, i-100):i+1]) for i in range(len(scores))]
    plot_path = os.path.join(plot_dir, 'final_training_progress.png')
    plot_scores(scores, avg_scores, plot_path, len(scores), phase_boundaries)
    
    # Save final statistics
    logger.save_final_stats()
    
    # ===== COMPREHENSIVE POST-TRAINING ANALYSIS =====
    print(f"\n{'='*80}")
    print("STARTING COMPREHENSIVE POST-TRAINING ANALYSIS")
    print(f"{'='*80}")
    print("This will generate detailed visualizations to understand:")
    print("  - Training progression across all phases")
    print("  - Phase-by-phase performance comparison")
    print("  - Phase 4 deep dive (extended training analysis)")
    print("  - Exploration strategy effectiveness")
    print("  - Statistical summary dashboard")
    print(f"{'='*80}\n")
    
    generate_comprehensive_analysis(scores, episodes_data, phase_boundaries, plot_dir)
    
    print(f"\n{'='*80}")
    print("CURRICULUM TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Total Episodes: {len(scores)}")
    print(f"Final Avg Score: {np.mean(scores_window):.2f}")
    print(f"Best Avg Score: {max(avg_scores):.2f}")
    print(f"Logs: {log_dir}")
    print(f"Analysis Plots: {os.path.join(plot_dir, 'analysis')}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    # Run curriculum training with EXTENDED Phase 4
    # Phase 1-3: 1500 episodes each (learn fundamentals)
    # Phase 4: 5500 episodes (deep mastery of full difficulty)
    # Total: 10,000 episodes
    train_curriculum(
        n_episodes=10000,
        max_steps=1000,
        verbose_every=10,
        phase_episodes=[1500, 1500, 1500, 5500]
    )
