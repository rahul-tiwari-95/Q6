"""
V0.4 Deep Dive Analysis Script (No Pandas Required)
===================================================
Analyzes phase-by-phase performance and progression patterns.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def filter_episodes(episodes, start_ep, end_ep):
    """Filter episodes within range"""
    return [e for e in episodes if start_ep < e['episode'] <= end_ep]

def calc_stats(episodes_list):
    """Calculate statistics for a list of episodes"""
    if not episodes_list:
        return None
    
    return {
        'count': len(episodes_list),
        'avg_score': np.mean([e['total_reward'] for e in episodes_list]),
        'avg_pellets': np.mean([e['pellets_collected'] for e in episodes_list]),
        'avg_caught': np.mean([e['times_caught'] for e in episodes_list]),
        'avg_steps': np.mean([e['total_steps'] for e in episodes_list]),
        'win_rate': np.mean([e['won'] for e in episodes_list]) * 100
    }

# Paths
v04_dir = Path('training_runs/20251211_032501')
logs_dir = v04_dir / 'logs'
plots_dir = v04_dir / 'plots' / 'deep_analysis'
plots_dir.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("V0.4 DEEP DIVE ANALYSIS")
print("=" * 80)
print()

# Load episodes
print("Loading episodes...")
episodes = []
with open(logs_dir / 'episodes.jsonl', 'r') as f:
    for line in f:
        episodes.append(json.loads(line))  # loads, not load

print(f"Loaded {len(episodes)} episodes")
print()

# === PHASE BREAKDOWN ===
print("╔" + "═"*78 + "╗")
print("║" + " PHASE-BY-PHASE BREAKDOWN ".center(78) + "║")
print("╠" + "═"*78 + "╣")
print()

phase_boundaries = [2500, 5000, 7500, 17500]
phase_names = ['Phase 1 (Easy)', 'Phase 2 (Medium)', 'Phase 3 (Hard)', 'Phase 4 (Expert)']
phase_stats = {}

for i, (phase_name, end) in enumerate(zip(phase_names, phase_boundaries)):
    start = 0 if i == 0 else phase_boundaries[i-1]
    phase_episodes = filter_episodes(episodes, start, end)
    stats = calc_stats(phase_episodes)
    phase_stats[f'phase{i+1}'] = stats
    
    print(f"{phase_name} (Episodes {start+1}-{end}):")
    print(f"  Win Rate:         {stats['win_rate']:.2f}%")
    print(f"  Avg Score:        {stats['avg_score']:.2f}")
    print(f"  Avg Pellets:      {stats['avg_pellets']:.2f}")
    print(f"  Avg Times Caught: {stats['avg_caught']:.2f}")
    print(f"  Avg Steps:        {stats['avg_steps']:.1f}")
    print()

print("╚" + "═"*78 + "╝")
print()

# === PHASE 4 PROGRESSION ===
print("╔" + "═"*78 + "╗")
print("║" + " PHASE 4 PROGRESSION (10,000 Episodes) ".center(78) + "║")
print("╠" + "═"*78 + "╣")
print()

phase4_episodes = filter_episodes(episodes, 7500, 17500)
third_size = len(phase4_episodes) // 3

phase4_early = phase4_episodes[:third_size]
phase4_mid = phase4_episodes[third_size:2*third_size]
phase4_late = phase4_episodes[2*third_size:]

p4_early_stats = calc_stats(phase4_early)
p4_mid_stats = calc_stats(phase4_mid)
p4_late_stats = calc_stats(phase4_late)

print(f"Early Third (Episodes 7501-{7500+third_size}):")
print(f"  Win Rate:         {p4_early_stats['win_rate']:.2f}%")
print(f"  Avg Score:        {p4_early_stats['avg_score']:.2f}")
print(f"  Avg Pellets:      {p4_early_stats['avg_pellets']:.2f}")
print(f"  Avg Times Caught: {p4_early_stats['avg_caught']:.2f}")
print()

print(f"Middle Third (Episodes {7500+third_size+1}-{7500+2*third_size}):")
print(f"  Win Rate:         {p4_mid_stats['win_rate']:.2f}%")
print(f"  Avg Score:        {p4_mid_stats['avg_score']:.2f}")
print(f"  Avg Pellets:      {p4_mid_stats['avg_pellets']:.2f}")
print(f"  Avg Times Caught: {p4_mid_stats['avg_caught']:.2f}")
print()

print(f"Late Third (Episodes {7500+2*third_size+1}-17500):")
print(f"  Win Rate:         {p4_late_stats['win_rate']:.2f}%")
print(f"  Avg Score:        {p4_late_stats['avg_score']:.2f}")
print(f"  Avg Pellets:      {p4_late_stats['avg_pellets']:.2f}")
print(f"  Avg Times Caught: {p4_late_stats['avg_caught']:.2f}")
print()

improvement = p4_late_stats['avg_score'] - p4_early_stats['avg_score']
print(f"Score Improvement (Late - Early): {improvement:+.2f}")
if improvement > 0:
    print("  ✅ PATTERNS EMERGING! Agent still learning in Phase 4!")
else:
    print("  ⚠️  No improvement detected in Phase 4")
print()
print("╚" + "═"*78 + "╝")
print()

# === V0.3 vs V0.4 COMPARISON ===
print("╔" + "═"*78 + "╗")
print("║" + " V0.3 VS V0.4: PHASE 4 COMPARISON ".center(78) + "║")
print("╠" + "═"*78 + "╣")
print()

# v0.3 Phase 4 stats (from previous analysis)
v03_phase4 = {
    'score': -1045.48,
    'pellets': 1.73,
    'caught': 2.40,
    'steps': 250.0
}

v04_phase4 = {
    'score': phase_stats['phase4']['avg_score'],
    'pellets': phase_stats['phase4']['avg_pellets'],
    'caught': phase_stats['phase4']['avg_caught'],
    'steps': phase_stats['phase4']['avg_steps']
}

print("                           v0.3 (5.5K)   v0.4 (10K)    Change")
print("-" * 78)
for key in ['score', 'pellets', 'caught', 'steps']:
    v3_val = v03_phase4[key]
    v4_val = v04_phase4[key]
    pct_change = ((v4_val / v3_val) - 1) * 100 if v3_val != 0 else float('inf')
    print(f"{key.capitalize():20} {v3_val:10.2f}    {v4_val:10.2f}    {pct_change:+6.1f}%")

print()
print("╚" + "═"*78 + "╝")
print()

# === VISUALIZATIONS ===
print("Generating visualizations...")

# Figure 1: Phase Comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

phases = ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4']
win_rates = [phase_stats[f'phase{i+1}']['win_rate'] for i in range(4)]
pellets = [phase_stats[f'phase{i+1}']['avg_pellets'] for i in range(4)]
caught = [phase_stats[f'phase{i+1}']['avg_caught'] for i in range(4)]
steps = [phase_stats[f'phase{i+1}']['avg_steps'] for i in range(4)]

colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']

axes[0, 0].bar(phases, win_rates, color=colors)
axes[0, 0].set_ylabel('Win Rate (%)')
axes[0, 0].set_title('Win Rate by Phase (v0.4)')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].bar(phases, pellets, color=colors)
axes[0, 1].set_ylabel('Avg Pellets')
axes[0, 1].set_title('Average Pellets Collected by Phase')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].bar(phases, caught, color=colors)
axes[1, 0].set_ylabel('Avg Times Caught')
axes[1, 0].set_title('Average Times Caught by Phase')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].bar(phases, steps, color=colors)
axes[1, 1].set_ylabel('Avg Steps')
axes[1, 1].set_title('Average Episode Length by Phase')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(plots_dir / '01_phase_breakdown.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: 01_phase_breakdown.png")
plt.close()

# Figure 2: Win Rate Over Time
fig, ax = plt.subplots(figsize=(15, 6))

episode_nums = [e['episode'] for e in episodes]
wins = [e['won'] for e in episodes]

# Calculate rolling average
window = 100
win_rate_ma = []
for i in range(len(wins)):
    start = max(0, i - window + 1)
    win_rate_ma.append(np.mean(wins[start:i+1]) * 100)

ax.plot(episode_nums, win_rate_ma, linewidth=2, color='green')

# Mark phase boundaries
for boundary, name in zip([2500, 5000, 7500], phase_names[:-1]):
    ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.text(boundary, 90, name, rotation=90, va='top', ha='right', fontsize=9)

ax.set_xlabel('Episode')
ax.set_ylabel('Win Rate (%) - MA(100)')
ax.set_title('Win Rate Progression Across All Phases (v0.4)')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 100])

plt.tight_layout()
plt.savefig(plots_dir / '02_win_rate_progression.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: 02_win_rate_progression.png")
plt.close()

# Figure 3: Phase 4 Progression Details
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

p4_episodes_sorted = sorted(phase4_episodes, key=lambda x: x['episode'])
p4_episode_nums = [e['episode'] - 7500 for e in p4_episodes_sorted]
p4_scores = [e['total_reward'] for e in p4_episodes_sorted]
p4_pellets = [e['pellets_collected'] for e in p4_episodes_sorted]
p4_caught = [e['times_caught'] for e in p4_episodes_sorted]

# Score
score_ma = []
for i in range(len(p4_scores)):
    start = max(0, i - 99)
    score_ma.append(np.mean(p4_scores[start:i+1]))

axes[0, 0].plot(p4_episode_nums, p4_scores, alpha=0.2, color='blue')
axes[0, 0].plot(p4_episode_nums, score_ma, color='darkblue', linewidth=2)
axes[0, 0].set_xlabel('Episode in Phase 4')
axes[0, 0].set_ylabel('Score')
axes[0, 0].set_title('Phase 4: Score Progression (MA100)')
axes[0, 0].grid(True, alpha=0.3)

# Pellets
pellets_ma = []
for i in range(len(p4_pellets)):
    start = max(0, i - 99)
    pellets_ma.append(np.mean(p4_pellets[start:i+1]))

axes[0, 1].plot(p4_episode_nums, p4_pellets, alpha=0.2, color='green')
axes[0, 1].plot(p4_episode_nums, pellets_ma, color='darkgreen', linewidth=2)
axes[0, 1].set_xlabel('Episode in Phase 4')
axes[0, 1].set_ylabel('Pellets Collected')
axes[0, 1].set_title('Phase 4: Pellets Collection (MA100)')
axes[0, 1].grid(True, alpha=0.3)

# Win Rate
p4_wins = [e['won'] for e in p4_episodes_sorted]
win_ma = []
for i in range(len(p4_wins)):
    start = max(0, i - 99)
    win_ma.append(np.mean(p4_wins[start:i+1]) * 100)

axes[1, 0].plot(p4_episode_nums, win_ma, color='purple', linewidth=2)
axes[1, 0].set_xlabel('Episode in Phase 4')
axes[1, 0].set_ylabel('Win Rate (%) - MA(100)')
axes[1, 0].set_title('Phase 4: Win Rate Progression')
axes[1, 0].grid(True, alpha=0.3)

# Times Caught
caught_ma = []
for i in range(len(p4_caught)):
    start = max(0, i - 99)
    caught_ma.append(np.mean(p4_caught[start:i+1]))

axes[1, 1].plot(p4_episode_nums, p4_caught, alpha=0.2, color='red')
axes[1, 1].plot(p4_episode_nums, caught_ma, color='darkred', linewidth=2)
axes[1, 1].set_xlabel('Episode in Phase 4')
axes[1, 1].set_ylabel('Times Caught')
axes[1, 1].set_title('Phase 4: Times Caught (MA100)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(plots_dir / '03_phase4_details.png', dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: 03_phase4_details.png")
plt.close()

print()
print("=" * 80)
print("ANALYSIS COMPLETE!")
print(f"Results saved to: {plots_dir}")
print("=" * 80)
