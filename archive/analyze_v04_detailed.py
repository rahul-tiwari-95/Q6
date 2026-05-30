"""
V0.4 Deep Dive Analysis Script
===============================
Analyzes phase-by-phase performance and progression patterns.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
v04_dir = Path('training_runs/20251211_032501')
logs_dir = v04_dir / 'logs'
plots_dir = v04_dir / 'plots' / 'deep_analysis'
plots_dir.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("V0.4 DEEP DIVE ANALYSIS")
print("=" * 80)
print()

# Load stats
with open(logs_dir / 'stats.json', 'r') as f:
    stats = json.load(f)

# Load episodes
episodes = []
with open(logs_dir / 'episodes.jsonl', 'r') as f:
    for line in f:
        episodes.append(json.load(line))

print(f"Loaded {len(episodes)} episodes")
print()

# === PHASE BREAKDOWN ===
print("╔" + "═"*78 + "╗")
print("║" + " PHASE-BY-PHASE BREAKDOWN ".center(78) + "║")
print("╠" + "═"*78 + "╣")
print()

# Define phase boundaries
phase_boundaries = [2500, 5000, 7500, 17500]
phase_names = ['Phase 1 (Easy)', 'Phase 2 (Medium)', 'Phase 3 (Hard)', 'Phase 4 (Expert)']

for i, (phase_name, end) in enumerate(zip(phase_names, phase_boundaries)):
    start = 0 if i == 0 else phase_boundaries[i-1]
    phase_df = df[(df['episode'] > start) & (df['episode'] <= end)]
    
    avg_score = phase_df['score'].mean()
    avg_pellets = phase_df['pellets_collected'].mean()
    avg_caught = phase_df['times_caught'].mean()
    avg_steps = phase_df['steps'].mean()
    win_rate = phase_df['won'].sum() / len(phase_df) * 100
    
    print(f"{phase_name} (Episodes {start+1}-{end}):")
    print(f"  Win Rate:         {win_rate:.2f}%")
    print(f"  Avg Score:        {avg_score:.2f}")
    print(f"  Avg Pellets:      {avg_pellets:.2f}")
    print(f"  Avg Times Caught: {avg_caught:.2f}")
    print(f"  Avg Steps:        {avg_steps:.1f}")
    print()

print("╚" + "═"*78 + "╝")
print()

# === PHASE 4 PROGRESSION ===
print("╔" + "═"*78 + "╗")
print("║" + " PHASE 4 PROGRESSION (10,000 Episodes) ".center(78) + "║")
print("╠" + "═"*78 + "╣")
print()

phase4_df = df[df['episode'] > 7500]
third_size = len(phase4_df) // 3

phase4_early = phase4_df.iloc[:third_size]
phase4_mid = phase4_df.iloc[third_size:2*third_size]
phase4_late = phase4_df.iloc[2*third_size:]

for name, subphase_df in [('Early Third', phase4_early), 
                           ('Middle Third', phase4_mid), 
                           ('Late Third', phase4_late)]:
    print(f"{name}:")
    print(f"  Win Rate:         {subphase_df['won'].sum() / len(subphase_df) * 100:.2f}%")
    print(f"  Avg Score:        {subphase_df['score'].mean():.2f}")
    print(f"  Avg Pellets:      {subphase_df['pellets_collected'].mean():.2f}")
    print(f"  Avg Times Caught: {subphase_df['times_caught'].mean():.2f}")
    print()

improvement = phase4_late['score'].mean() - phase4_early['score'].mean()
print(f"Score Improvement (Late - Early): {improvement:+.2f}")
print()
print("╚" + "═"*78 + "╝")
print()

# === V0.3 vs V0.4 COMPARISON ===
print("╔" + "═"*78 + "╗")
print("║" + " V0.3 VS V0.4: PHASE 4 COMPARISON ".center(78) + "║")
print("╠" + "═"*78 + "╣")
print()

# v0.3 Phase 4 stats (from previous analysis)
v03_phase4_stats = {
    'score': -1045.48,
    'pellets': 1.73,
    'caught': 2.40,
    'steps': 250
}

v04_phase4_df = df[df['episode'] > 7500]
v04_phase4_stats = {
    'score': v04_phase4_df['score'].mean(),
    'pellets': v04_phase4_df['pellets_collected'].mean(),
    'caught': v04_phase4_df['times_caught'].mean(),
    'steps': v04_phase4_df['steps'].mean()
}

print("                           v0.3 (5.5K)   v0.4 (10K)    Change")
print("-" * 78)
for key in ['score', 'pellets', 'caught', 'steps']:
    v3_val = v03_phase4_stats[key]
    v4_val = v04_phase4_stats[key]
    pct_change = ((v4_val / v3_val) - 1) * 100 if v3_val != 0 else float('inf')
    print(f"{key.capitalize():20} {v3_val:10.2f}    {v4_val:10.2f}    {pct_change:+6.1f}%")

print()
print("╚" + "═"*78 + "╝")
print()

# === VISUALIZATIONS ===
print("Generating visualizations...")

# Figure 1: Phase Comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

phase_data = {
    'Phase 1': df[(df['episode'] > 0) & (df['episode'] <= 2500)],
    'Phase 2': df[(df['episode'] > 2500) & (df['episode'] <= 5000)],
    'Phase 3': df[(df['episode'] > 5000) & (df['episode'] <= 7500)],
    'Phase 4': df[df['episode'] > 7500]
}

# Win Rate by Phase
win_rates = [phase_df['won'].sum() / len(phase_df) * 100 
             for phase_df in phase_data.values()]
axes[0, 0].bar(phase_data.keys(), win_rates, color=['lightblue', 'lightgreen', 'lightyellow', 'lightcoral'])
axes[0, 0].set_ylabel('Win Rate (%)')
axes[0, 0].set_title('Win Rate by Phase (v0.4)')
axes[0, 0].grid(True, alpha=0.3)

# Pellets by Phase
pellets = [phase_df['pellets_collected'].mean() for phase_df in phase_data.values()]
axes[0, 1].bar(phase_data.keys(), pellets, color=['lightblue', 'lightgreen', 'lightyellow', 'lightcoral'])
axes[0, 1].set_ylabel('Avg Pellets')
axes[0, 1].set_title('Average Pellets Collected by Phase')
axes[0, 1].grid(True, alpha=0.3)

# Times Caught by Phase
caught = [phase_df['times_caught'].mean() for phase_df in phase_data.values()]
axes[1, 0].bar(phase_data.keys(), caught, color=['lightblue', 'lightgreen', 'lightyellow', 'lightcoral'])
axes[1, 0].set_ylabel('Avg Times Caught')
axes[1, 0].set_title('Average Times Caught by Phase')
axes[1, 0].grid(True, alpha=0.3)

# Steps by Phase
steps = [phase_df['steps'].mean() for phase_df in phase_data.values()]
axes[1, 1].bar(phase_data.keys(), steps, color=['lightblue', 'lightgreen', 'lightyellow', 'lightcoral'])
axes[1, 1].set_ylabel('Avg Steps')
axes[1, 1].set_title('Average Episode Length by Phase')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(plots_dir / '01_phase_breakdown.png', dpi=150)
print(f"  ✓ Saved: {plots_dir / '01_phase_breakdown.png'}")
plt.close()

# Figure 2: Phase 4 Progression
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

p4_df = df[df['episode'] > 7500].copy()
p4_df['episode_in_phase'] = p4_df['episode'] - 7500

# Score progression
axes[0, 0].plot(p4_df['episode_in_phase'], p4_df['score'], alpha=0.3, color='blue')
axes[0, 0].plot(p4_df['episode_in_phase'], p4_df['score'].rolling(100).mean(), 
                color='red', linewidth=2, label='MA(100)')
axes[0, 0].set_xlabel('Episode in Phase 4')
axes[0, 0].set_ylabel('Score')
axes[0, 0].set_title('Phase 4: Score Progression')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Pellets progression
axes[0, 1].plot(p4_df['episode_in_phase'], p4_df['pellets_collected'], alpha=0.3, color='green')
axes[0, 1].plot(p4_df['episode_in_phase'], p4_df['pellets_collected'].rolling(100).mean(),
                color='darkgreen', linewidth=2, label='MA(100)')
axes[0, 1].set_xlabel('Episode in Phase 4')
axes[0, 1].set_ylabel('Pellets Collected')
axes[0, 1].set_title('Phase 4: Pellets Collection Progression')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Win rate progression
p4_df['win_ma'] = p4_df['won'].rolling(100).mean() * 100
axes[1, 0].plot(p4_df['episode_in_phase'], p4_df['win_ma'], color='purple', linewidth=2)
axes[1, 0].set_xlabel('Episode in Phase 4')
axes[1, 0].set_ylabel('Win Rate (%) - MA(100)')
axes[1, 0].set_title('Phase 4: Win Rate Progression')
axes[1, 0].grid(True, alpha=0.3)

# Caught progression
axes[1, 1].plot(p4_df['episode_in_phase'], p4_df['times_caught'], alpha=0.3, color='red')
axes[1, 1].plot(p4_df['episode_in_phase'], p4_df['times_caught'].rolling(100).mean(),
                color='darkred', linewidth=2, label='MA(100)')
axes[1, 1].set_xlabel('Episode in Phase 4')
axes[1, 1].set_ylabel('Times Caught')
axes[1, 1].set_title('Phase 4: Times Caught Progression')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(plots_dir / '02_phase4_progression.png', dpi=150)
print(f"  ✓ Saved: {plots_dir / '02_phase4_progression.png'}")
plt.close()

# Figure 3: Win Rate Over Time
fig, ax = plt.subplots(figsize=(15, 6))

df['win_rate_ma100'] = df['won'].rolling(100).mean() * 100
ax.plot(df['episode'], df['win_rate_ma100'], linewidth=2, color='green')

# Mark phase boundaries
for boundary, name in zip(phase_boundaries[:-1], phase_names[:-1]):
    ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.text(boundary, ax.get_ylim()[1]*0.9, name, rotation=90, va='top', ha='right')

ax.set_xlabel('Episode')
ax.set_ylabel('Win Rate (%) - MA(100)')
ax.set_title('Win Rate Progression Across All Phases (v0.4)')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 100])

plt.tight_layout()
plt.savefig(plots_dir / '03_win_rate_progression.png', dpi=150)
print(f"  ✓ Saved: {plots_dir / '03_win_rate_progression.png'}")
plt.close()

print()
print("=" * 80)
print("ANALYSIS COMPLETE!")
print(f"Results saved to: {plots_dir}")
print("=" * 80)
