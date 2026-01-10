import json
import numpy as np

# Load stats
with open('training_runs/20251213_142452/logs/stats.json', 'r') as f:
    stats = json.load(f)

# Load episodes
episodes = []
with open('training_runs/20251213_142452/logs/episodes.jsonl', 'r') as f:
    for line in f:
        episodes.append(json.loads(line))

print('=' * 80)
print('📊 V0.5 TRAINING RESULTS ANALYSIS')
print('=' * 80)
print()

# Overall stats
print('📈 OVERALL PERFORMANCE:')
print(f'  Total Episodes: {len(episodes):,}')
print(f'  Training Time: {stats.get("training_time", "N/A")}')
print(f'  Episodes Won: {stats["episodes_won"]:,} ({stats["episodes_won"]/len(episodes)*100:.1f}%)')
print(f'  Total Pellets: {stats["pellets_collected_total"]:,} / {len(episodes)*4:,} ({stats["pellets_collected_total"]/(len(episodes)*4)*100:.1f}%)')
print(f'  Total Catches: {stats["times_caught_total"]:,}')
print()

# Phase breakdown
phases = {1: [], 2: [], 3: [], 4: []}
for ep in episodes:
    phase = ep.get('phase', 1)
    phases[phase].append(ep)

print('📊 PHASE-BY-PHASE BREAKDOWN:')
print()
for phase_num in [1, 2, 3, 4]:
    phase_eps = phases[phase_num]
    if not phase_eps:
        continue
    
    hunter_names = {1: 'Random', 2: 'Greedy', 3: 'Smart Greedy', 4: 'A*'}
    
    # Calculate metrics
    success_scores = [ep.get('success_score', 0) for ep in phase_eps]
    pellets = [ep.get('pellets_collected', 0) for ep in phase_eps]
    catches = [ep.get('times_caught', 0) for ep in phase_eps]
    wins = sum(1 for ep in phase_eps if ep.get('won', False))
    
    # Tier distribution
    tiers = {'S': 0, 'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
    for ep in phase_eps:
        tier = ep.get('success_tier', 'F')
        if tier in tiers:
            tiers[tier] += 1
    
    print(f'🔹 PHASE {phase_num}: {hunter_names[phase_num]} Hunter (Episodes {phase_eps[0]["episode"]}-{phase_eps[-1]["episode"]})')
    print(f'   Success Score: {np.mean(success_scores):.3f} ± {np.std(success_scores):.3f}')
    print(f'   Win Rate: {wins/len(phase_eps)*100:.1f}% ({wins}/{len(phase_eps)})')
    print(f'   Avg Pellets: {np.mean(pellets):.2f} / 4')
    print(f'   Avg Catches: {np.mean(catches):.2f} / 3')
    print(f'   Tier Distribution:')
    print(f'     🏆 S: {tiers["S"]/len(phase_eps)*100:5.1f}%  ', end='')
    print(f'⭐ A: {tiers["A"]/len(phase_eps)*100:5.1f}%  ', end='')
    print(f'✅ B: {tiers["B"]/len(phase_eps)*100:5.1f}%')
    print(f'     ✓  C: {tiers["C"]/len(phase_eps)*100:5.1f}%  ', end='')
    print(f'⚠️  D: {tiers["D"]/len(phase_eps)*100:5.1f}%  ', end='')
    print(f'❌ F: {tiers["F"]/len(phase_eps)*100:5.1f}%')
    print()

# Phase 4 detailed analysis (KEY METRIC!)
print('=' * 80)
print('🎯 PHASE 4 DETAILED ANALYSIS (A* HUNTER - THE KEY METRIC)')
print('=' * 80)
phase4 = phases[4]
if phase4:
    # Split into early, mid, late
    n = len(phase4)
    early = phase4[:n//3]
    mid = phase4[n//3:2*n//3]
    late = phase4[2*n//3:]
    
    for name, subset in [('Early (1-2000)', early), ('Mid (2001-4000)', mid), ('Late (4001-6000)', late)]:
        success = np.mean([ep.get('success_score', 0) for ep in subset])
        wins = sum(1 for ep in subset if ep.get('won', False))
        pellets = np.mean([ep.get('pellets_collected', 0) for ep in subset])
        catches = np.mean([ep.get('times_caught', 0) for ep in subset])
        
        print(f'{name}:')
        print(f'  Success: {success:.3f} | Wins: {wins/len(subset)*100:.1f}% | Pellets: {pellets:.2f} | Catches: {catches:.2f}')
    
    print()
    print('📈 Phase 4 Learning Curve:')
    final_100 = phase4[-100:]
    final_success = np.mean([ep.get('success_score', 0) for ep in final_100])
    final_wins = sum(1 for ep in final_100 if ep.get('won', False))
    print(f'  Final 100 Episodes: Success={final_success:.3f}, Win Rate={final_wins}%')

# Compare to V0.4
print()
print('=' * 80)
print('📊 V0.5 vs V0.4 COMPARISON')
print('=' * 80)
print()
print('V0.4 Results (from previous runs):')
print('  Phase 4 Win Rate: 17.98%')
print('  Phase 4 Success: ~0.180')
print()
if phase4:
    phase4_wins = sum(1 for ep in phase4 if ep.get('won', False))
    phase4_success = np.mean([ep.get('success_score', 0) for ep in phase4])
    print('V0.5 Results (Progressive Hunter):')
    print(f'  Phase 4 Win Rate: {phase4_wins/len(phase4)*100:.2f}%')
    print(f'  Phase 4 Success: {phase4_success:.3f}')
    print()
    improvement_wins = (phase4_wins/len(phase4)*100) / 17.98
    improvement_success = phase4_success / 0.180
    print('📈 IMPROVEMENT:')
    print(f'  Win Rate: {improvement_wins:.2f}x  ({phase4_wins/len(phase4)*100:.2f}% vs 17.98%)')
    print(f'  Success Score: {improvement_success:.2f}x  ({phase4_success:.3f} vs 0.180)')

print()
print('=' * 80)
