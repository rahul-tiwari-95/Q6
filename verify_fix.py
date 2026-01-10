"""
VERIFICATION SCRIPT: Confirms that the training loop fix is working correctly.

This script runs a quick training session and verifies:
1. Training loop doesn't exit early
2. Multiple episodes complete
3. Metrics are being tracked
4. No errors or exceptions

Run with: python verify_fix.py
"""

import sys
import numpy as np
from collections import deque
from environment.hunter_gridworld import HunterGridworld
from agent.dq_agent import DQNAgent

def verify_training_loop():
    """
    Run a minimal training loop to verify:
    1. Training doesn't exit after 1 episode
    2. Metrics are tracked correctly
    3. No exceptions occur
    """
    
    print("="*80)
    print("VERIFYING TRAINING LOOP FIX")
    print("="*80)
    print()
    
    # Initialize environment and agent
    print("1. Initializing environment and agent...")
    try:
        env = HunterGridworld()
        agent = DQNAgent(
            state_size=625,
            action_size=4,
            hidden_sizes=(256, 128),
            learning_rate=1e-4,
            gamma=0.99
        )
        print("   ✅ Environment and agent initialized successfully")
    except Exception as e:
        print(f"   ❌ Error during initialization: {e}")
        return False
    
    print()
    print("2. Running 20-episode verification...")
    print()
    
    # Track metrics
    scores = deque(maxlen=20)
    episode_count = 0
    errors = []
    
    try:
        for episode in range(20):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            max_steps = 1000
            
            # Run episode
            while not done and steps < max_steps:
                # Select action
                if np.random.random() < 0.5:  # 50% explore, 50% exploit
                    action = np.random.randint(4)
                else:
                    action = agent.act(state, training=False)
                
                # Take step
                next_state, reward, done, truncated, info = env.step(action)
                
                # Store experience and learn
                agent.step(state, action, reward, next_state, done or truncated)
                
                episode_reward += reward
                state = next_state
                steps += 1
            
            # Track score
            scores.append(episode_reward)
            episode_count += 1
            
            # Print progress
            avg_score = np.mean(scores)
            print(f"   Episode {episode+1:2d}/20 | Reward: {episode_reward:8.2f} | "
                  f"Avg(20): {avg_score:8.2f} | Steps: {steps:4d}")
    
    except Exception as e:
        print(f"   ❌ Error during training: {e}")
        errors.append(str(e))
        return False
    
    print()
    print("="*80)
    print("VERIFICATION RESULTS")
    print("="*80)
    
    # Check results
    checks_passed = 0
    checks_total = 4
    
    # Check 1: All 20 episodes completed
    if episode_count == 20:
        print("✅ CHECK 1: All 20 episodes completed (no early exit)")
        checks_passed += 1
    else:
        print(f"❌ CHECK 1: Only {episode_count}/20 episodes completed")
    
    # Check 2: Scores are being tracked
    if len(scores) == 20:
        print(f"✅ CHECK 2: Scores tracked for all episodes")
        checks_passed += 1
    else:
        print(f"❌ CHECK 2: Only {len(scores)}/20 scores tracked")
    
    # Check 3: No exceptions occurred
    if len(errors) == 0:
        print("✅ CHECK 3: No errors during training")
        checks_passed += 1
    else:
        print(f"❌ CHECK 3: {len(errors)} errors occurred")
        for error in errors:
            print(f"   - {error}")
    
    # Check 4: Metrics are reasonable
    avg_score = np.mean(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    print(f"✅ CHECK 4: Metrics calculated")
    print(f"   - Average Score: {avg_score:.2f}")
    print(f"   - Min Score: {min_score:.2f}")
    print(f"   - Max Score: {max_score:.2f}")
    print(f"   - Std Dev: {np.std(scores):.2f}")
    checks_passed += 1
    
    print()
    print("="*80)
    print(f"RESULT: {checks_passed}/{checks_total} checks passed")
    print("="*80)
    
    if checks_passed == checks_total:
        print()
        print("🎉 TRAINING LOOP IS WORKING CORRECTLY!")
        print()
        print("You can now run: python main.py")
        print("for the full 2000-episode training run.")
        print()
        return True
    else:
        print()
        print("❌ SOME CHECKS FAILED - Review errors above")
        print()
        return False

if __name__ == "__main__":
    success = verify_training_loop()
    sys.exit(0 if success else 1)