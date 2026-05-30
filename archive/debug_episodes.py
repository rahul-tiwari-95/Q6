"""
Debug Script - Test if training runs properly
Runs just 5 episodes to quickly identify issues
"""

import sys
import traceback

try:
    from environment.hunter_gridworld import HunterGridworld
    from agent.dq_agent import DQNAgent
    from utils.environment_wrapper import EnvironmentMetricsWrapper
    
    print("✓ All imports successful\n")
    
    # Initialize
    print("Initializing environment...")
    base_env = HunterGridworld()
    env = EnvironmentMetricsWrapper(base_env)
    
    print("Initializing agent...")
    agent = DQNAgent(state_size=625, action_size=4)
    
    print("✓ Initialization successful\n")
    
    # Test 5 episodes
    print("=" * 80)
    print("TESTING 5 EPISODES")
    print("=" * 80 + "\n")
    
    for episode in range(1, 6):
        print(f"Episode {episode}:")
        try:
            state, info = env.reset()
            print(f"  Reset successful. Initial score: {info['score']}")
            
            step_count = 0
            done = False
            
            for step in range(100):  # Limit to 100 steps for quick test
                action = agent.act(state, training=True)
                next_state, reward, done, truncated, info = env.step(action)
                agent.step(state, action, reward, next_state, done or truncated)
                state = next_state
                step_count += 1
                
                if done or truncated:
                    break
            
            print(f"  Steps taken: {step_count}")
            print(f"  Done: {done}, Truncated: {truncated}")
            print(f"  Final score: {info['score']}")
            print(f"  Lives remaining: {info['lives']}")
            print(f"  ✓ Episode completed successfully\n")
            
        except Exception as e:
            print(f"  ✗ ERROR in episode {episode}:")
            print(f"  {type(e).__name__}: {e}\n")
            traceback.print_exc()
            break
    
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    
except Exception as e:
    print(f"\n✗ CRITICAL ERROR:")
    print(f"{type(e).__name__}: {e}\n")
    traceback.print_exc()
    sys.exit(1)
