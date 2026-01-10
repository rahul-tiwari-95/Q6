"""
Validation Script
-----------------
Tests that all imports work and the system is ready for training.
Run this BEFORE running main.py to catch issues early.
"""

import sys

def test_imports():
    """Test that all required modules can be imported."""
    
    print("Testing imports...")
    print("=" * 60)
    
    tests = [
        ("NumPy", "import numpy as np"),
        ("PyTorch", "import torch"),
        ("Gymnasium", "import gymnasium as gym"),
        ("Matplotlib", "import matplotlib.pyplot as plt"),
    ]
    
    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"✓ {name:20s} - OK")
        except ImportError as e:
            print(f"✗ {name:20s} - FAILED")
            print(f"  Error: {e}")
            return False
    
    print("\nTesting project modules...")
    print("=" * 60)
    
    project_tests = [
        ("HunterGridworld", "from environment.hunter_gridworld import HunterGridworld"),
        ("DQNAgent", "from agent.dq_agent import DQNAgent"),
        ("QNetwork", "from model.q_network import QNetwork"),
        ("TrainingLogger", "from utils.logger import TrainingLogger"),
        ("EnvironmentWrapper", "from utils.environment_wrapper import EnvironmentMetricsWrapper"),
    ]
    
    for name, import_stmt in project_tests:
        try:
            exec(import_stmt)
            print(f"✓ {name:20s} - OK")
        except ImportError as e:
            print(f"✗ {name:20s} - FAILED")
            print(f"  Error: {e}")
            return False
    
    return True

def test_environment():
    """Test that environment initializes correctly."""
    
    print("\nTesting environment initialization...")
    print("=" * 60)
    
    try:
        from environment.hunter_gridworld import HunterGridworld
        
        env = HunterGridworld()
        state, info = env.reset()
        
        print(f"✓ Environment initialized")
        print(f"  State shape: {state.shape}")
        print(f"  State dtype: {state.dtype}")
        print(f"  Initial score: {info['score']}")
        print(f"  Initial lives: {info['lives']}")
        
        # Try a few steps
        for i in range(3):
            action = env.action_space.sample()
            state, reward, done, truncated, info = env.step(action)
            print(f"  Step {i+1}: reward={reward:.2f}, score={info['score']}")
        
        return True
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agent():
    """Test that agent initializes correctly."""
    
    print("\nTesting agent initialization...")
    print("=" * 60)
    
    try:
        from agent.dq_agent import DQNAgent
        import numpy as np
        
        agent = DQNAgent(state_size=625, action_size=4)
        
        print(f"✓ Agent initialized")
        print(f"  State size: {agent.state_size}")
        print(f"  Action size: {agent.action_size}")
        print(f"  Device: {agent.device}")
        print(f"  Epsilon: {agent.epsilon}")
        
        # Test action selection
        dummy_state = np.zeros(625)
        action = agent.act(dummy_state, training=False)
        print(f"✓ Agent can select actions: {action}")
        
        return True
    except Exception as e:
        print(f"✗ Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validation tests."""
    
    print("\n" + "=" * 60)
    print("Q6 PROJECT VALIDATION")
    print("=" * 60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Environment", test_environment()))
    results.append(("Agent", test_agent()))
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20s} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All tests passed! Ready to run: python main.py\n")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix errors above before running training.\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
