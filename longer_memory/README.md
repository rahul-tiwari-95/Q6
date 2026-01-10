# LONGER MEMORY - Quick Reference

**Purpose:** This folder contains documentation for future features and reminders for long-term project goals.

---

## **CONTENTS**

### **1. DASHBOARD_REQUIREMENTS.md**
Complete specifications for the Q6 Dashboard (Flask + PostgreSQL).

**When to implement:** After training foundation is validated and working.

**What it includes:**
- Flask web interface for visualizing training runs
- PostgreSQL database for storing historical data
- Comparison tools for different hyperparameters
- Simple, functional design (no fancy animations)
- Estimated 15-20 hours of development

**Technology Stack:**
- Flask (backend)
- PostgreSQL + SQLAlchemy (database)
- Chart.js (visualizations)
- Simple HTML/CSS (frontend)

---

## **FUTURE ADDITIONS**

This folder will grow to include:

### **Transfer Learning Experiments**
- Document attempts to transfer learned policy to new environments
- Compare raw state vs feature-engineered approaches
- Measure transferability metrics

### **Continual Learning Research**
- Notes on adapting agent to changing environments
- Catastrophic forgetting mitigation strategies
- Lifelong learning implementations

### **Architecture Experiments**
- CNN vs MLP for grid world
- Attention mechanisms
- Recurrent networks for temporal dependencies

### **Advanced Training Techniques**
- Prioritized experience replay
- Dueling DQN
- Rainbow DQN improvements

---

## **HOW TO USE THIS FOLDER**

1. **Add new feature ideas here**
   - Create markdown files for each major feature
   - Include requirements, architecture, and timeline
   - Reference these when implementing

2. **Document experiments**
   - Keep notes on what worked and what didn't
   - Track hyperparameter experiments
   - Store insights for future reference

3. **Long-term roadmap**
   - This is your "remember to do this later" folder
   - Prevents good ideas from being forgotten
   - Organized reference for complex features

---

## **NAMING CONVENTION**

Files in this folder should follow:
```
[FEATURE_NAME]_[TYPE].md

Examples:
DASHBOARD_REQUIREMENTS.md
TRANSFER_LEARNING_EXPERIMENTS.md
CONTINUAL_LEARNING_NOTES.md
CNN_ARCHITECTURE_PROPOSAL.md
```

---

## **CURRENT STATUS**

### **Planned (Not Yet Implemented):**
- ⏳ Q6 Dashboard (Flask + PostgreSQL)
- ⏳ Transfer Learning Experiments
- ⏳ Continual Learning Framework
- ⏳ Advanced DQN Variants

### **Priority:**
1. Get basic training working (curriculum learning)
2. Validate agent can learn
3. Then implement dashboard
4. Then explore advanced features

---

**This folder is your project memory - use it well! 🧠**
