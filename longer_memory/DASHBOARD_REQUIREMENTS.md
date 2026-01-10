# Q6 DASHBOARD - Requirements & Specifications

**Created:** December 10, 2025
**Status:** Planning Phase (Implement after training foundation works)

---

## **VISION**

A simple, clean Flask-based web dashboard to visualize ALL training runs with:
- Historical data from all training sessions
- Graphs, charts, and metrics
- Clickable navigation between runs
- PostgreSQL backend for persistent storage
- Simple, functional design (NOT fancy animations)

---

## **CORE FEATURES**

### **1. Training Runs List Page**
- Table showing all training runs
- Columns:
  - Run ID
  - Timestamp
  - Total Episodes
  - Final Avg Score
  - Win Rate
  - Duration
  - Status (Complete/In-Progress/Failed)
- Click row → Navigate to detailed view
- Sort by date, score, win rate

### **2. Run Detail Page**
- Full statistics for selected run
- Graphs:
  - Score over time (line chart)
  - 100-episode moving average (line chart)
  - Pellets collected histogram
  - Times caught histogram
  - Action distribution pie chart
  - Epsilon decay curve
- Metrics summary panel:
  - Total episodes
  - Average reward
  - Max/Min reward
  - Win rate
  - Total training time
  - Hyperparameters used

### **3. Comparison Page**
- Select 2+ training runs
- Side-by-side comparison:
  - Overlayed score curves
  - Performance metrics table
  - Hyperparameter differences highlighted

### **4. Live Training Monitor (Optional)**
- Real-time updates during training
- Current episode number
- Live score graph updating
- Recent episode details
- Estimated time remaining

---

## **TECHNOLOGY STACK**

### **Backend:**
- **Flask** - Web framework
- **PostgreSQL** - Database for storing training data
- **SQLAlchemy** - ORM for database operations
- **Psycopg2** - PostgreSQL adapter

### **Frontend:**
- **HTML/CSS** - Simple, clean layout
- **Chart.js** - For graphs and visualizations
- **Bootstrap** - For responsive, clean UI (optional)
- **Vanilla JavaScript** - For interactivity

### **Data Flow:**
```
Training Script (main.py)
    ↓
Logger saves to files (current)
    ↓
Dashboard Importer (new script)
    ↓
PostgreSQL Database
    ↓
Flask API endpoints
    ↓
HTML pages with Chart.js
```

---

## **DATABASE SCHEMA**

### **Table: training_runs**
```sql
CREATE TABLE training_runs (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(50) UNIQUE NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    total_episodes INTEGER,
    final_avg_score FLOAT,
    win_rate FLOAT,
    duration_seconds INTEGER,
    status VARCHAR(20),
    hyperparameters JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### **Table: episodes**
```sql
CREATE TABLE episodes (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(50) REFERENCES training_runs(run_id),
    episode_number INTEGER NOT NULL,
    total_reward FLOAT,
    steps_taken INTEGER,
    pellets_collected INTEGER,
    times_caught INTEGER,
    walls_hit INTEGER,
    epsilon FLOAT,
    won BOOLEAN,
    action_distribution JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### **Table: checkpoints**
```sql
CREATE TABLE checkpoints (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(50) REFERENCES training_runs(run_id),
    episode_number INTEGER NOT NULL,
    checkpoint_path VARCHAR(255),
    model_size_mb FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## **FILE STRUCTURE**

```
Q6/
├── longer_memory/                    (This folder)
│   ├── DASHBOARD_REQUIREMENTS.md    (This file)
│   ├── IMPLEMENTATION_PLAN.md       (Step-by-step guide)
│   └── DATABASE_SETUP.md            (SQL scripts, setup guide)
│
├── dashboard/                        (To be created)
│   ├── app.py                       (Flask application)
│   ├── config.py                    (Database config)
│   ├── models.py                    (SQLAlchemy models)
│   ├── routes.py                    (API endpoints)
│   ├── importer.py                  (Import training data to DB)
│   ├── templates/
│   │   ├── index.html              (Training runs list)
│   │   ├── run_detail.html         (Single run details)
│   │   └── compare.html            (Compare multiple runs)
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css
│   │   └── js/
│   │       └── charts.js
│   └── requirements.txt             (Flask, psycopg2, etc.)
```

---

## **IMPLEMENTATION PHASES**

### **Phase 1: Database Setup**
1. Install PostgreSQL locally
2. Create database `q6_training_data`
3. Run schema creation scripts
4. Test connection from Python

### **Phase 2: Data Importer**
1. Create script to parse existing training logs
2. Import historical data to PostgreSQL
3. Create real-time logging hook in main.py
4. Test data insertion

### **Phase 3: Flask Backend**
1. Create Flask app structure
2. Define SQLAlchemy models
3. Create API endpoints:
   - GET /api/runs (list all runs)
   - GET /api/runs/<id> (single run details)
   - GET /api/runs/<id>/episodes (episode data)
   - GET /api/compare?ids=1,2,3 (comparison data)
4. Test endpoints with curl/Postman

### **Phase 4: Frontend Pages**
1. Create HTML templates with Bootstrap
2. Add Chart.js for visualizations
3. Wire up JavaScript to fetch from API
4. Test in browser

### **Phase 5: Polish & Deploy**
1. Add error handling
2. Add data validation
3. Create simple navigation menu
4. Document usage
5. (Optional) Deploy to local server

---

## **DESIGN PRINCIPLES**

### **Keep It Simple:**
- ✅ Clean, functional design
- ✅ Fast page loads
- ✅ Clear navigation
- ❌ NO fancy animations
- ❌ NO complex frameworks (React, Vue, etc.)
- ❌ NO over-engineering

### **Focus on Data:**
- All important metrics visible at a glance
- Easy comparison between runs
- Historical trends clearly shown
- Actionable insights (which hyperparameters worked?)

### **Maintainable:**
- Simple code structure
- Well-commented
- Easy to extend with new features
- Documented setup process

---

## **EXAMPLE WORKFLOWS**

### **Workflow 1: Review Latest Training**
1. Open browser → http://localhost:5000
2. See list of all training runs, sorted by date
3. Click on latest run
4. View detailed graphs and metrics
5. Download checkpoint if performance is good

### **Workflow 2: Compare Different Hyperparameters**
1. Navigate to "Compare" page
2. Select 3 training runs:
   - Run A: epsilon_min=0.01
   - Run B: epsilon_min=0.05
   - Run C: epsilon_min=0.10
3. View overlayed score curves
4. See which epsilon_min performed best
5. Use insights for next training run

### **Workflow 3: Historical Analysis**
1. View all training runs from past month
2. Filter by "Win Rate > 0%"
3. Identify trends:
   - Does curriculum learning help?
   - Do fewer enemies lead to better learning?
   - Which reward structure works best?
4. Document findings

---

## **NICE-TO-HAVE FEATURES (Future)**

- Export training data to CSV
- Email alerts when training completes
- Automatic best checkpoint detection
- Training run notes/comments
- Search/filter functionality
- Dark mode toggle
- Mobile-responsive design

---

## **ESTIMATED EFFORT**

- **Database Setup:** 1-2 hours
- **Data Importer:** 2-3 hours
- **Flask Backend:** 3-4 hours
- **Frontend Pages:** 4-5 hours
- **Testing & Polish:** 2-3 hours

**Total: ~15-20 hours of development**

Can be done incrementally over multiple sessions.

---

## **DEPENDENCIES**

```txt
# dashboard/requirements.txt
Flask==3.0.0
psycopg2-binary==2.9.9
SQLAlchemy==2.0.23
python-dotenv==1.0.0
pandas==2.1.3
```

---

## **NEXT STEPS**

1. ✅ Fix training foundation (reduce enemies, curriculum learning)
2. ✅ Validate agent can learn with new setup
3. ✅ Gather 3-5 successful training runs
4. 🔄 Implement Phase 1: Database Setup
5. 🔄 Implement Phase 2: Data Importer
6. 🔄 Implement Phase 3: Flask Backend
7. 🔄 Implement Phase 4: Frontend Pages
8. 🔄 Implement Phase 5: Polish & Deploy

**Priority: Foundation first, dashboard second!**

---

**This document will be referenced when we implement the Q6 Dashboard.**
