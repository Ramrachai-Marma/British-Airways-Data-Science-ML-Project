# British Airways Data Science Project

Two data science tasks on British Airways data: a data analysis of lounge demand at Heathrow Terminal 3 and a machine learning model that predicts customer bookings. The tasks and datasets come from the [British Airways Data Science programme on Forage](https://www.theforage.com/). Each task has its own folder with the problem statement, the data, the code and a write-up of what I did and found.

| Task | Problem | What I built | Headline result |
| --- | --- | --- | --- |
| [1. Data analysis: lounge eligibility at Heathrow Terminal 3](task1-data-analysis-lounge-eligibility/) | Estimate how many passengers are eligible for each lounge tier, without relying on exact flight or aircraft details | A reusable lookup table by route type and region, calculated from 10,000 scheduled departures | Haul drives eligibility: Tier 3 is 16.9% of seats on short-haul and 10.5% on long-haul. Time of day makes no difference |
| [2. Machine learning: predicting customer bookings](task2-machine-learning-booking-prediction/) | Predict which customers complete a booking and explain which variables drive it | A random forest on 49,281 bookings, with 5-fold cross-validation and permutation importance | ROC AUC 0.80, recall 0.66, precision 0.34. Booking origin and route carry most of the predictive power |

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── task1-data-analysis-lounge-eligibility/
│   ├── README.md                  Problem, approach and results
│   ├── lounge_eligibility.py      Builds the lookup table from the schedule
│   ├── data/                      Summer schedule dataset (10,000 flights)
│   ├── outputs/                   Lookup table and supporting tables (CSV)
│   └── docs/                      Problem statement (PDF)
└── task2-machine-learning-booking-prediction/
    ├── README.md                  Problem, approach and results
    ├── booking_model.ipynb        Main analysis notebook
    ├── task2_analysis.py          Script version with a one-hot baseline
    ├── feature_importance.png     Chart produced by the notebook
    ├── data/                      Customer booking dataset (50,000 rows)
    ├── reports/                   Metrics, tables and charts from the script
    ├── presentation/              One-slide summary (PDF)
    └── docs/                      Problem statement (PDF)
```

## How to run

Python 3.9 or newer.

```bash
python -m pip install -r requirements.txt

# Task 1: writes the lookup table to task1-data-analysis-lounge-eligibility/outputs/
python task1-data-analysis-lounge-eligibility/lounge_eligibility.py

# Task 2: open the notebook from inside its folder
cd task2-machine-learning-booking-prediction
jupyter notebook booking_model.ipynb

# Task 2, script version: writes to task2-machine-learning-booking-prediction/reports/
python task2-machine-learning-booking-prediction/task2_analysis.py
```

## Tools

Python, pandas, scikit-learn, matplotlib, Jupyter.

## Data and disclaimer

The datasets and task briefs were supplied by British Airways through Forage for learning purposes. This is a portfolio project, not official British Airways work, and the models are not intended for production decisions.
