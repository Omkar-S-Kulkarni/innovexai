InnovexAI — Production Model Drift Monitoring System

Live Demo:
🔗 https://innovexbyprimedummu.streamlit.app/

Overview
InnovexAI is an end-to-end system for monitoring machine learning models in production without using ground truth labels.
It simulates a real deployment environment where:
The model is trained once and never retrained
Data distribution changes over time
Performance degradation happens silently
The system detects → warns → explains drift early

Demo Story
Healthy system
→ Gradual drift
→ Localized subgroup failure
→ Early warning
→ Root-cause diagnosis

The model keeps running the entire time.

Problem
In real-world deployments:
Labels arrive late or never
Models degrade silently
Failures often affect only specific user groups
Most monitoring systems rely on retraining or ground truth
InnovexAI solves this by detecting risk purely from model behavior.

Key Features
1. Time-Based Simulation
Sequential streaming data
No shuffling
Controlled drift scenarios:
Gradual global drift
Localized subgroup drift
Confidence collapse

2. Frozen Production Model
Trained only on early data
Model is saved and loaded once
No retraining during monitoring
Only predictions and probabilities are used

3. Output Drift Detection (Label-Free)
Tracks changes in:
Predicted class distribution
Probability distribution
Metrics:
KL Divergence
PSI (Population Stability Index)
Wasserstein Distance
Sliding window vs reference window comparison.

4. Confidence & Uncertainty Monitoring
Behavior signals:
Mean confidence
Probability margin (p1 − p2)
Prediction entropy
Detects silent degradation before accuracy drops.

5. Rolling Reference Windows
Reference updates dynamically
Prevents comparison with stale training data
Configurable window sizes from UI

6. Slice / Segment Monitoring
Detect hidden bias:
Manual feature-based slices
Cluster-based segments
Slice-level drift scores
Risk ranking vs global behavior

7. Non-Blocking Alert System
System never stops running
Severity levels:
Low
Medium
High
Cooldown to avoid alert spam
Clear explanation messages

8. Early Warning Intelligence
Behavior fingerprint:
Confidence trend
Entropy trend
Class balance
Drift slope / acceleration
Shows trend direction and risk evolution.

9. Automatic Slice Discovery
Unsupervised clustering (KMeans)
Detects hidden risky groups
Displays top high-risk clusters

10. Drift Attribution
Explains why drift happened
Feature distribution comparison
Stable vs risky window analysis
High-risk vs low-risk slice comparison

11. Stability & Fragility Tests
Noise perturbation
Decision flip rate
Counterfactual sensitivity
Detects model brittleness.

12. Composite Drift Risk Score
Combines:
Output drift
Confidence
Entropy
Stability
Produces a single risk score for decision makers.

13. Drift Regime Classification
Identifies drift patterns:
Gradual
Sudden
Localized
Oscillatory

14. Automated Self-Audit Report
Generates:
What changed
When it changed
Where the risk exists
Confidence level
Known blind spots

15. Failure & Blind-Spot Matrix
Explicitly documents:
Detectable:
Distribution drift
Confidence collapse
Subgroup bias
Limitations:
Label noise detection
Concept drift with unchanged outputs
Perfectly calibrated silent errors
Application Structure
Streamlit Tabs

Overview
Drift Signals
Slice Monitoring
Diagnosis
Audit Report
Sidebar controls:
Window size
Thresholds
Slice selection
Reference configuration

Tech Stack
Python
Streamlit
NumPy, Pandas
Scikit-learn
SciPy
Plotly / Matplotlib

Architecture Modules:
model/
training
inference engine
monitoring/
window manager
output distribution
confidence & entropy
rolling reference
alert engine

How to Run Locally

git clone <https://github.com/Omkar-S-Kulkarni/innovexai.git>

cd innovexai

pip install -r requirements.txt

streamlit run test.py


Deployment

Hosted on Streamlit Cloud:

🔗 https://innovexbyprimedummu.streamlit.app/

Final Demo Guarantees
No ground truth used for detection
Model never retrains
Drift is gradual and realistic
Localized bias is surfaced
System never stops running

Clear pipeline:
Detect → Warn → Explain
