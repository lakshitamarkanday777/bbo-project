# 🦉 BBO Owl Detection & Movement Modeling  
### Behavioral Insights, Predictive Modeling, and Explainable AI

## 📌 Project Overview
This project was developed in collaboration with the **Beaverhill Bird Observatory (BBO)** to analyze and model the movement behavior of **Northern Saw-whet Owls** using data from the **Motus Wildlife Tracking System**.

The project delivers a complete **end-to-end machine learning pipeline**, transforming raw wildlife telemetry data into **interpretable behavioral insights** through data analysis, predictive modeling, explainable AI, and an interactive Streamlit application.

---

## 🎯 Objectives
- Analyze owl movement patterns across time, direction, and signal strength  
- Engineer meaningful behavioral features from raw telemetry detections  
- Build machine learning models to:
  - Classify migration activity  
  - Predict detection duration  
  - Predict signal strength (SNR)  
- Apply **Explainable AI (SHAP)** to ensure model transparency  
- Deploy an interactive **Streamlit dashboard** for researchers and stakeholders  

---

## 🗂️ Dataset Description
- **Source:** Motus Wildlife Tracking System (via BBO)  
- **Monitoring Period:** July 2024 – November 2024  
- **Dataset Size:**  
  - 102,678 detection records  
  - 57 features (raw + engineered)

### Key Data Features
- Telemetry signals: signal strength (SNR), noise, frequency drift  
- Temporal data: timestamps, hour, day, month  
- Antenna direction / port data (SE, SW, N, Omni)  
- Tag deployment metadata  
- Engineered features:
  - `migration_activity`
  - `direction_str`
  - `days_since_deployment`
  - Movement category labels  

---

## 🔍 Exploratory Data Analysis (EDA)
Key insights from EDA include:
- Peak detection activity between **4–8 AM**, confirming nocturnal behavior  
- Highest detections during **September–October**, aligning with fall migration  
- Dominant movement directions toward **SE and SW**  
- Migratory detections exhibit **higher SNR values**  
- Signal strength decreases over time due to tag aging  
- Distinct movement categories such as one-night passage, short-stay, and multi-day migrants  

EDA results are available through visualizations embedded in the Streamlit app.

---

## 🤖 Machine Learning Models
Three machine learning models were developed using **Random Forest** algorithms:

### Model A – Migration Activity Classification
- **Type:** Binary Classification  
- **Goal:** Predict migratory vs non-migratory detections  
- **Techniques:** SMOTE for class imbalance  
- **Performance:**  
  - Accuracy: ~96%  
  - F1-score: ~0.97  

### Model B – Detection Duration Regression
- **Type:** Regression  
- **Goal:** Predict duration of owl detection events  
- **Performance:**  
  - R² ≈ 0.99  

### Model C – Signal Strength (SNR) Regression
- **Type:** Regression  
- **Goal:** Predict signal strength based on tag age and movement features  
- **Performance:**  
  - R² ≈ 0.85  

---

## 🧠 Explainable AI (XAI)
To ensure transparency and interpretability:
- **SHAP** was used for both global and local explanations  
- Feature importance highlights key predictors such as:
  - Signal strength  
  - Direction of movement  
  - Time of detection  
  - Days since deployment  

XAI outputs are integrated directly into the Streamlit application.

---

## 🚀 Deployment
The project is deployed as a **multi-page Streamlit application**, enabling users to:
- Explore EDA interactively  
- Run predictions using trained models  
- Interpret results through SHAP visualizations  
- Explore owl movement behavior via dashboards  

### Tools Used
- Python, Pandas, NumPy  
- Scikit-learn  
- SHAP  
- Plotly  
- Streamlit  
- GitHub  
- Google Colab  

---


---

## ⚠️ Challenges & Solutions
- **Noisy telemetry data:** Cleaned and standardized timestamps and signals  
- **Imbalanced classes:** Addressed using SMOTE  
- **Large model sizes:** Optimized storage and deployment strategy  
- **Complex behavior patterns:** Used tree-based models for robustness  

---

## 👥 Stakeholder Collaboration
- Continuous feedback from BBO researchers  
- Focus on multi-day tag analysis (e.g., tags 80830, 80821, 80805)  
- Visualization and modeling aligned with ecological research needs  

---

## ✅ Conclusion
This project successfully demonstrates how machine learning and explainable AI can support ecological research. By integrating predictive modeling, interpretability, and interactive deployment, the solution provides BBO with a scalable and transparent tool to better understand owl migration behavior and telemetry patterns.

---

## 📎 Repository
🔗 GitHub: https://github.com/lakshitamarkanday777/bbo-project

