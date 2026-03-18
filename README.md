# 💾 NVMe Drive Failure Pattern Analysis & Prediction

> A machine learning system that analyzes NVMe SSD SMART telemetry to detect early signs of drive failure — built as a predictive maintenance tool for enterprise storage environments.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Project Overview

NVMe SSDs are widely used in modern computing systems for their high performance and low latency. However, like all storage devices, they degrade over time due to write endurance limits, thermal stress, power interruptions, and firmware issues. When failure occurs, it can lead to data corruption, performance loss, or complete device outage.

This project analyzes NVMe drive SMART telemetry to:
- Identify the **top failure patterns** across a fleet of drives
- Understand the relationship between **temperature, usage, and error accumulation**
- Predict **which drives are at risk** before they fail
- Enable **early detection and proactive maintenance** decisions

---

## 🖥️ Dashboard Preview

The project ships as a fully interactive **5-page Streamlit dashboard**:

| Page | Description |
|---|---|
| 🏠 Fleet Overview | KPI metrics, failure mode breakdown, risk distribution, vendor & firmware analysis |
| 🔍 Failure Pattern Analysis | EDA charts, SMART metric comparisons, scatter plots, correlation heatmap |
| 🤖 Live Drive Predictor | Enter any drive's SMART values → instant risk score + recommended action |
| 📈 ML Model Performance | Accuracy, confusion matrix, ROC curve, feature importance |
| ⚠️ At-Risk Drive Table | Sortable risk table with CSV export |

---

## 📂 Project Structure

```
nvme-failure-analysis/
│
├── nvme_complete.py                  # Main file — run this with Streamlit
├── NVMe_Drive_Failure_Dataset.csv    # Dataset (place in same folder)
└── README.md                         # This file
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

**Step 1 — Clone the repository**
```bash
git clone https://github.com/your-username/nvme-failure-analysis.git
cd nvme-failure-analysis
```

**Step 2 — Install required libraries**
```bash
pip install streamlit plotly scikit-learn imbalanced-learn pandas numpy
```

**Step 3 — Place your dataset in the project folder**
```
nvme-failure-analysis/
├── nvme_complete.py
└── NVMe_Drive_Failure_Dataset.csv   ← here
```

**Step 4 — Run the dashboard**
```bash
streamlit run nvme_complete.py
```

Your browser will open automatically at `http://localhost:8501`

---

## 📊 Dataset

The dataset contains **10,000 NVMe drive health snapshots** with the following features:

### Feature Columns

| Column | Type | Description |
|---|---|---|
| `Drive_ID` | string | Unique drive identifier |
| `Vendor` | string | Drive manufacturer |
| `Model` | string | Drive model name |
| `Firmware_Version` | string | Installed firmware version |
| `Power_On_Hours` | int | Total hours the drive has been powered on |
| `Total_TBW_TB` | float | Total terabytes written |
| `Total_TBR_TB` | float | Total terabytes read |
| `Temperature_C` | float | Drive temperature in Celsius |
| `Percent_Life_Used` | float | Percentage of drive lifespan consumed |
| `Media_Errors` | int | Count of media errors reported |
| `Unsafe_Shutdowns` | int | Count of unexpected power-off events |
| `CRC_Errors` | int | CRC error count |
| `Read_Error_Rate` | float | Read error rate value |
| `Write_Error_Rate` | float | Write error rate value |
| `SMART_Warning_Flag` | int | 0 = no warning, 1 = SMART warning active |

### Target Columns

| Column | Values | Description |
|---|---|---|
| `Failure_Flag` | 0 = Healthy, 1 = Failing | Binary classification target |
| `Failure_Mode` | 0, 1, 4, 5 | Multi-class failure category |

### Failure Mode Mapping

| Mode | Label | Description |
|---|---|---|
| 0 | Healthy | No abnormal metrics detected |
| 1 | Wear-Out Failure | Flash nearing end-of-life (Percent Life Used ~99%) |
| 2 | Thermal Failure | Sustained high temperature >70°C *(not present in dataset)* |
| 3 | Power-Related Failure | Multiple unsafe shutdowns causing corruption *(not present in dataset)* |
| 4 | Controller / Firmware Failure | Elevated read errors + media errors |
| 5 | Early-Life Defect | High error rate within first 3,000 hours |

---

## 🔍 Key Findings

### Top Failure Patterns Detected

| Rank | Failure Mode | Count | Key Signal | Finding |
|---|---|---|---|---|
| 1 | Controller / Firmware | 85 drives | Read Error Rate | 2× higher than healthy drives |
| 2 | Early-Life Defect | 78 drives | Power-On Hours | Fails at avg 1,608 hrs vs 29,968 healthy |
| 3 | Wear-Out Failure | 31 drives | Percent Life Used | Avg 99.7% vs 22% in healthy drives |

### Class Imbalance
The dataset has a **50:1 class imbalance** (9,806 healthy vs 194 failing). This is handled using **SMOTE (Synthetic Minority Oversampling Technique)** combined with `class_weight='balanced'` in the Random Forest classifier.

---

## 🤖 Machine Learning Model

### Architecture
- **Algorithm:** Random Forest Classifier (200 trees, max depth 12)
- **Imbalance handling:** SMOTE oversampling on training set
- **Train/test split:** 80% / 20% stratified
- **Two models trained:**
  - Binary classifier → Healthy vs Failing
  - Multiclass classifier → Which failure mode

### Model Performance

| Metric | Score |
|---|---|
| Accuracy | 100% |
| Precision (Failing) | 100% |
| Recall (Failing) | 100% |
| F1-Score | 100% |
| ROC-AUC | 1.0000 |

> **Note:** Near-perfect scores are expected with a synthetic dataset where failure signals are clearly defined. Real-world performance would typically range from 85–95%.

### Top Predictive Features
1. `Percent_Life_Used` — strongest predictor of wear-out failure
2. `Read_Error_Rate` — key signal for controller/firmware issues
3. `Power_On_Hours` — early-life defects cluster at very low hours
4. `Write_Error_Rate` — elevated in early-life and wear-out modes
5. `Media_Errors` — elevated in controller failures

---

## ⚙️ How It Works

```
Dataset loaded
      ↓
Preprocessing — drop Drive_ID, encode Vendor/Model/Firmware
      ↓
EDA — distributions, correlations, failure pattern analysis
      ↓
Train Random Forest (binary + multiclass) with SMOTE
      ↓
Batch score all 10,000 drives → Risk_Pct + Risk_Level
      ↓
Streamlit dashboard — 5 interactive pages
      ↓
Live predictor — enter any drive's SMART values → instant result
```

---

## 📦 Dependencies

```txt
streamlit
plotly
scikit-learn
imbalanced-learn
pandas
numpy
```

Install all at once:
```bash
pip install streamlit plotly scikit-learn imbalanced-learn pandas numpy
```

---

## 🔮 Future Improvements

- [ ] Add support for real-time SMART data ingestion via `nvme-cli`
- [ ] Include Thermal (Mode 2) and Power-Related (Mode 3) failure samples
- [ ] Add time-series analysis for drive degradation trends
- [ ] Deploy as a web app using Streamlit Cloud or Docker
- [ ] Add email/SMS alert system for CRITICAL risk drives
- [ ] Extend to support HDD and SATA SSD telemetry

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Dataset: Synthetic NVMe SMART telemetry representing real-world drive behavior
- Built with [Streamlit](https://streamlit.io), [scikit-learn](https://scikit-learn.org), and [Plotly](https://plotly.com)
- Developed as part of a predictive maintenance research project

---

*Made with ❤️ for proactive storage management*
