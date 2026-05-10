# Literature Survey: Predictive Maintenance for Turbofan Engines

## 1. Introduction
The MTU SensorMind project leverages the **NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation)** dataset. Predictive maintenance in aerospace requires high-fidelity Remaining Useful Life (RUL) estimation to ensure flight safety and optimize Maintenance, Repair, and Overhaul (MRO) cycles.

## 2. Key Research Papers Surveyed

### A. LSTM-based RUL Prediction (Saxena et al., 2008)
*   **Context:** Foundations of the CMAPSS dataset.
*   **Methodology:** Identified that sensor signals exhibit non-linear degradation. Recurrent Neural Networks (RNNs) like LSTMs are superior to linear regression because they capture temporal dependencies (memory of past cycles).
*   **Our Implementation:** We utilized an **LSTM Autoencoder** for anomaly detection, as it effectively learns the "normal" manifold of healthy engine operation.

### B. Gradient Boosting for RUL (Zheng et al., 2017)
*   **Context:** Comparison of Deep Learning vs. Ensemble Methods.
*   **Methodology:** Found that while LSTMs excel at sequence modeling, **XGBoost (Extreme Gradient Boosting)** is highly robust for cycle-by-cycle regression when rolling statistics (mean, std) are used as features.
*   **Our Implementation:** We used XGBoost for the final RUL regressor, utilizing a rolling window of 10 cycles to stabilize sensor noise.

### C. RAG and Agentic Maintenance (Modern Trend)
*   **Context:** The shift from "Diagnostics" to "Guided Troubleshooting."
*   **Methodology:** Large Language Models (LLMs) can synthesize complex sensor data with technical manuals. However, Retrieval Augmented Generation (RAG) is necessary to ground the LLM in specific company maintenance rules.
*   **Our Implementation:** We implemented a **FAISS-based RAG** loop that grounds the Gemini 1.5 Flash agent in 30 MTU-specific maintenance rules.

## 3. Comparative Analysis
| Model | Strength | Weakness | Used In |
| :--- | :--- | :--- | :--- |
| **LSTM** | Handles time-series dependencies | Computationally expensive | Anomaly Detection |
| **XGBoost** | High accuracy, fast inference | Requires manual feature engineering | RUL Prediction |
| **Vanilla LLM** | Excellent reasoning | Hallucinations | N/A |
| **Agentic RAG** | Traceable, Guided Troubleshooting | Requires high-quality knowledge base | Maintenance Planning |

## 4. Conclusion
The hybrid architecture of MTU SensorMind—combining Deep Learning (LSTM), Gradient Boosting (XGBoost), and Agentic AI (LangGraph/RAG)—is the optimal path for industrial-grade Equipment Health Monitoring (EHM).
