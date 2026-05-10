# MTU SensorMind: Enterprise Equipment Health Co-Pilot
*A Data Science & Software Engineering Portfolio Project for Rolls-Royce Power Systems*

MTU SensorMind is an end-to-end agentic system designed for industrial turbofan/gas engine maintenance. It bridges the gap between raw **Data Science (Python/ML)** and **Production Software Engineering (C#/.NET)**. 

It uses machine learning for anomaly detection and RUL prediction, combined with a LangGraph-powered agent that reasons over maintenance knowledge bases to generate actionable **Maintenance Repair and Overhaul (MRO)** work orders.

## 🏗️ System Architecture

```text
+-----------------------------------------------------------------------+
|                 MTU Fleet Manager Bridge (C# / .NET)                  |
|       (Native Windows executable to launch and monitor the UI)        |
+-----------------------------------------------------------------------+
                                  |
+-----------------------------------------------------------------------+
|                          MTU SensorMind Dashboard                     |
|      (Streamlit: Fleet Monitor, Engine Deep Dive, Agent Trace)        |
+-----------------------------------------------------------------------+
           ^                                         |
           | (State Updates)                         | (User Queries)
           |                                         v
+-----------------------+              +--------------------------------+
|    LangGraph Agent    | <-----------> |       Voice/Text Input         |
|   (State Management)  |              |    (Groq Whisper Integration)  |
+-----------------------+              +--------------------------------+
           |
           | (Orchestrates Guided Troubleshooting)
           v
+-----------------------------------------------------------------------+
|                             AGENT TOOLS                               |
| +-------------------+  +-------------------+  +-------------------+   |
| | Anomaly Analyzer  |  | RAG Knowledge Base|  | Gemini 1.5 Flash  |   |
| | (Z-Score Analysis)|  | (FAISS + MiniLM)  |  | (Action Planning)  |   |
| +-------------------+  +-------------------+  +-------------------+   |
|          |                      |                      |              |
| +-------------------+  +-------------------+          v              |
| | What-If Simulator |  | PDF Work Order Gen|  +-------------------+   |
| | (Degradation Proj)|  | (ReportLab)       |  |     Work Order    |   |
| +-------------------+  +-------------------+  +-------------------+   |
+-----------------------------------------------------------------------+
           ^
           | (Sensor Data & Data Quality Checks)
           |
+-----------------------------------------------------------------------+
|                            ML ENGINE                                  |
| +-----------------------------------+  +----------------------------+ |
| |       LSTM Autoencoder            |  |      XGBoost Regressor     | |
| |      (Anomaly Detection)          |  |      (RUL Prediction)      | |
+-------------------------------------+  +----------------------------+ |
+-----------------------------------------------------------------------+
|             NASA CMAPSS Dataset (FD001) / Edge Sensors                |
+-----------------------------------------------------------------------+
```

## 🌟 Key Enterprise Features (Aligning with RR Requirements)

1. **C# Windows Tooling:** Includes a native C# `.exe` launcher (`MTU_Fleet_Manager.exe`) that bridges the Python data science environment with a standard Windows desktop environment.
2. **Data Quality Layer:** Industrial sensors fail. The ML inference pipeline includes a data sanity check to identify "frozen" sensors (0 variance) before running complex ML, preventing false positives.
3. **MRO & Guided Troubleshooting:** The LLM does not hallucinate repairs. It uses a **FAISS Vector Database** to perform RAG over 30 MTU-specific maintenance rules. The agent is forced to **cite its sources** and consider the engine's **Service History** before generating a Work Order.
4. **Predictive Analytics:** An LSTM Autoencoder (Deep Learning) detects anomalous sensor behavior, while an XGBoost model predicts the exact Remaining Useful Life (RUL).

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Windows OS (for C# Launcher)
- API Keys: Gemini API Key & Groq API Key (added to `.env` file)

### 2. Installation
```bash
pip install -r requirements.txt
```

### 3. Launching the System
You can launch the system using the native C# Windows tool:
1. Navigate to the project root.
2. Double-click `MTU_Fleet_Manager.exe`.
3. The C# bridge will initialize the Python backend and open the dashboard in your default browser.

*(Alternatively, run `streamlit run dashboard/app.py`)*

## ☁️ Azure Cloud Deployment Roadmap
To scale this solution globally across the MTU fleet, the following Azure architecture would be implemented:
*   **Data Lake:** Bronze/Silver/Gold data architecture in **Azure Databricks** to process TBs of engine telemetry using PySpark.
*   **Model Deployment:** The LSTM and XGBoost models would be containerized and hosted on **Azure Machine Learning** managed endpoints.
*   **Knowledge Base:** The FAISS index would be migrated to **Azure AI Search** for enterprise-grade semantic retrieval.
*   **Dashboard:** Deployed via **Azure App Service** or wrapped in a **Power BI** custom visual for executive reporting.
