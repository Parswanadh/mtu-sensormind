# Agile Development Process: MTU SensorMind

This project followed an **Agile SCRUM** methodology to ensure iterative delivery of the Equipment Health Co-Pilot.

## 1. User Stories

| ID | Role | Requirement | Goal |
| :--- | :--- | :--- | :--- |
| US-01 | Fleet Manager | I want to see the status of all engines in a single heatmap. | To identify high-risk assets immediately. |
| US-02 | Maintenance Tech | I want a guided troubleshooting plan with rule citations. | To perform repairs without searching manuals. |
| US-03 | Reliability Eng | I want a "What-If" simulator for maintenance delays. | To justify costs and safety risks to management. |
| US-04 | IT Lead | I want a native Windows launcher (C#). | To integrate the tool into existing workstation environments. |

## 2. Sprint Cycles

### Sprint 1: Data Engineering & ML Foundation (1 Week)
*   **Goal:** Clean CMAPSS data and train baseline models.
*   **Outcome:** `train_models.py` and `inference.py` operational.

### Sprint 2: Agentic Reasoning & RAG (1 Week)
*   **Goal:** Build the LangGraph orchestration and FAISS index.
*   **Outcome:** Agent successfully generates Work Orders based on MTU rules.

### Sprint 3: Enterprise Integration & UX (1 Week)
*   **Goal:** Build Streamlit dashboard and C# Launcher.
*   **Outcome:** End-to-end "shippable" product.

## 3. Backlog (Future Enhancements)
*   [ ] **Real-time IoT Integration:** Replace static CMAPSS files with a live MQTT stream.
*   [ ] **Azure SQL Migration:** Move SQLite asset data to Azure SQL Database.
*   [ ] **Multi-Agent Collaboration:** Dedicated agents for "Parts Inventory" vs. "Engine Diagnostics."

## 4. Retrospective
The project successfully met the core goal of bridging the gap between ML models and actionable MRO workflows. The addition of the **Data Quality Gate** was a critical "Sprint 3" pivot after recognizing the frequency of sensor failures in industrial environments.
