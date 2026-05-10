import os
import json
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Sensor mappings for interpretability
SENSOR_MAP = {
    's2': 'T24 (Total temp HPC outlet)',
    's3': 'T30 (Total temp LPC outlet)',
    's4': 'T50 (Total temp LPT outlet)',
    's7': 'Ps30 (Static pressure HPC outlet)',
    's8': 'Nf (Physical fan speed)',
    's9': 'Nc (Physical core speed)',
    's11': 'Ps50 (Static pressure HPC outlet)',
    's12': 'phi (Ratio of fuel flow to Ps30)',
    's13': 'NRf (Corrected fan speed)',
    's14': 'NRc (Corrected core speed)',
    's15': 'BPR (Bypass Ratio)',
    's17': 'htBleed (Bleed Enthalpy)',
    's20': 'W31 (HPT coolant bleed)',
    's21': 'W32 (LPT coolant bleed)'
}

_faiss_index = None
_corpus = None
_embed_model = None

def _load_kb():
    global _faiss_index, _corpus, _embed_model
    if _faiss_index is not None:
        return
        
    base_dir = os.path.dirname(os.path.dirname(__file__))
    index_path = os.path.join(base_dir, 'knowledge_base', 'faiss_index.bin')
    corpus_path = os.path.join(base_dir, 'knowledge_base', 'rules_corpus.pkl')
    
    if os.path.exists(index_path) and os.path.exists(corpus_path):
        _faiss_index = faiss.read_index(index_path)
        with open(corpus_path, 'rb') as f:
            _corpus = pickle.load(f)
        import logging
        logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
        _embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    else:
        print("Warning: Knowledge base not found.")

def analyze_anomaly(sensor_deviations: dict) -> list:
    sorted_sensors = sorted(sensor_deviations.items(), key=lambda x: x[1], reverse=True)
    top_3 = sorted_sensors[:3]
    
    analysis = []
    for sensor, dev in top_3:
        human_name = SENSOR_MAP.get(sensor, sensor)
        analysis.append(f"{human_name}: deviation {dev:.4f}")
    return analysis

def query_maintenance_knowledge(fault_description: str) -> list:
    """
    RAG tool. Returns the exact rules to enforce Guided Troubleshooting.
    """
    _load_kb()
    if _faiss_index is None:
        return ["No knowledge base available."]
        
    query_emb = _embed_model.encode([fault_description], convert_to_numpy=True)
    distances, indices = _faiss_index.search(query_emb, k=2)
    
    results = []
    for idx in indices[0]:
        if idx < len(_corpus):
            results.append(f"MTU Rule Match: '{_corpus[idx]}'")
    return results

def generate_maintenance_action(engine_id: int, fault_analysis: list, procedures: list, rul: int, service_history: str = "No recent major overhauls.") -> dict:
    """
    Calls Gemini to synthesize a structured JSON maintenance plan.
    Enforces MRO history and Knowledge Base citation.
    """
    prompt = PromptTemplate.from_template(
        """You are a Senior MTU Turbofan Engine Reliability Engineer.
        
        Engine ID: {engine_id}
        Estimated Remaining Useful Life (RUL): {rul} cycles.
        
        MRO Service History:
        {history}
        
        Top Anomalous Sensors Detected:
        {faults}
        
        Guided Troubleshooting Procedures (from MTU Knowledge Base):
        {procedures}
        
        Synthesize this into a structured JSON action plan.
        CRITICAL: You MUST explicitly cite the 'MTU Rule Match' in your technician_notes to prove guided troubleshooting. 
        Take the MRO Service History into account (e.g. if a part was just replaced, it might be an installation error).
        
        Output Schema:
        {{
            "action": "A clear, concise 1-sentence action summary",
            "urgency": "Choose one: IMMEDIATE, 48HR, SCHEDULED",
            "estimated_downtime_hrs": 12,
            "parts_required": ["part 1", "part 2"],
            "technician_notes": "Detailed notes. MUST begin with 'Based on [MTU Rule Citation]...'"
        }}
        """
    )
    
    faults_str = "\n".join(f"- {f}" for f in fault_analysis)
    proc_str = "\n".join(f"- {p}" for p in procedures)
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    chain = prompt | llm
    
    response = chain.invoke({
        "engine_id": engine_id,
        "rul": rul,
        "history": service_history,
        "faults": faults_str,
        "procedures": proc_str
    })
    
    txt = response.content.strip()
    if txt.startswith("```json"): txt = txt[7:]
    if txt.startswith("```"): txt = txt[3:]
    if txt.endswith("```"): txt = txt[:-3]
        
    try:
        plan = json.loads(txt.strip())
        return plan
    except Exception as e:
        print(f"Error parsing Gemini response: {e}")
        return {
            "action": "Inspect engine sensors based on deviation.",
            "urgency": "SCHEDULED",
            "estimated_downtime_hrs": 24,
            "parts_required": [],
            "technician_notes": "Failed to generate LLM response."
        }

def simulate_whatif(rul: int, anomaly_score: float, delay_cycles: int) -> dict:
    base_drop = delay_cycles
    penalty = delay_cycles * (anomaly_score / 0.5) * 0.2
    projected_rul = max(0, int(rul - base_drop - penalty))
    
    base_prob = max(0, (50 - rul) * 2)
    new_prob = max(0, (50 - projected_rul) * 2)
    prob_increase = min(100.0, float(new_prob - base_prob))
    
    if projected_rul < 5:
        recommendation = "CRITICAL RISK: Delaying will likely result in in-flight shutdown (IFSD)."
    elif prob_increase > 20:
        recommendation = "HIGH RISK: Delaying significantly accelerates component fatigue."
    else:
        recommendation = "ACCEPTABLE: Delaying is within operational safety margins."
        
    return {
        "current_rul": rul,
        "projected_rul_after_delay": projected_rul,
        "failure_probability_increase": round(prob_increase, 1),
        "recommendation": recommendation
    }
