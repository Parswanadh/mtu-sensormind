import os
import json
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from .tools import (
    analyze_anomaly,
    query_maintenance_knowledge,
    generate_maintenance_action,
    simulate_whatif
)
from utils.work_order import generate_pdf

load_dotenv()

class AgentState(TypedDict):
    engine_id: int
    sensor_deviations: Dict[str, float]
    anomaly_score: float
    is_anomaly: bool
    rul: int
    risk_level: str
    delay_cycles: Optional[int]
    service_history: str # Added MRO context
    
    fault_analysis: List[str]
    retrieved_procedures: List[str]
    maintenance_plan: Dict[str, Any]
    simulation_result: Dict[str, Any]
    work_order_path: str
    
    trace_log: List[str]

def anomaly_detector_node(state: AgentState) -> AgentState:
    log = []
    log.append(f"🔍 Analyzing Engine {state['engine_id']}...")
    if state['is_anomaly']:
        log.append(f"⚠️ Anomaly detected! Score: {state['anomaly_score']} (Threshold: 0.5)")
    elif state['risk_level'] == "DATA_FAULT":
        log.append(f"❌ Data Quality Fault detected. Sensors may be frozen or malfunctioning.")
    else:
        log.append(f"✅ Engine {state['engine_id']} operating normally. RUL: {state['rul']} cycles.")
        
    return {"trace_log": state.get('trace_log', []) + log}

def routing_condition(state: AgentState) -> str:
    if state['is_anomaly'] or state['risk_level'] in ['HIGH', 'CRITICAL', 'DATA_FAULT']:
        return "reasoning"
    return "end"

def reasoning_node(state: AgentState) -> AgentState:
    log = []
    
    log.append("📊 Identifying top anomalous sensors...")
    faults = analyze_anomaly(state['sensor_deviations'])
    faults_str = ", ".join([f.split(':')[0] for f in faults])
    log.append(f"⚠️ Top deviations: {faults_str}")
    
    log.append("📚 Querying Guided Troubleshooting Knowledge Base...")
    query_str = " ".join(faults) + f" RUL {state['rul']}"
    procedures = query_maintenance_knowledge(query_str)
    for p in procedures:
        log.append(f"💡 {p}") # Explicitly show the citation in the trace
    
    log.append(f"📋 Reviewing MRO Service History: {state.get('service_history', 'None')}")
    
    log.append("🤖 Synthesizing Action Plan via LLM...")
    plan = generate_maintenance_action(
        state['engine_id'], 
        faults, 
        procedures, 
        state['rul'],
        state.get('service_history', '')
    )
    log.append(f"🔧 Recommended Action: {plan.get('action')}")
    log.append(f"🚨 Urgency: {plan.get('urgency')}")
    
    return {
        "fault_analysis": faults,
        "retrieved_procedures": procedures,
        "maintenance_plan": plan,
        "trace_log": state.get('trace_log', []) + log
    }

def simulation_node(state: AgentState) -> AgentState:
    log = []
    delay = state.get('delay_cycles', 10)
    
    log.append(f"🔮 Simulating what-if scenario: Delay maintenance by {delay} cycles...")
    sim_res = simulate_whatif(state['rul'], state['anomaly_score'], delay)
    
    log.append(f"📉 Projected RUL: {sim_res['projected_rul_after_delay']} cycles.")
    log.append(f"⚠️ Failure Probability Increase: {sim_res['failure_probability_increase']}%")
    
    return {
        "simulation_result": sim_res,
        "trace_log": state.get('trace_log', []) + log
    }

def work_order_node(state: AgentState) -> AgentState:
    log = []
    log.append("📄 Generating official Work Order PDF...")
    
    fault_summary = "\n".join(state['fault_analysis'])
    pdf_path = generate_pdf(state['engine_id'], state['maintenance_plan'], fault_summary, state['rul'])
    
    wo_name = os.path.basename(pdf_path)
    log.append(f"✅ Work Order Generated: {wo_name}")
    
    return {
        "work_order_path": pdf_path,
        "trace_log": state.get('trace_log', []) + log
    }

def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)
    workflow.add_node("anomaly_detector", anomaly_detector_node)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("simulation", simulation_node)
    workflow.add_node("work_order", work_order_node)
    workflow.set_entry_point("anomaly_detector")
    workflow.add_conditional_edges("anomaly_detector", routing_condition, {"reasoning": "reasoning", "end": END})
    workflow.add_edge("reasoning", "simulation")
    workflow.add_edge("simulation", "work_order")
    workflow.add_edge("work_order", END)
    return workflow.compile()
