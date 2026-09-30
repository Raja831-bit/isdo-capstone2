"""
ISDO Lab C6/C7 - LangGraph Orchestrator with an extended HITL approval gate.

    START -> triage -> resolution -> sla --(hitl_required?)--> hitl -> communication -> END
                                        \-------------(no)-------------^

Nodes (each reads the shared TicketState and returns only the fields it owns):
  triage_node         Claude classifies the ticket (structured JSON via tool call)
  resolution_node     ChromaDB 'isdo_kb' search (Lab C1) + Claude drafts the fix steps
  sla_node            SLA deadline check (Lab C5 rules) + all HITL trigger checks
  hitl_node           human approval via input(), shows hitl_reason
  communication_node  Claude drafts the user message; sets final_status

HITL triggers (Lab C7) - any one sets hitl_required=True and adds to hitl_reason:
  P1_SLA          P1 ticket whose SLA is CRITICAL or BREACHED
  LOW_CONFIDENCE  Resolution Agent confidence is LOW (any priority)
  ACCESS_GRANT    category 'Access' and request_type 'Access Grant' (any priority)

Guardrails are decided in code, not by the model: confidence comes from the KB score,
auto_resolve is never True for P1, LOW confidence or access grants.
Every node appends to audit_log (timestamp, agent, action, detail) -> persisted in Lab C9.

Run from the project folder (ServiceNow 5001 + Jira 5002 shims recommended):
    python orchestrator/supervisor.py              # all test tickets
    python orchestrator/supervisor.py REQ-1002     # just one (e.g. to re-test y/n)

Temperature is not set: claude-opus-5 and later reject non-default sampling
parameters with a 400 error (see Lab C3 notes).
"""

import importlib.util
import json
import operator
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict

import anthropic
import chromadb
import requests
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")        # safe printing of the arrows on Windows

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = PROJECT_ROOT / "data" / "kb"
DB_DIR = PROJECT_ROOT / "data" / "chroma_db"
HITL_LOG = PROJECT_ROOT / "logs" / "hitl_decisions.jsonl"
SNOW_URL = "http://localhost:5001/api/now/table/incident"
JIRA_URL = "http://localhost:5002/rest/api/2/issue"

load_dotenv(PROJECT_ROOT / ".env")
if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit("ANTHROPIC_API_KEY is not set. Add it to the .env file in the project folder.")
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5")
client = anthropic.Anthropic()

# Rules shared with Labs C3-C5
HIGH_THRESHOLD, MEDIUM_THRESHOLD = 0.60, 0.35
SIMULATED_NOW = datetime(2024, 1, 15, 10, 30)
SLA_TARGET_MIN = {"P1": 60, "P2": 240, "P3": 480, "P4": 1440}
ASSIGNMENT_GROUPS = ["Network-Ops", "App-Support", "Desktop-Support", "Email-Support",
                     "Service-Desk", "Security-Ops", "Server-Ops", "DBA-Team"]
ESCALATION_TEAMS = {"Network": "L2-Network-Ops", "Application": "L2-App-Support",
                    "Server": "L2-Server-Ops", "Access": "L2-Security-Ops",
                    "Hardware": "L2-Desktop-Support", "Software": "L2-Desktop-Support",
                    "Email": "L2-Email-Support"}

# -- SHARED STATE -------------------------------------------------------------

class TicketState(TypedDict, total=False):
    # input
    ticket_number: str
    short_description: str
    description: str
    category: str
    priority: str
    sla_due: str
    request_type: str              # C7: e.g. 'Access Grant' for Jira REQ- tickets
    # triage
    triage_category: str
    triage_priority: str
    triage_assignment_group: str
    pii_detected: bool
    # resolution
    kb_article: str
    resolution_text: str
    auto_resolve: bool
    confidence: str
    # sla + HITL checks
    sla_breach_risk: str
    escalation_required: bool
    hitl_required: bool
    hitl_reason: str               # C7: human-readable reason(s) shown at the gate
    hitl_triggers: list            # C7: codes P1_SLA / LOW_CONFIDENCE / ACCESS_GRANT
    # hitl
    hitl_approved: bool
    # communication
    user_message: str
    final_status: str
    # every node appends; operator.add merges the lists instead of overwriting
    audit_log: Annotated[list, operator.add]


def audit(agent: str, action: str, detail: str) -> list:
    """One audit entry, returned as a list so LangGraph appends it to audit_log."""
    print(f"  [AUDIT] {agent}: {action}")
    return [{"timestamp": datetime.now().isoformat(timespec="seconds"),
             "agent": agent, "action": action, "detail": detail}]


def header(title: str) -> None:
    print(f"\n▶ {title}")

# -- HELPERS ------------------------------------------------------------------

def ask_claude(prompt: str, system: str, tools: list | None = None, max_tokens: int = 2048):
    """Single Claude call. Returns the response, or None on API error (callers fall back)."""
    kwargs = {"tools": tools} if tools else {}
    try:
        return client.messages.create(model=MODEL, max_tokens=max_tokens,
                                      output_config={"effort": "low"}, system=system,
                                      messages=[{"role": "user", "content": prompt}], **kwargs)
    except anthropic.APIError as exc:
        print(f"  ! Claude API error: {exc}")
        return None


def text_of(response) -> str:
    if response is None or response.stop_reason not in ("end_turn", "stop_sequence"):
        return ""
    return "\n".join(b.text for b in response.content if b.type == "text").strip()


def is_request(state: TicketState) -> bool:
    """Jira service requests (REQ-...) vs ServiceNow incidents (INC...)."""
    return state["ticket_number"].upper().startswith("REQ-")


def is_access_grant(state: TicketState) -> bool:
    category_is_access = "Access" in (state.get("category"), state.get("triage_category"))
    return category_is_access and (state.get("request_type") or "").lower() == "access grant"


def update_ticket(ticket_number: str, state: str | None = None, group: str | None = None,
                  note: str | None = None) -> str:
    """Write to the right mock system: Jira (REQ-) or ServiceNow (INC).
    Returns 'Jira mock', 'ServiceNow mock' or 'simulated'."""
    try:
        if ticket_number.upper().startswith("REQ-"):
            fields = {k: v for k, v in {"status": state, "assignee": group,
                                        "work_notes": note}.items() if v}
            r = requests.put(f"{JIRA_URL}/{ticket_number}", json={"fields": fields}, timeout=3)
            return "Jira mock" if r.status_code == 200 else "simulated"
        fields = {k: v for k, v in {"state": state, "assignment_group": group,
                                    "work_notes": note}.items() if v}
        r = requests.patch(f"{SNOW_URL}/{ticket_number}", json=fields, timeout=3)
        return "ServiceNow mock" if r.status_code == 200 else "simulated"
    except requests.RequestException:
        return "simulated"


def lookup_request_type(ticket_number: str) -> str:
    """Read issuetype from the Jira mock (Lab C2) for REQ- tickets; '' if unavailable."""
    try:
        r = requests.get(f"{JIRA_URL}/{ticket_number}", timeout=3)
        if r.status_code == 200:
            return r.json()["fields"]["issuetype"]["name"] or ""
    except (requests.RequestException, KeyError, ValueError):
        pass
    return ""


def load_kb():
    """Open the persistent Lab C1 collection (cosine); build it via Lab/kb_setup.py if missing."""
    db = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        kb = db.get_collection("isdo_kb")
        if kb.count():
            return kb
    except Exception:
        pass
    print("KB not found - building it with Lab/kb_setup.py ...")
    spec = importlib.util.spec_from_file_location("kb_setup", PROJECT_ROOT / "Lab" / "kb_setup.py")
    kb_setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kb_setup)
    chunks = [c for name, text in kb_setup.load_articles(KB_DIR).items()
              for c in kb_setup.split_into_chunks(name, text)]
    return kb_setup.build_collection(chunks)


KB = load_kb()

# -- NODE 1: TRIAGE -----------------------------------------------------------

CLASSIFY_TOOL = {
    "name": "classify_ticket",
    "description": "Record the triage classification for the ticket.",
    "input_schema": {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": ["Network", "Application", "Hardware", "Access",
                                                    "Email", "Server", "Software"]},
            "priority": {"type": "string", "enum": ["P1", "P2", "P3", "P4"]},
            "assignment_group": {"type": "string", "enum": ASSIGNMENT_GROUPS},
            "pii_detected": {"type": "boolean"},
            "reasoning": {"type": "string", "description": "One sentence"},
        },
        "required": ["category", "priority", "assignment_group", "pii_detected", "reasoning"],
    },
}

TRIAGE_SYSTEM = """You are the ISDO Triage Agent. Call classify_ticket exactly once.
Priority: P1 = service down / many users / critical system (SAP, Exchange, core network);
P2 = team or department impact, or one user fully blocked; P3 = single user with a
workaround; P4 = service request. Network/VPN -> Network-Ops; business apps -> App-Support;
laptops/printers -> Desktop-Support; email -> Email-Support; passwords/accounts ->
Service-Desk; MFA/security/access grants -> Security-Ops; servers -> Server-Ops;
databases -> DBA-Team. Requests to grant access (including VPN access) are category Access.
pii_detected = true for names, emails, employee IDs, phone numbers or IP addresses."""


def triage_node(state: TicketState) -> dict:
    header(f"TRIAGE AGENT — {state['ticket_number']}")
    update = {}
    if is_request(state) and not state.get("request_type"):          # C7: ask Jira
        update["request_type"] = lookup_request_type(state["ticket_number"])
        if update["request_type"]:
            print(f"  Request type (from Jira mock): {update['request_type']}")

    resp = ask_claude(f"Ticket: {state['ticket_number']}\nSummary: {state['short_description']}\n"
                      f"Details: {state['description']}", TRIAGE_SYSTEM, tools=[CLASSIFY_TOOL])
    result = next((b.input for b in (resp.content if resp else [])
                   if b.type == "tool_use" and b.name == "classify_ticket"), None)
    if result is None:                        # fall back to the ticket's own fields
        result = {"category": state["category"], "priority": state["priority"],
                  "assignment_group": "Service-Desk", "pii_detected": False,
                  "reasoning": "fallback: model did not classify"}
    print(f"  Category: {result['category']:<12} Priority: {result['priority']}")
    print(f"  Assign To: {result['assignment_group']:<11} PII: {result['pii_detected']}")
    return {**update, "triage_category": result["category"], "triage_priority": result["priority"],
            "triage_assignment_group": result["assignment_group"],
            "pii_detected": bool(result["pii_detected"]),
            "audit_log": audit("TriageAgent", "classify_ticket",
                               f"{result['category']}/{result['priority']} -> "
                               f"{result['assignment_group']}: {result['reasoning']}")}

# -- NODE 2: RESOLUTION -------------------------------------------------------

def level_for(score: float) -> str:
    return "HIGH" if score > HIGH_THRESHOLD else "MEDIUM" if score > MEDIUM_THRESHOLD else "LOW"


def resolution_node(state: TicketState) -> dict:
    header("RESOLUTION AGENT — searching KB")
    query = f"{state['short_description']}. {state['description']}"
    raw = KB.query(query_texts=[query], n_results=min(10, KB.count()))
    best = {}
    for meta, dist in zip(raw["metadatas"][0], raw["distances"][0]):
        name = meta.get("article") or Path(meta.get("filename", "unknown")).stem
        best[name] = min(dist, best.get(name, dist))
    article, dist = min(best.items(), key=lambda kv: kv[1])
    score = max(0.0, 1 - dist)
    confidence = level_for(score)
    priority = state.get("triage_priority") or state["priority"]
    text = (KB_DIR / f"{article}.md").read_text(encoding="utf-8")

    # Guardrails in code: HIGH only, never P1, never an access grant (a security
    # approval, not a fix), and the article must allow L1 auto-resolve
    l1_allowed = "not l1 auto-resolvable" not in text.lower()
    auto_resolve = (confidence == "HIGH" and priority != "P1" and l1_allowed
                    and not is_access_grant(state))

    if confidence == "LOW":
        kb_article = "none"
        resolution = "No matching KB article - route to L2 for investigation."
    else:
        kb_article = f"{article}.md"
        resolution = text_of(ask_claude(
            f"Ticket: {state['short_description']}\nDetails: {state['description']}\n\n"
            f"KB article {kb_article}:\n{text}",
            "Write 3-4 numbered resolution steps for this ticket, copied from the KB "
            "article's Resolution Steps. Plain text, no preamble.")) or \
            "See KB article " + kb_article

    print(f"  KB Article: {kb_article}")
    print(f"  Confidence: {confidence} ({score:.0%}) | Auto-resolve: {auto_resolve}")
    return {"kb_article": kb_article, "confidence": confidence, "resolution_text": resolution,
            "auto_resolve": auto_resolve,
            "audit_log": audit("ResolutionAgent", "search_kb",
                               f"{kb_article} score={score:.2f} {confidence}, "
                               f"auto_resolve={auto_resolve}")}

# -- NODE 3: SLA + HITL TRIGGERS ----------------------------------------------

def sla_node(state: TicketState) -> dict:
    header("SLA AGENT — checking deadline and HITL triggers")
    num = state["ticket_number"]
    priority = state.get("triage_priority") or state["priority"]
    target = SLA_TARGET_MIN[priority]
    minutes = int((datetime.strptime(state["sla_due"], "%Y-%m-%d %H:%M:%S")
                   - SIMULATED_NOW).total_seconds() // 60)
    risk = ("BREACHED" if minutes < 0 else "CRITICAL" if minutes <= target * 0.2
            else "AT_RISK" if minutes <= target * 0.5 else "ON_TRACK")
    escalation_required = risk in ("CRITICAL", "BREACHED") and priority in ("P1", "P2")
    print(f"  SLA Risk: {risk} | Minutes remaining: {minutes} of {target}")
    entries = audit("SLAAgent", "get_sla_status",
                    f"{priority} {risk}, {minutes} min left, escalation_required={escalation_required}")

    # C7: evaluate every HITL trigger; reasons accumulate
    triggers, reasons = [], []
    if escalation_required and priority == "P1":
        triggers.append("P1_SLA")
        reasons.append(f"P1 SLA {risk} - escalation needs human sign-off")
    if state.get("confidence") == "LOW":
        triggers.append("LOW_CONFIDENCE")
        reasons.append("LOW KB CONFIDENCE - no clear fix in the knowledge base, L2 must review")
    if is_access_grant(state):
        triggers.append("ACCESS_GRANT")
        reasons.append(f"ACCESS GRANT - {state['short_description']} requires security approval")
    hitl_required = bool(triggers)
    hitl_reason = "; ".join(reasons)

    if escalation_required and priority != "P1":          # P2: escalate without a human
        team = ESCALATION_TEAMS.get(state.get("triage_category", ""), "L2-Service-Desk")
        src = update_ticket(num, "Escalated", team, f"SLA {risk} - auto-escalated to {team}")
        print(f"  [{src}] ESCALATED {num} -> {team}")
        entries += audit("SLAAgent", "update_ticket", f"escalated to {team} ({src})")

    if hitl_required:
        print(f"  HITL required -> {', '.join(triggers)}")
    entries += audit("SLAAgent", "hitl_check",
                     f"hitl_required={hitl_required}" + (f" [{', '.join(triggers)}]" if triggers else ""))
    return {"sla_breach_risk": risk, "escalation_required": escalation_required,
            "hitl_required": hitl_required, "hitl_reason": hitl_reason,
            "hitl_triggers": triggers, "audit_log": entries}


def route_after_sla(state: TicketState) -> str:
    return "hitl" if state.get("hitl_required") else "communication"

# -- NODE 4: HITL -------------------------------------------------------------

def proposed_actions(state: TicketState) -> list[str]:
    team = ESCALATION_TEAMS.get(state.get("triage_category", ""), "L2-Service-Desk")
    actions = []
    for t in state.get("hitl_triggers", []):
        if t == "P1_SLA":
            actions.append(f"Escalate to {team}")
        elif t == "LOW_CONFIDENCE" and "P1_SLA" not in state.get("hitl_triggers", []):
            actions.append(f"Route to {team} for manual investigation")
        elif t == "ACCESS_GRANT":
            actions.append("Grant the requested access")
    return actions


def hitl_node(state: TicketState) -> dict:
    header("HITL GATE — human approval required")
    num = state["ticket_number"]
    team = ESCALATION_TEAMS.get(state.get("triage_category", ""), "L2-Service-Desk")
    actions = proposed_actions(state)
    bar = "  " + "WARNING " * 8
    print(bar)
    print(f"  Ticket:  {num} | Priority: {state.get('triage_priority')} | "
          f"SLA: {state.get('sla_breach_risk')}")
    print(f"  Issue:   {state['short_description']}")
    for i, reason in enumerate(state.get("hitl_reason", "").split("; ")):
        print(f"  {'Reason:' if i == 0 else '':<8} {reason}")
    print(f"  Action:  {' + '.join(actions)}")
    print(bar)
    try:
        answer = input("  Approve action? [y/n]: ").strip().lower()
    except EOFError:                        # no human available -> never auto-approve
        answer = ""
    approved = answer in ("y", "yes")
    decision = "APPROVED" if approved else "REJECTED"

    HITL_LOG.parent.mkdir(exist_ok=True)
    with open(HITL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"logged_at": datetime.now().isoformat(timespec="seconds"),
                            "ticket": num, "triggers": state.get("hitl_triggers", []),
                            "reason": state.get("hitl_reason", ""), "actions": actions,
                            "decision": decision, "source": "C7 orchestrator"}) + "\n")
    entries = audit("HITLGate", "approval_decision",
                    f"{decision} - {state.get('hitl_reason', '')} -> {' + '.join(actions)}")
    print(f"  Decision: {decision}")

    triggers = state.get("hitl_triggers", [])
    if approved:
        if "ACCESS_GRANT" in triggers:
            src = update_ticket(num, "Approved", "security-team", "Access grant approved by human")
            print(f"  [{src}] ACCESS APPROVED {num}")
            entries += audit("HITLGate", "update_ticket", f"access grant approved ({src})")
        if "P1_SLA" in triggers or "LOW_CONFIDENCE" in triggers:
            src = update_ticket(num, "Escalated", team, f"Human approved: {' + '.join(actions)}")
            print(f"  [{src}] ESCALATED {num} -> {team}")
            entries += audit("HITLGate", "update_ticket", f"escalated to {team} ({src})")
    else:
        src = update_ticket(num, "Pending Approval", None, f"HITL rejected: {state.get('hitl_reason', '')}")
        entries += audit("HITLGate", "update_ticket", f"marked Pending Approval ({src})")
    return {"hitl_approved": approved, "audit_log": entries}

# -- NODE 5: COMMUNICATION ----------------------------------------------------

def communication_node(state: TicketState) -> dict:
    header("COMMUNICATION AGENT")
    num, team = state["ticket_number"], state.get("triage_assignment_group", "the support team")
    triggers = state.get("hitl_triggers", [])
    l2_team = ESCALATION_TEAMS.get(state.get("triage_category", ""), "L2")

    if state.get("hitl_required") and not state.get("hitl_approved"):
        status = "PENDING_APPROVAL"
        brief = ("Pending approval: the requested action needs further approval before we can "
                 "proceed. The ticket stays open and the user will be updated. Do not promise a "
                 "fix or a date.")
    elif state.get("hitl_approved") and "ACCESS_GRANT" in triggers:
        status = "APPROVED"
        brief = ("The access grant request has been approved by the security approver and "
                 "access is being provisioned.")
    elif state.get("hitl_approved"):
        status = "ESCALATED"
        brief = (f"The ticket was reviewed and escalated to {l2_team} (priority "
                 f"{state.get('triage_priority')}) for specialist investigation.")
    elif state.get("auto_resolve"):
        status = "RESOLVED"
        brief = f"Self-service resolution. Include these steps exactly:\n{state.get('resolution_text', '')}"
    elif state.get("escalation_required"):
        status = "ESCALATED"
        brief = "The ticket breached its SLA and was escalated to an L2 team."
    else:
        status = "ASSIGNED"
        brief = f"Assignment notification: the ticket is assigned to {team}."

    greeting = "Dear Requester," if is_request(state) else "Dear User,"
    message = text_of(ask_claude(
        f"Ticket {num}: {state['short_description']}\n{brief}",
        f"You are the ISDO Communication Agent. Write a short, polite email body (max 120 "
        f"words) to the requester, starting '{greeting}'. Mention the ticket number. Never "
        f"include personal data such as names, emails or employee IDs.", max_tokens=1024)) \
        or f"{greeting} regarding {num}: status is {status}. {brief}"

    new_state = {"RESOLVED": "Resolved", "ESCALATED": "Escalated", "APPROVED": "Approved",
                 "PENDING_APPROVAL": "Pending Approval"}.get(status, "In Progress")
    update_ticket(num, new_state, None, f"Communication sent - final status {status}")
    print("  USER MESSAGE:")
    for line in message.splitlines():
        print(f"    {line}")
    print(f"\n✅ FINAL STATUS: {status}")
    return {"user_message": message, "final_status": status,
            "audit_log": audit("CommunicationAgent", "draft_user_message", f"final_status={status}")}

# -- BUILD THE GRAPH ----------------------------------------------------------

def build_graph():
    g = StateGraph(TicketState)
    g.add_node("triage", triage_node)
    g.add_node("resolution", resolution_node)
    g.add_node("sla", sla_node)
    g.add_node("hitl", hitl_node)
    g.add_node("communication", communication_node)

    g.add_edge(START, "triage")
    g.add_edge("triage", "resolution")
    g.add_edge("resolution", "sla")
    g.add_conditional_edges("sla", route_after_sla, {"hitl": "hitl", "communication": "communication"})
    g.add_edge("hitl", "communication")
    g.add_edge("communication", END)
    return g.compile()


graph = build_graph()


def process_ticket(ticket: dict) -> TicketState:
    print(f"\n{'═' * 55}\nPROCESSING TICKET: {ticket['ticket_number']}\n{'═' * 55}")
    return graph.invoke({**ticket, "audit_log": []})

# -- RUN ----------------------------------------------------------------------

TEST_TICKETS = [
    # C6: P2 VPN, AT_RISK, HIGH confidence -> auto-resolved, no HITL
    {"ticket_number": "INC0001001", "short_description": "VPN not connecting after password change",
     "description": "User reports VPN client fails to connect after AD password was reset. "
                    "Error: authentication failed.",
     "category": "Network", "priority": "P2", "sla_due": "2024-01-15 12:00:00"},
    # C6/C7 Step 2: P1 SAP, CRITICAL -> HITL trigger P1_SLA (run with y, then n)
    {"ticket_number": "INC0001002", "short_description": "Cannot access ERP system - login error",
     "description": "Multiple users in Finance unable to login to SAP. Error code: DBCON_FAIL. "
                    "Started 09:00 today.",
     "category": "Application", "priority": "P1", "sla_due": "2024-01-15 10:40:00"},
    # C7 Step 3: no KB article covers Webex -> HITL trigger LOW_CONFIDENCE (even for P3)
    {"ticket_number": "INC0001099", "short_description": "Cisco Webex not launching on MacBook M2 after Sonoma update",
     "description": "Cisco Webex not launching on MacBook M2 after Sonoma update.",
     "category": "Software", "priority": "P3", "sla_due": "2024-01-15 18:00:00"},
    # C7 Step 4: access grant request -> HITL trigger ACCESS_GRANT (any priority)
    {"ticket_number": "REQ-1002", "short_description": "VPN access for new contractor",
     "description": "Contractor needs VPN access. Email: contractor@client.com",
     "category": "Access", "request_type": "Access Grant", "priority": "P2",
     "sla_due": "2024-01-15 15:00:00"},
]

if __name__ == "__main__":
    wanted = {a.upper() for a in sys.argv[1:]}
    tickets = [t for t in TEST_TICKETS if not wanted or t["ticket_number"].upper() in wanted]
    if not tickets:
        sys.exit(f"No test ticket matches {sorted(wanted)}. "
                 f"Choose from: {', '.join(t['ticket_number'] for t in TEST_TICKETS)}")

    results = [process_ticket(t) for t in tickets]

    for r in results:
        path = ["triage", "resolution", "sla"] + (["hitl"] if r.get("hitl_required") else []) + ["communication"]
        print(f"\n{'═' * 55}\nAUDIT LOG: {r['ticket_number']}  ->  {r['final_status']}")
        print(f"Path: {' → '.join(path)}")
        if r.get("hitl_required"):
            print(f"HITL: {'APPROVED' if r.get('hitl_approved') else 'REJECTED'} - {r.get('hitl_reason')}")
        print("═" * 55)
        for e in r["audit_log"]:
            print(f"  {e['timestamp']}  {e['agent']:<19} {e['action']:<19} {e['detail']}")

    print(f"\n{'═' * 55}\nSUMMARY\n{'═' * 55}")
    print(f"  {'Ticket':<12}{'Pri':<5}{'Confidence':<12}{'HITL triggers':<30}Final status")
    for r in results:
        print(f"  {r['ticket_number']:<12}{r.get('triage_priority', ''):<5}{r.get('confidence', ''):<12}"
              f"{', '.join(r.get('hitl_triggers', [])) or '-':<30}{r['final_status']}")
    print(f"\n  HITL decisions also written to: {HITL_LOG}")
