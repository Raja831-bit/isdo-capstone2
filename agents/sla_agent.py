"""
ISDO Lab C5 - SLA & Escalation Agent
Checks SLA breach risk, escalates CRITICAL/BREACHED P1/P2 tickets, and pauses at a
Human-in-the-Loop (HITL) gate before ANY P1 escalation.

Tools
  get_sla_status - ticket_number, sla_due, priority -> minutes_remaining, breach_risk
                   (BREACHED / CRITICAL / AT_RISK / ON_TRACK), requires_escalation
  update_ticket  - escalate / add_note / update_state. PATCHes the ServiceNow mock
                   (Lab C2, port 5001); falls back to a simulated update if it is down.

SLA targets: P1=60, P2=240, P3=480, P4=1440 minutes. Simulated now = 2024-01-15 10:30.
Risk: BREACHED (past due) | CRITICAL (<=20% of target left) | AT_RISK (<=50% left) | ON_TRACK
Escalation policy (enforced in code): only CRITICAL/BREACHED tickets with P1/P2.
HITL gate (enforced in code): every P1 escalation needs a human 'y'. Decisions are
logged to logs/hitl_decisions.jsonl.

Run from the project folder (ServiceNow shim running):
    python agents/sla_agent.py

Temperature is not set: claude-opus-5 and later reject non-default sampling
parameters with a 400 error (see Lab C3 notes).
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import anthropic
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SNOW_URL = "http://localhost:5001/api/now/table/incident"
AUDIT_LOG = PROJECT_ROOT / "logs" / "hitl_decisions.jsonl"

SIMULATED_NOW = datetime(2024, 1, 15, 10, 30)   # fixed 'now' for reproducible demos
SLA_TARGET_MIN = {"P1": 60, "P2": 240, "P3": 480, "P4": 1440}
ESCALATE_RISKS = {"BREACHED", "CRITICAL"}
ESCALATE_PRIORITIES = {"P1", "P2"}
HITL_PRIORITIES = {"P1"}

ESCALATION_TEAMS = {
    "Network": "L2-Network-Ops", "Application": "L2-App-Support", "Server": "L2-Server-Ops",
    "Access": "L2-Security-Ops", "Hardware": "L2-Desktop-Support",
    "Software": "L2-Desktop-Support", "Email": "L2-Email-Support",
}

MAX_TOKENS = 2048
MAX_TURNS = 6       # status -> escalate -> note -> confirm, plus spare

load_dotenv(PROJECT_ROOT / ".env")
if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit("ANTHROPIC_API_KEY is not set. Add it to the .env file in the project folder.")
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5")
client = anthropic.Anthropic()

# -- TOOL DEFINITIONS ---------------------------------------------------------

tools = [
    {
        "name": "get_sla_status",
        "description": "Check a ticket's SLA: minutes remaining, breach risk level and "
                       "whether escalation is required by policy.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_number": {"type": "string"},
                "sla_due": {"type": "string", "description": "YYYY-MM-DD HH:MM:SS"},
                "priority": {"type": "string", "enum": ["P1", "P2", "P3", "P4"]},
            },
            "required": ["ticket_number", "sla_due", "priority"],
        },
    },
    {
        "name": "update_ticket",
        "description": "Update the ticket in ServiceNow: escalate it, add a work note, "
                       "or change its state.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_number": {"type": "string"},
                "action": {"type": "string", "enum": ["escalate", "add_note", "update_state"]},
                "escalation_team": {"type": "string", "enum": sorted(set(ESCALATION_TEAMS.values()))},
                "note": {"type": "string", "description": "Work note text"},
                "new_state": {"type": "string", "enum": ["In Progress", "On Hold", "Resolved"]},
            },
            "required": ["ticket_number", "action"],
        },
    },
]

# -- TOOL IMPLEMENTATIONS -----------------------------------------------------

def get_sla_status(ticket_number: str, sla_due: str, priority: str) -> dict:
    """Minutes remaining vs the priority's SLA target -> breach risk."""
    try:
        due = datetime.strptime(sla_due, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return {"error": f"Invalid sla_due '{sla_due}', expected YYYY-MM-DD HH:MM:SS"}
    if priority not in SLA_TARGET_MIN:
        return {"error": f"Unknown priority '{priority}'"}

    target = SLA_TARGET_MIN[priority]
    minutes = int((due - SIMULATED_NOW).total_seconds() // 60)
    if minutes < 0:
        risk, msg = "BREACHED", f"SLA breached {abs(minutes)} minutes ago"
    elif minutes <= target * 0.2:
        risk, msg = "CRITICAL", f"Only {minutes} minutes remaining - breach imminent"
    elif minutes <= target * 0.5:
        risk, msg = "AT_RISK", f"{minutes} minutes remaining - at risk"
    else:
        risk, msg = "ON_TRACK", f"{minutes} minutes remaining - on track"

    return {"ticket_number": ticket_number, "priority": priority, "sla_due": sla_due,
            "sla_target_minutes": target, "minutes_remaining": minutes,
            "percent_remaining": round(max(minutes, 0) / target * 100),
            "breach_risk": risk, "status_message": msg,
            "requires_escalation": risk in ESCALATE_RISKS and priority in ESCALATE_PRIORITIES}


def update_ticket(ticket_number: str, action: str, escalation_team: str | None = None,
                  note: str | None = None, new_state: str | None = None) -> dict:
    """PATCH the ServiceNow mock; simulate the update if the mock is not running."""
    stamp = SIMULATED_NOW.strftime("%Y-%m-%d %H:%M")
    if action == "escalate":
        team = escalation_team or "L2-Service-Desk"
        fields = {"state": "Escalated", "assignment_group": team,
                  "work_notes": f"[{stamp}] SLA agent escalated to {team}. {note or ''}".strip()}
        label = f"ESCALATED {ticket_number} -> {team}"
    elif action == "add_note":
        fields = {"work_notes": f"[{stamp}] {note or ''}"}
        label = f"NOTE ADDED to {ticket_number}: {(note or '')[:60]}"
    elif action == "update_state":
        fields = {"state": new_state or "In Progress"}
        label = f"STATE CHANGED {ticket_number} -> {fields['state']}"
    else:
        return {"success": False, "error": f"Unknown action '{action}'"}

    try:
        resp = requests.patch(f"{SNOW_URL}/{ticket_number}", json=fields, timeout=3)
        if resp.status_code == 404:          # ticket not in the mock -> simulate
            raise requests.RequestException("not found in mock")
        resp.raise_for_status()
        source = "ServiceNow mock"
    except requests.RequestException:
        source = "simulated"
    print(f"  [{source}] {label}")
    return {"success": True, "ticket_number": ticket_number, "action": action,
            "fields": fields, "source": source, "timestamp": stamp}

# -- HITL GATE ----------------------------------------------------------------

def hitl_approve(ticket_number: str, action: str, detail: str) -> bool:
    """Ask a human before a P1 escalation. No answer (non-interactive) = rejected."""
    print(f"\n  {'!!! ' * 10}\n  HITL APPROVAL REQUIRED")
    print(f"  Ticket:  {ticket_number}\n  Action:  {action}\n  Detail:  {detail}")
    print(f"  {'!!! ' * 10}")
    try:
        answer = input("  Approve escalation? [y/n]: ").strip().lower()
    except EOFError:
        answer = ""
    approved = answer in ("y", "yes")
    print(f"  Decision: {'APPROVED' if approved else 'REJECTED'}")

    AUDIT_LOG.parent.mkdir(exist_ok=True)
    with open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"logged_at": datetime.now().isoformat(timespec="seconds"),
                            "ticket": ticket_number, "action": action, "detail": detail,
                            "decision": "APPROVED" if approved else "REJECTED"}) + "\n")
    return approved

# -- SLA AGENT ----------------------------------------------------------------

SYSTEM_PROMPT = f"""You are the ISDO SLA & Escalation Agent for Zensar's IT Service Desk.

For each ticket:
1. Call get_sla_status once.
2. If requires_escalation is true, call update_ticket with action "escalate" and the
   team for the ticket's category:
   {json.dumps(ESCALATION_TEAMS)} (anything else: L2-Service-Desk).
   Put a one-line reason in the note.
3. If escalation is not required, do NOT escalate. For AT_RISK tickets add a short
   work note (action "add_note") saying the SLA is being watched. For ON_TRACK
   tickets take no action.
4. If an escalation is rejected by the human approver, do not retry it; add a work
   note recording that escalation was declined and the ticket stays with its group.
5. Finish with one short summary line."""


def monitor_ticket(ticket_number: str, short_description: str, category: str,
                   priority: str, sla_due: str) -> dict:
    """Run the agentic loop on one ticket. Returns the SLA outcome (used by C6)."""
    print(f"\n{'=' * 55}\nSLA Check: {ticket_number} | {priority} | Category: {category}")
    print(f"{'=' * 55}\nIssue: {short_description}  (SLA due {sla_due})")

    status = get_sla_status(ticket_number, sla_due, priority)   # ground truth for guardrails
    outcome = {"ticket_number": ticket_number, "priority": priority,
               "breach_risk": status.get("breach_risk"),
               "minutes_remaining": status.get("minutes_remaining"),
               "escalated": False, "hitl_decision": None, "actions": []}

    messages = [{"role": "user", "content":
                 f"Monitor SLA for this ticket and escalate if needed:\n\n"
                 f"Ticket: {ticket_number}\nDescription: {short_description}\n"
                 f"Category: {category}\nPriority: {priority}\nSLA Due: {sla_due}"}]

    for _ in range(MAX_TURNS):
        response = client.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS, output_config={"effort": "medium"},
            system=SYSTEM_PROMPT, tools=tools, messages=messages)

        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    print(f"  Agent: {block.text.strip()}")
            break
        if response.stop_reason != "tool_use":
            print(f"  ! Stopped early: stop_reason={response.stop_reason}")
            break

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            args = {**block.input, "ticket_number": ticket_number}   # never touch another ticket

            if block.name == "get_sla_status":
                result = get_sla_status(ticket_number, sla_due, priority)   # real ticket data
                print(f"  -> Risk Level: {result.get('breach_risk')}")
                print(f"  -> Status:     {result.get('status_message')}")

            elif block.name == "update_ticket":
                action = args.get("action")
                if action == "escalate" and not status.get("requires_escalation"):
                    result = {"success": False, "error": f"Policy: escalation not allowed for "
                              f"{priority} {status.get('breach_risk')} (needs CRITICAL/BREACHED + P1/P2)"}
                    print(f"  ! Blocked by policy: {result['error']}")
                elif (action == "escalate" and priority in HITL_PRIORITIES and
                      not hitl_approve(ticket_number, "Escalate ticket",
                                       f"Escalate to {args.get('escalation_team', 'L2 team')}")):
                    outcome["hitl_decision"] = "REJECTED"
                    result = {"success": False, "message": "Escalation rejected by human approver"}
                    print("  Escalation cancelled - decision logged.")
                else:
                    if action == "escalate" and priority in HITL_PRIORITIES:
                        outcome["hitl_decision"] = "APPROVED"
                    result = update_ticket(ticket_number, action, args.get("escalation_team"),
                                           args.get("note"), args.get("new_state"))
                    outcome["escalated"] |= action == "escalate"
                    outcome["actions"].append(action)
            else:
                result = {"error": f"Unknown tool: {block.name}"}

            tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                                 "content": json.dumps(result)})
        messages.append({"role": "user", "content": tool_results})
    else:
        print(f"  ! Loop stopped after {MAX_TURNS} turns")

    if status.get("requires_escalation") and not outcome["escalated"] and outcome["hitl_decision"] != "REJECTED":
        print("  ! WARNING: policy required escalation but none was made - flag for review")
    return outcome

# -- RUN SLA MONITORING -------------------------------------------------------

if __name__ == "__main__":
    # One ticket per SLA state (simulated now = 2024-01-15 10:30).
    # Demo deadlines are chosen so each state appears once - they differ from incidents.csv.
    test_tickets = [   # (number, description, category, priority, sla_due)
        # P1, 10 of 60 min left (17%) -> CRITICAL -> HITL prompt: type y
        ("INC0001002", "Cannot access ERP - SAP login failure", "Application", "P1",
         "2024-01-15 10:40:00"),
        # P1, due 09:30 -> BREACHED -> HITL prompt: type n
        ("INC0001010", "Exchange server high CPU alert", "Server", "P1",
         "2024-01-15 09:30:00"),
        # P2, 90 of 240 min left (38%) -> AT_RISK -> work note only, no escalation
        # Step 5: change to "2024-01-15 10:00:00" -> BREACHED -> auto-escalated (no HITL for P2)
        ("INC0001001", "VPN not connecting after password change", "Network", "P2",
         "2024-01-15 12:00:00"),
        # P3, due in 2 days -> ON_TRACK -> monitored only
        ("INC0001003", "Laptop running very slowly", "Hardware", "P3",
         "2024-01-17 09:00:00"),
    ]

    results = [monitor_ticket(*t) for t in test_tickets]

    print(f"\n{'=' * 55}\nSLA SUMMARY  (now = {SIMULATED_NOW:%Y-%m-%d %H:%M})\n{'=' * 55}")
    print(f"  {'Ticket':<12}{'Pri':<5}{'Risk':<10}{'Min left':>9}  {'Escalated':<10}HITL")
    for r in results:
        print(f"  {r['ticket_number']:<12}{r['priority']:<5}{r['breach_risk']:<10}"
              f"{r['minutes_remaining']:>9}  {str(r['escalated']):<10}{r['hitl_decision'] or '-'}")
    print(f"\n  HITL decisions logged to: {AUDIT_LOG}")
