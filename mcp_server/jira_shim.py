"""
ISDO Lab C2 - Mock Jira Service Management REST API (Flask), port 5002.
  GET  /rest/agile/1.0/board/requests   all requests (?request_type= ?priority= ?assignee= ?status=)
  GET  /rest/api/2/issue                same list and filters
  GET  /rest/api/2/issue/<key>          one request, Jira-style nested "fields"
  PUT  /rest/api/2/issue/<key>          update a request ({"fields": {...}} or flat)
  POST /rest/api/2/issue                create a request ({"fields": {"summary": ...}})
  GET  /health                          service status
Run from the project folder:  python mcp_server/jira_shim.py
"""
import csv
from pathlib import Path

from flask import Flask, jsonify, request

PORT = 5002
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "requests.csv"
FILTERS = ["request_type", "priority", "assignee", "status"]

app = Flask(__name__)


def load_requests() -> dict:
    """Load requests.csv into an in-memory dict keyed by request key."""
    if not DATA_FILE.exists():
        print(f"Warning: {DATA_FILE} not found - starting empty.")
        return {}
    with open(DATA_FILE, newline="", encoding="utf-8") as f:
        return {r["key"]: {k: v for k, v in r.items() if k} for r in csv.DictReader(f)}


REQUESTS = load_requests()

def not_found(key):
    return jsonify({"errorMessages": [f"Issue {key} does not exist"]}), 404


@app.get("/rest/agile/1.0/board/requests")
@app.get("/rest/api/2/issue")
def list_requests():
    rows = list(REQUESTS.values())
    for field in FILTERS:
        val = request.args.get(field)  # Flask already decodes "Access+Grant" -> "Access Grant"
        if val:
            rows = [r for r in rows if r.get(field, "").lower() == val.lower()]
    return jsonify({"issues": rows, "total": len(rows)})


@app.get("/rest/api/2/issue/<key>")
def get_request(key):
    r = REQUESTS.get(key)
    if not r:
        return not_found(key)
    return jsonify({"key": key, "fields": {           # Jira's nested structure
        "summary": r.get("summary"),
        "issuetype": {"name": r.get("request_type")},
        "priority": {"name": r.get("priority")},
        "status": {"name": r.get("status")},
        "assignee": {"displayName": r.get("assignee")},
        "customfield_sla": r.get("sla"),
    }})


@app.put("/rest/api/2/issue/<key>")
def update_request(key):
    if key not in REQUESTS:
        return not_found(key)
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"errorMessages": ["No update body provided"]}), 400
    fields = data.get("fields", data)
    fields.pop("key", None)
    REQUESTS[key].update(fields)
    print(f"[Jira Mock] Updated {key}: {fields}")
    return jsonify({"key": key, "message": "Updated successfully"})


@app.post("/rest/api/2/issue")
def create_request():
    fields = (request.get_json(silent=True) or {}).get("fields", {})
    if not fields.get("summary"):
        return jsonify({"errorMessages": ["Missing required field: fields.summary"]}), 400
    key = f"REQ-{max([int(k.split('-')[1]) for k in REQUESTS] or [1000]) + 1}"
    REQUESTS[key] = {"key": key, "summary": fields["summary"], "assignee": "", "sla": "", "status": "Open",
                     "request_type": fields.get("issuetype", {}).get("name", ""),
                     "priority": fields.get("priority", {}).get("name", "Medium")}
    print(f"[Jira Mock] Created request: {key}")
    return jsonify({"key": key, "message": "Request created"}), 201


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "Jira Mock", "requests_loaded": len(REQUESTS)})


if __name__ == "__main__":
    print(f"Jira Mock API starting on http://localhost:{PORT}")
    print(f"Loaded {len(REQUESTS)} requests from {DATA_FILE}")
    app.run(port=PORT, debug=True)
