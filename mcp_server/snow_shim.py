"""
ISDO Lab C2 - Mock ServiceNow Table API (Flask), port 5001.

  GET   /api/now/table/incident            all incidents (?category= ?priority= ?state= ?assignment_group=)
  GET   /api/now/table/incident/<number>   one incident
  PATCH /api/now/table/incident/<number>   update fields in memory (e.g. state, work_notes)
  POST  /api/now/table/incident            create an incident (needs "number")
  GET   /health                            service status

Run from the project folder:  python mcp_server/snow_shim.py
"""
import csv
from pathlib import Path

from flask import Flask, jsonify, request

PORT = 5001
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "incidents.csv"
FILTERS = ["category", "priority", "state", "assignment_group"]

app = Flask(__name__)


def load_incidents() -> dict:
    """Load incidents.csv into an in-memory dict keyed by incident number."""
    if not DATA_FILE.exists():
        print(f"Warning: {DATA_FILE} not found - starting empty.")
        return {}
    with open(DATA_FILE, newline="", encoding="utf-8") as f:
        # drop stray None keys caused by unquoted commas in a row
        return {r["number"]: {k: v for k, v in r.items() if k} for r in csv.DictReader(f)}


INCIDENTS = load_incidents()


def not_found(number):
    return jsonify({"error": f"Incident {number} not found"}), 404


@app.get("/api/now/table/incident")
def list_incidents():
    rows = list(INCIDENTS.values())
    for field in FILTERS:
        val = request.args.get(field)
        if val:
            rows = [r for r in rows if r.get(field, "").lower() == val.lower()]
    return jsonify({"result": rows, "total": len(rows)})  # ServiceNow-style envelope


@app.get("/api/now/table/incident/<number>")
def get_incident(number):
    inc = INCIDENTS.get(number)
    return jsonify({"result": inc}) if inc else not_found(number)


@app.patch("/api/now/table/incident/<number>")
def update_incident(number):
    if number not in INCIDENTS:
        return not_found(number)
    updates = request.get_json(silent=True)
    if not updates:
        return jsonify({"error": "No update body provided"}), 400
    updates.pop("number", None)  # the record key cannot be changed
    INCIDENTS[number].update(updates)
    print(f"[ServiceNow Mock] Updated {number}: {updates}")
    return jsonify({"result": INCIDENTS[number], "message": "Updated successfully"})


@app.post("/api/now/table/incident")
def create_incident():
    data = request.get_json(silent=True)
    if not data or "number" not in data:
        return jsonify({"error": "Missing required field: number"}), 400
    INCIDENTS[data["number"]] = data
    print(f"[ServiceNow Mock] Created incident: {data['number']}")
    return jsonify({"result": data, "message": "Incident created"}), 201


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "ServiceNow Mock", "incidents_loaded": len(INCIDENTS)})


if __name__ == "__main__":
    print(f"ServiceNow Mock API starting on http://localhost:{PORT}")
    print(f"Loaded {len(INCIDENTS)} incidents from {DATA_FILE}")
    app.run(port=PORT, debug=True)
