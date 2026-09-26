"""
test_c2_shims.py - ISDO Lab C2: verify the ServiceNow (5001) and Jira (5002) mock APIs.

Start both shims first, each in its own terminal (from the project folder):
    python mcp_server\\snow_shim.py
    python mcp_server\\jira_shim.py

Then, in a third terminal:
    python Lab\\test_c2_shims.py
"""

import json
import sys

import requests

SNOW = "http://localhost:5001"
JIRA = "http://localhost:5002"

results = []  # (name, passed)


def check(name, method, url, expect_status=200, body=None, verify=None, show=True):
    """Call an endpoint, print a short preview, and record PASS/FAIL."""
    print(f"\n--- Testing {method} {url.replace(SNOW, '').replace(JIRA, '')} ---")
    try:
        r = requests.request(method, url, json=body, timeout=5)
    except requests.ConnectionError:
        print(f"    FAIL: cannot connect - is the shim on {url.split('/')[2]} running?")
        results.append((name, False))
        return None
    data = r.json()
    if show:
        preview = json.dumps(data)
        print("    " + (preview[:220] + " ..." if len(preview) > 220 else preview))
    ok = r.status_code == expect_status and (verify is None or verify(data))
    print(f"    {'PASS' if ok else 'FAIL'} (HTTP {r.status_code}) - {name}")
    results.append((name, ok))
    return data


def main():
    print("=" * 70)
    print("ServiceNow Mock  (port 5001)")
    print("=" * 70)
    # Step 6 - health
    check("SNOW health", "GET", f"{SNOW}/health",
          verify=lambda d: d["status"] == "ok" and d["incidents_loaded"] == 15)
    # Step 3 - all incidents
    check("List all 15 incidents", "GET", f"{SNOW}/api/now/table/incident",
          verify=lambda d: d["total"] == 15)
    # Step 4 - filters and single record
    check("Filter priority=P1 (expect 4)", "GET", f"{SNOW}/api/now/table/incident?priority=P1",
          verify=lambda d: d["total"] == 4 and all(i["priority"] == "P1" for i in d["result"]))
    check("Filter category=Network (expect 3)", "GET", f"{SNOW}/api/now/table/incident?category=Network",
          verify=lambda d: d["total"] == 3)
    check("Get INC0001001", "GET", f"{SNOW}/api/now/table/incident/INC0001001",
          verify=lambda d: d["result"]["number"] == "INC0001001")
    check("Unknown incident -> 404", "GET", f"{SNOW}/api/now/table/incident/INC9999999",
          expect_status=404)
    # PATCH + read back (used by the SLA Agent in Lab C5)
    check("PATCH INC0001008 state=Escalated", "PATCH", f"{SNOW}/api/now/table/incident/INC0001008",
          body={"state": "Escalated", "work_notes": "Escalated by C2 test"},
          verify=lambda d: d["result"]["state"] == "Escalated")
    check("Read back INC0001008", "GET", f"{SNOW}/api/now/table/incident/INC0001008",
          verify=lambda d: d["result"]["state"] == "Escalated")

    print("\n" + "=" * 70)
    print("Jira Mock  (port 5002)")
    print("=" * 70)
    check("Jira health", "GET", f"{JIRA}/health",
          verify=lambda d: d["status"] == "ok" and d["requests_loaded"] >= 10)
    # Step 5 - all requests
    check("List all service requests", "GET", f"{JIRA}/rest/agile/1.0/board/requests",
          verify=lambda d: d["total"] >= 10)
    check("Filter request_type=Access Grant (expect 2)", "GET",
          f"{JIRA}/rest/agile/1.0/board/requests?request_type=Access+Grant",
          verify=lambda d: d["total"] == 2)
    check("Get REQ-1002 (nested 'fields')", "GET", f"{JIRA}/rest/api/2/issue/REQ-1002",
          verify=lambda d: d["fields"]["priority"]["name"] == "High")
    check("PUT REQ-1002 status=In Progress", "PUT", f"{JIRA}/rest/api/2/issue/REQ-1002",
          body={"fields": {"status": "In Progress"}})
    check("Read back REQ-1002", "GET", f"{JIRA}/rest/api/2/issue/REQ-1002",
          verify=lambda d: d["fields"]["status"]["name"] == "In Progress")
    check("Create new request (POST)", "POST", f"{JIRA}/rest/api/2/issue", expect_status=201,
          body={"fields": {"summary": "C2 test request", "issuetype": {"name": "Access"},
                           "priority": {"name": "Low"}}},
          verify=lambda d: d["key"].startswith("REQ-"))

    passed = sum(ok for _, ok in results)
    print("\n" + "=" * 70)
    print(f"RESULT: {passed}/{len(results)} checks passed")
    for name, ok in results:
        if not ok:
            print(f"  FAILED: {name}")
    print("=" * 70)
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
