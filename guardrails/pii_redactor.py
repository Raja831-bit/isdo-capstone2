"""
ISDO Lab C9 - PII Redaction Middleware + Audit Trail
Masks PII before any ticket data is sent to Claude.

Detection layers (all run on every call):
  1. Regex for structured PII  - email, IP address, employee ID, phone, usernames
     (DOMAIN\\user, "username: x", "login id x", first.last handles)
  2. spaCy NER (PERSON)       - only if the en_core_web_sm model is installed
  3. Rule-based name finder   - works WITHOUT spaCy: names after cue words
     ("User John Smith", "reset for Michael D'Souza", "Dear Priya") and names
     that start with a known first name ("Rahul Verma reports ...")
  4. Propagation              - once "John Smith" is found, a later bare "Smith"
     is masked too

Usage:
    from guardrails.pii_redactor import redact, restore, AuditLogger

    clean_text, mapping = redact(raw_text)
    # ... send clean_text to Claude ...
    original_text = restore(claude_response, mapping)

Install the spaCy model once (inside labenv) for the best name detection:
    python -m spacy download en_core_web_sm
"""

import json
import re
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# -- spaCy (optional) ---------------------------------------------------------

nlp = None
for _model in ("en_core_web_sm", "en_core_web_md", "en_core_web_lg"):
    try:
        import spacy
        nlp = spacy.load(_model)
        break
    except (ImportError, OSError):
        continue
SPACY_AVAILABLE = nlp is not None
if not SPACY_AVAILABLE:
    print("NOTE: spaCy model not found - names are detected with rules only.\n"
          "      For better name detection run:  python -m spacy download en_core_web_sm")

# -- REGEX PATTERNS (structured PII) ------------------------------------------
# Order matters: emails first, so 'john.smith' inside an email is not seen twice.

PATTERNS = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("IP_ADDRESS", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")),
    ("EMPLOYEE_ID", re.compile(r"\b(?:EMP|ZEN)[-\s]?\d{3,6}\b", re.IGNORECASE)),
    # (?<![\w+]) instead of \b so the '+91-' prefix is masked together with the number
    ("PHONE", re.compile(r"(?<![\w+])(?:\+?91[\-\s]?)?(?:\d{10}|\d{5}[\-\s]\d{5}|\d{3}[\-\s]\d{3}[\-\s]\d{4})\b")),
    # DOMAIN\username
    # (not a folder path like %AppData%\SAP\Common)
    ("USERNAME", re.compile(r"(?<![\\/%:\w])[A-Za-z][A-Za-z0-9\-]{1,15}\\[A-Za-z][\w.\-]{1,30}(?![\\/\w])")),
]

# "username: jsmith", "user id = rk2041", "login id is p.sharma", "AD account: jdoe"
USERNAME_CUE = re.compile(
    r"\b(?:user\s?name|user\s?id|login\s?id|logon\s?name|sam\s?account\s?name|"
    r"ad\s+account|account\s+name|login|logon|account)\s*(?:[:=]|\bis\b)\s*[\"']?"
    r"(?P<value>[A-Za-z][\w.\-]{1,30})", re.IGNORECASE)
# strong cues may be followed by the value with only a space: "username jsmith"
USERNAME_CUE_SPACE = re.compile(
    r"\b(?:user\s?name|user\s?id|login\s?id|sam\s?account\s?name)\s+[\"']?"
    r"(?P<value>[A-Za-z][\w.\-]{1,30})", re.IGNORECASE)
# standalone first.last handle (not part of an email, path, domain or file name)
DOTTED_HANDLE = re.compile(r"(?<![\w@./\\\-\[])(?P<value>[a-z]{2,}\.[a-z]{2,})(?![\w@\]]|\.[a-z])")
NOT_A_HANDLE_SUFFIX = {"md", "py", "txt", "com", "net", "org", "io", "in", "co", "internal", "local",
                       "exe", "msi", "json", "csv", "html", "log", "pdf", "docx", "xlsx", "zip",
                       "ps1", "bat", "sh", "yaml", "yml", "cfg", "ini", "dll", "app", "dmg"}
USERNAME_STOPWORDS = {"is", "and", "or", "the", "for", "to", "after", "was", "has", "not", "field",
                      "locked", "disabled", "expired", "failed", "error", "reset", "unlock", "setup",
                      "page", "screen", "details", "required", "missing", "none"}

# -- NAME RULES (work without spaCy) ------------------------------------------

_NAME_WORD = r"[A-Z](?:[a-z]+|'[A-Z][a-z]+)(?:[A-Z][a-z]+)*(?:['\-][A-Z]?[a-z]+)*"
NAME_AFTER_CUE = re.compile(
    r"\b(?:[Uu]ser|[Cc]ontractor|[Ee]mployee|[Rr]equester|[Cc]aller|[Mm]anager|[Ee]ngineer|"
    r"[Cc]olleague|[Cc]ontact|Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Dear|Hi|Hello|[Nn]amed|[Cc]alled|"
    r"for|by|from|with|cc|CC)[:,]?\s+"
    rf"(?P<name>{_NAME_WORD}(?:\s+{_NAME_WORD}){{0,3}})")
CAPITALISED_RUN = re.compile(rf"\b(?P<name>{_NAME_WORD}(?:\s+{_NAME_WORD}){{0,3}})")

# Common first names (Indian + international). Ambiguous English words such as
# Will, Mark, Grant, Bill, Chase are deliberately left out.
FIRST_NAMES = set("""
Aarav Abhishek Aditi Aditya Ajay Akash Akshay Alok Aman Amit Amol Anand Anil Anita Anjali Ankit
Anup Arjun Arun Aruna Ashok Ashwin Basavaraj Bharat Deepa Deepak Dinesh Divya Ganesh Gaurav Girish
Harish Hemant Imran Jatin Jyoti Karan Kavita Kiran Kishore Krishna Kumar Lakshmi Mahesh Manish
Manoj Meena Mohan Mukesh Nandini Naresh Neha Nikhil Nilesh Nitin Pankaj Pooja Prakash Pranav
Prasad Prashant Pratik Praveen Priya Rahul Rajesh Rajiv Rakesh Ramesh Ravi Rekha Ritu Rohit
Sachin Sagar Sameer Sandeep Sanjay Santosh Sarita Satish Shalini Shilpa Shreya Shweta Siddharth
Sneha Sonal Sunil Sunita Suresh Swati Tarun Uday Usha Varun Vijay Vikas Vikram Vinay Vinod Vishal
Yogesh Adam Alex Alice Amanda Andrew Anna Anthony Ben Brian Carlos Charles Chris Daniel David
Emily Emma Eric George Hannah James Jane Jason Jennifer Jessica John Joseph Kate Kevin Laura Linda
Lisa Maria Mary Matthew Michael Mohammed Nancy Paul Peter Rachel Richard Robert Ryan Sarah Steven
Susan Thomas Tom Victoria Wei
""".split())

# Capitalised words that are NOT names in IT tickets (stops false positives)
NON_NAME_WORDS = set("""
User Users Contractor Employee Requester Caller Manager Engineer Team Teams Desk Service Support
Finance Sales Marketing Legal Admin Board Building Floor Room Office Site Branch Department
VPN SAP ERP Exchange Outlook Windows Mac Macbook MacBook Sonoma Office365 Microsoft Cisco Webex Zoom
Salesforce Adobe Acrobat Python Oracle Linux Android Apple Google Chrome Edge Firefox SharePoint
OneDrive Azure Active Directory AnyConnect Authenticator Teams Slack Jira ServiceNow HP LaserJet Pro
Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April May June July
August September October November December Today Tomorrow Yesterday
Password Reset Access Grant Request Ticket Incident Error Issue Problem Network Server Laptop Printer
Email Mail Account Login Logon Contact Please Thanks Thank Regards Dear Hi Hello The This That These
New Old Multiple Senior Junior Cannot Unable Install Setup Update Upgrade Phoenix Project Projects
Step Steps Note Warning Critical High Medium Low Open Closed Resolved Pending Approved Escalated
Network-Ops App-Support Desktop-Support Email-Support Service-Desk Security-Ops Server-Ops DBA-Team
Zensar IT ISDO L1 L2 Push Pull Settings Fetch Data
Mobile Device Devices Phone Phones Wi-Fi Wifi Internet Database Backup Job Nightly Client Clients
""".split())

# -- AUDIT LOG (redaction events) ---------------------------------------------

audit_log = []


def _audit(action, detail):
    entry = {"timestamp": datetime.now().isoformat(), "module": "PIIRedactor",
             "action": action, "detail": detail}
    audit_log.append(entry)
    return entry

# -- HELPERS ------------------------------------------------------------------

def _clean_name(candidate: str) -> str:
    """Trim a capitalised run to its name part: drop non-name words at either end."""
    words = candidate.split()
    while words and words[0] in NON_NAME_WORDS:
        words.pop(0)
    cut = next((i for i, w in enumerate(words) if w in NON_NAME_WORDS), len(words))
    words = words[:cut]
    if words and words[-1].endswith("'s"):
        words[-1] = words[-1][:-2]
    return " ".join(words)


def _rule_based_names(text: str) -> list[str]:
    names = []
    for m in NAME_AFTER_CUE.finditer(text):                       # "User John Smith"
        name = _clean_name(m.group("name"))
        if name:
            names.append(name)
    for m in CAPITALISED_RUN.finditer(text):                      # "Rahul Verma reports"
        name = _clean_name(m.group("name"))
        if name and name.split()[0] in FIRST_NAMES:
            names.append(name)
    return names


def _spacy_names(text: str) -> list[str]:
    if not SPACY_AVAILABLE:
        return []
    names = []
    for ent in nlp(text).ents:
        if ent.label_ != "PERSON" or ent.text.isupper() or "[" in ent.text:
            continue                       # skip acronyms (VPN, SLA) and existing tokens
        name = _clean_name(ent.text.strip())
        if name and any(c.isalpha() for c in name):
            names.append(name)
    return names

# -- REDACT / RESTORE ---------------------------------------------------------

def redact(text: str) -> tuple[str, dict]:
    """
    Redact PII from text. Returns (clean_text, mapping).
      clean, m = redact("User John Smith, login id jsmith01, call +91-9876543210")
      clean -> "User [NAME_1], login id [USERNAME_1], call [PHONE_1]"
      m     -> {"[NAME_1]": "John Smith", "[USERNAME_1]": "jsmith01", "[PHONE_1]": "+91-9876543210"}
    The same value always gets the same token within one call.
    """
    if not text:
        return text, {}
    mapping, value_to_token, counters = {}, {}, {}
    clean = text

    def mask(value: str, label: str):
        nonlocal clean
        value = value.strip()
        if not value or value.startswith("["):
            return
        pattern = rf"(?<![\w\[]){re.escape(value)}(?![\w\]])"
        if not re.search(pattern, clean):          # already masked / not present
            return
        token = value_to_token.get(value)
        if token is None:
            counters[label] = counters.get(label, 0) + 1
            token = f"[{label}_{counters[label]}]"
            value_to_token[value], mapping[token] = token, value
        # whole-word replacement only, never inside another word or token
        clean = re.sub(pattern, token, clean)

    # 1) structured PII
    for label, pattern in PATTERNS:
        for m in list(pattern.finditer(clean)):
            mask(m.group(0), label)
    for pattern in (USERNAME_CUE, USERNAME_CUE_SPACE):
        for m in list(pattern.finditer(clean)):
            value = m.group("value").rstrip(".-")
            if value.lower() not in USERNAME_STOPWORDS:
                mask(value, "USERNAME")
    for m in list(DOTTED_HANDLE.finditer(clean)):
        if m.group("value").split(".")[-1] not in NOT_A_HANDLE_SUFFIX:
            mask(m.group("value"), "USERNAME")

    # 2) + 3) person names - longest first, so "John Smith" wins over "John"
    names = set(_spacy_names(clean)) | set(_rule_based_names(clean))
    for name in sorted(names, key=len, reverse=True):
        mask(name, "NAME")

    # 4) propagate: a bare surname/first name seen later in the text
    for token, value in list(mapping.items()):
        if token.startswith("[NAME_"):
            for part in value.split():
                if len(part) >= 3 and part not in NON_NAME_WORDS:
                    mask(part, "NAME")

    _audit("redact", f"{len(mapping)} PII item(s) masked: {list(mapping.keys())}"
           if mapping else "No PII detected")
    return clean, mapping


def restore(text: str, mapping: dict) -> str:
    """Restore PII tokens back to original values (for system-of-record logging only)."""
    restored = text
    for token in sorted(mapping, key=len, reverse=True):
        restored = restored.replace(token, mapping[token])
    _audit("restore", f"{len(mapping)} PII item(s) restored")
    return restored


def get_audit_log() -> list:
    """Return all PII redaction audit entries."""
    return audit_log

# -- AUDIT TRAIL LOGGER -------------------------------------------------------

class AuditLogger:
    """Logs every agent action with timestamp, agent name, tool, rationale, approval.
    The rationale is PII-redacted before it is printed or written to disk."""

    def __init__(self, log_file: str | Path = PROJECT_ROOT / "logs" / "audit_trail.jsonl",
                 echo: bool = True):
        self.log_file = Path(log_file)
        if not self.log_file.is_absolute():
            self.log_file = PROJECT_ROOT / self.log_file      # independent of the cwd
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.entries = []
        self.echo = echo            # False = write to file only (caller prints its own view)

    def log(self, agent: str, action: str, ticket_number: str = "",
            tool: str = "", rationale: str = "", approval_status: str = "N/A"):
        safe_rationale, pii = redact(rationale) if rationale else ("", {})
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "action": action,
            "ticket_number": ticket_number,
            "tool": tool,
            "rationale": safe_rationale[:200],
            "pii_redacted": len(pii),
            "approval_status": approval_status,
        }
        self.entries.append(entry)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        if self.echo:
            print(f"  [AUDIT] {agent} | {action} | {ticket_number} | {approval_status}")
        return entry

    def print_trail(self):
        print(f"\n{'=' * 55}\nFULL AUDIT TRAIL ({len(self.entries)} entries)\n{'=' * 55}")
        for e in self.entries:
            print(f"  {e['timestamp'][:19]}  {e['agent']:<22} {e['action']:<20} {e['approval_status']}")

# -- DEMO ---------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print(f"PII REDACTION DEMO   (spaCy NER: {'ON' if SPACY_AVAILABLE else 'OFF - rules only'})")
    print("=" * 55)

    sample_tickets = [
        "User John Smith (emp ID ZEN-9823) reports VPN failure. Contact: john.smith@zensar.com or +91-9876543210.",
        "Contractor sarah.jones@client.com needs access to REQ-1002. IP: 192.168.1.45.",
        "Password reset for Michael D'Souza. Employee EMP-00142. No PII in this part.",
        "VPN not connecting after password change. Error: authentication failed. Ticket INC0001001.",
        # usernames
        "AD account locked for username rkumar01. Rahul Verma says Verma's laptop also fails.",
        "Login failure for CORP\\psharma on SAP. User id: p.sharma, phone 080-555-1234.",
        "Dear Priya, user anita.desai cannot open vpn_troubleshooting.md on sap-monitor.zensar.internal.",
    ]

    for i, ticket in enumerate(sample_tickets, 1):
        print(f"\n--- Ticket {i} ---")
        print(f"Original : {ticket}")
        clean, mapping = redact(ticket)
        print(f"Redacted : {clean}")
        if mapping:
            print(f"Mapping  : {mapping}")
        assert restore(clean, mapping) == ticket, "restore() must give back the original"

    print("\n" + "=" * 55)
    print("AUDIT TRAIL DEMO")
    print("=" * 55)

    logger = AuditLogger("logs/demo_audit.jsonl")
    logger.log("TriageAgent", "classify_ticket", "INC0001001", "classify_ticket",
               "Network/P2 - VPN failure after password change", "Auto")
    logger.log("ResolutionAgent", "search_kb", "INC0001001", "search_kb",
               "KB article found: vpn_troubleshooting.md (85% confidence)", "Auto")
    logger.log("SLAAgent", "get_sla_status", "INC0001001", "get_sla_status",
               "SLA AT_RISK - 90 min remaining of 240 min total", "Auto")
    logger.log("HITLGate", "approval_request", "REQ-1002", "",
               "Access grant for contractor sarah.jones@client.com needs approval", "PENDING")
    logger.log("HITLGate", "approval_decision", "INC0001002", "",
               "Human operator approved P1 escalation", "APPROVED")
    logger.log("CommunicationAgent", "post_comment", "INC0001001", "post_comment",
               "Resolution sent to user John Smith - auto-resolved L1 ticket", "Auto")

    logger.print_trail()
    print(f"\nAudit log saved to: {logger.log_file}  (rationales are PII-redacted)")
