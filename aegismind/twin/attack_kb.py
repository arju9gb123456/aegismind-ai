"""Small MITRE ATT&CK lookup used to tag *simulated* steps.

This is only a labelling table (technique IDs + names) so that dashboard
and paper use standard vocabulary. It contains no attack instructions.
Reference: https://attack.mitre.org/
"""
from __future__ import annotations

# service / relation used in the twin -> (technique id, technique name, tactic)
SERVICE_TO_TECHNIQUE: dict[str, tuple[str, str, str]] = {
    "phishing": ("T1566", "Phishing", "Initial Access"),
    "public-web": ("T1190", "Exploit Public-Facing Application", "Initial Access"),
    "smb": ("T1021.002", "Remote Services: SMB/Windows Admin Shares", "Lateral Movement"),
    "rdp": ("T1021.001", "Remote Services: Remote Desktop Protocol", "Lateral Movement"),
    "ssh": ("T1021.004", "Remote Services: SSH", "Lateral Movement"),
    "winrm": ("T1021.006", "Remote Services: Windows Remote Management", "Lateral Movement"),
    "http": ("T1210", "Exploitation of Remote Services", "Lateral Movement"),
    "sql": ("T1210", "Exploitation of Remote Services", "Lateral Movement"),
    "smtp": ("T1210", "Exploitation of Remote Services", "Lateral Movement"),
    "ldap": ("T1087.002", "Account Discovery: Domain Account", "Discovery"),
    "cached-admin": ("T1078.002", "Valid Accounts: Domain Accounts", "Privilege Escalation"),
    "service-account": ("T1078", "Valid Accounts", "Defense Evasion"),
    "domain-trust": ("T1484", "Domain or Tenant Policy Modification", "Privilege Escalation"),
}

UNKNOWN = ("T0000", "Unmapped", "Unknown")


def technique_for(service: str) -> dict[str, str]:
    tid, name, tactic = SERVICE_TO_TECHNIQUE.get(service, UNKNOWN)
    return {"technique_id": tid, "technique": name, "tactic": tactic}
