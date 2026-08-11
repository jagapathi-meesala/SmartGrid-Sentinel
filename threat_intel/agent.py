# =============================================================================
# threat_intel/agent.py
# Threat Intelligence Integration Agent for SmartGrid Sentinel.
#
# Fetches external threat feeds, correlates threat intelligence indicators
# with the Digital Twin's live simulation event stream, enriches anomaly events
# with MISP threat categories, and updates threat scenario parameters.
# =============================================================================

import os
import sys
import json
from datetime import datetime, timezone
from typing import List, Dict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config.settings import RESULTS_DIR, LOGS_DIR
from threat_intel.database import ThreatIntelligenceDatabase

TWIN_EVENTS_PATH = os.path.join(RESULTS_DIR, "twin_events.json")
ENRICHED_EVENTS_PATH = os.path.join(RESULTS_DIR, "threat_intelligence_events.json")
THREAT_LOG_PATH = os.path.join(LOGS_DIR, "threat_intel.log")


def log_threat_agent(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] [THREAT-INTEL-AGENT] {msg}"
    print(line)
    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(THREAT_LOG_PATH, "a") as f:
        f.write(line + "\n")


class ThreatIntelligenceAgent:
    """
    Threat Intelligence Integration Agent.
    - Connects to ThreatIntelligenceDatabase
    - Fetches external threat feeds (STIX / MISP indicators)
    - Enriches Digital Twin traffic events with threat metadata
    - Generates threat threat-score correlation analysis
    """

    def __init__(self):
        self.db = ThreatIntelligenceDatabase()
        log_threat_agent(f"Threat Intelligence Agent initialized with {len(self.db.get_all_indicators())} Threat DB indicators.")

    def fetch_external_feeds(self) -> List[Dict]:
        """Simulate fetching external threat intelligence feed updates (e.g. MISP / CISA / ICS-CERT)."""
        log_threat_agent("Fetching external threat intelligence feeds (MISP / CISA ICS-CERT)...")
        feed_updates = [
            {
                "ioc_id": "IOC-2026-FDI-005",
                "threat_type": "False Data Injection (FDI)",
                "protocol": "IEC 61850",
                "severity": "CRITICAL",
                "signature_pattern": "frequency_instability > 0.05 Hz",
                "misp_category": "External feed update",
                "description": "Grid frequency offset attack targeting AGC (Automatic Generation Control).",
                "confidence": 0.96,
                "first_seen": datetime.now(timezone.utc).isoformat()
            }
        ]
        for item in feed_updates:
            self.db.add_indicator(item)
            log_threat_agent(f"Ingested new threat indicator {item['ioc_id']} into Threat DB.")
        return feed_updates

    def process_twin_events(self, twin_events: List[Dict]) -> List[Dict]:
        """
        Connect to Digital Twin and correlate flagged anomaly flows with Threat DB indicators.
        Adds threat_ioc, misp_category, and threat_severity to twin events.
        """
        log_threat_agent(f"Processing {len(twin_events)} Digital Twin simulation events...")
        indicators = self.db.get_all_indicators()
        enriched = []

        for ev in twin_events:
            ev_copy = dict(ev)
            if ev.get("flagged") or ev.get("was_injected"):
                # Match against Threat DB indicators based on predicted attack class or risk
                matched_ioc = None
                pred_class = ev.get("predicted_class", "Attack")
                for ioc in indicators:
                    if ioc["threat_type"].lower() in pred_class.lower() or "attack" in ioc["threat_type"].lower():
                        matched_ioc = ioc
                        break
                
                if matched_ioc:
                    ev_copy["threat_ioc"] = matched_ioc["ioc_id"]
                    ev_copy["misp_category"] = matched_ioc["misp_category"]
                    ev_copy["threat_severity"] = matched_ioc["severity"]
                    ev_copy["threat_confidence"] = matched_ioc["confidence"]
                    log_threat_agent(
                        f"CORRELATED tick={ev['tick']:02d} sub={ev['substation']} "
                        f"pred={pred_class} -> {matched_ioc['ioc_id']} ({matched_ioc['severity']})"
                    )
            enriched.append(ev_copy)

        return enriched

    def run(self):
        # Step 1: Fetch feeds
        self.fetch_external_feeds()

        # Step 2: Read Digital Twin events
        if not os.path.exists(TWIN_EVENTS_PATH):
            log_threat_agent(f"Warning: {TWIN_EVENTS_PATH} not found. Operating in standalone mode.")
            return

        with open(TWIN_EVENTS_PATH) as f:
            twin_events = json.load(f)

        # Step 3: Correlate & Enrich
        enriched_events = self.process_twin_events(twin_events)

        # Step 4: Persist enriched threat scenario dataset
        with open(ENRICHED_EVENTS_PATH, "w") as f:
            json.dump(enriched_events, f, indent=2)

        log_threat_agent(f"Enriched threat scenario dataset saved -> {ENRICHED_EVENTS_PATH}")
        return enriched_events


def main():
    agent = ThreatIntelligenceAgent()
    agent.run()


if __name__ == "__main__":
    main()
