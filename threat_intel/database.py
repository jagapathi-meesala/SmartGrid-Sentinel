# =============================================================================
# threat_intel/database.py
# Threat Intelligence Database for SmartGrid Sentinel.
# Stores STIX 2.1 / MISP format cyberattack IOCs, ICS threat signatures,
# and historical attack indicators for power grid substations.
# =============================================================================

import os
import sys
import json
import sqlite3
from typing import List, Dict, Optional
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from config.settings import RESULTS_DIR, LOGS_DIR

DB_PATH = os.path.join(RESULTS_DIR, "threat_intelligence.db")
JSON_PATH = os.path.join(RESULTS_DIR, "threat_db.json")


DEFAULT_THREAT_INDICATORS = [
    {
        "ioc_id": "IOC-2026-FDI-001",
        "threat_type": "False Data Injection (FDI)",
        "protocol": "IEC 61850 / GOOSE",
        "severity": "HIGH",
        "signature_pattern": "sensor_voltage_delta > 0.15",
        "misp_category": "Payload delivery",
        "description": "Manipulated busbar voltage telemetry attempting state estimation corruption.",
        "confidence": 0.95,
        "first_seen": "2026-08-01T00:00:00Z"
    },
    {
        "ioc_id": "IOC-2026-DOS-002",
        "threat_type": "Denial of Service (DoS)",
        "protocol": "DNP3 / TCP",
        "severity": "CRITICAL",
        "signature_pattern": "packet_rate > 5000 pkts/sec",
        "misp_category": "Network activity",
        "description": "High-volume SYN flood targeting substation gateway router.",
        "confidence": 0.98,
        "first_seen": "2026-08-02T12:00:00Z"
    },
    {
        "ioc_id": "IOC-2026-RPL-003",
        "threat_type": "Replay Cyberattack",
        "protocol": "Modbus TCP",
        "severity": "HIGH",
        "signature_pattern": "sequence_id_duplicate == True",
        "misp_category": "Artifacts dropped",
        "description": "Replay of previous valid breaker status frame to mask unmitigated fault.",
        "confidence": 0.92,
        "first_seen": "2026-08-03T08:30:00Z"
    },
    {
        "ioc_id": "IOC-2026-CMD-004",
        "threat_type": "Unauthorized Substation Command",
        "protocol": "IEC 60870-5-104",
        "severity": "CRITICAL",
        "signature_pattern": "command_type == TRIP_OVERRIDE",
        "misp_category": "External threat feed",
        "description": "Unauthorized remote trip signal targeting Substation transformer relay.",
        "confidence": 0.99,
        "first_seen": "2026-08-05T14:15:00Z"
    }
]


class ThreatIntelligenceDatabase:
    """
    SQLite-backed Threat Intelligence Database with JSON mirror export.
    Stores threat indicators, MISP category attributes, and signature patterns.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS threat_iocs (
                ioc_id TEXT PRIMARY KEY,
                threat_type TEXT NOT NULL,
                protocol TEXT NOT NULL,
                severity TEXT NOT NULL,
                signature_pattern TEXT NOT NULL,
                misp_category TEXT NOT NULL,
                description TEXT NOT NULL,
                confidence REAL NOT NULL,
                first_seen TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS threat_feed_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feed_source TEXT NOT NULL,
                records_ingested INTEGER NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        conn.commit()

        # Seed default indicators if empty
        cursor.execute("SELECT COUNT(*) FROM threat_iocs")
        if cursor.fetchone()[0] == 0:
            for item in DEFAULT_THREAT_INDICATORS:
                cursor.execute("""
                    INSERT INTO threat_iocs 
                    (ioc_id, threat_type, protocol, severity, signature_pattern, misp_category, description, confidence, first_seen)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    item["ioc_id"], item["threat_type"], item["protocol"], item["severity"],
                    item["signature_pattern"], item["misp_category"], item["description"],
                    item["confidence"], item["first_seen"]
                ))
            conn.commit()

        conn.close()
        self.export_json()

    def add_indicator(self, indicator: Dict):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO threat_iocs
            (ioc_id, threat_type, protocol, severity, signature_pattern, misp_category, description, confidence, first_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            indicator["ioc_id"], indicator["threat_type"], indicator["protocol"], indicator["severity"],
            indicator["signature_pattern"], indicator["misp_category"], indicator["description"],
            indicator.get("confidence", 0.9), indicator.get("first_seen", datetime.now(timezone.utc).isoformat())
        ))
        conn.commit()
        conn.close()
        self.export_json()

    def get_all_indicators(self) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM threat_iocs")
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    def match_threat_signature(self, threat_type: str) -> Optional[Dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM threat_iocs WHERE threat_type LIKE ?", (f"%{threat_type}%",))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def export_json(self):
        indicators = self.get_all_indicators()
        with open(JSON_PATH, "w") as f:
            json.dump({"total_iocs": len(indicators), "indicators": indicators}, f, indent=2)


if __name__ == "__main__":
    db = ThreatIntelligenceDatabase()
    print(f"[ThreatDB] Initialized with {len(db.get_all_indicators())} IOC records.")
    print(f"[ThreatDB] Exported -> {JSON_PATH}")
