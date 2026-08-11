# =============================================================================
# ara/agent.py
# Autonomous Response Agent (ARA) — real, minimal, rule-based implementation.
#
# This is intentionally NOT an LLM agent and NOT a full decision-theoretic
# controller. It's a deterministic state machine that closes the loop the
# original C4 diagram left conceptual: it consumes the Digital Twin's alert
# stream (results/twin_events.json) and takes a simulated mitigation action
# when a substation's risk crosses the threshold, then restores it after a
# cooldown once risk subsides. This is honestly scoped: real code, real
# state transitions, logged and testable — not a stub, but also not an
# LLM-based decision system. Extending this to a learned policy or an
# LLM-reasoned response is future work (see README).
#
# Run: python ara/agent.py   (after digital_twin/simulator.py)
# =============================================================================

import os
import sys
import json
from datetime import datetime, timezone
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import RESULTS_DIR, LOGS_DIR, SUBSTATIONS

TWIN_EVENTS_PATH = os.path.join(RESULTS_DIR, "twin_events.json")
ARA_ACTIONS_PATH = os.path.join(RESULTS_DIR, "ara_actions.json")
ARA_LOG_PATH = os.path.join(LOGS_DIR, "ara.log")

# Cooldown: consecutive clean ticks required before an isolated substation
# is automatically restored to normal.
RESTORE_COOLDOWN_TICKS = 3


class SubstationState(Enum):
    NORMAL = "normal"
    ISOLATED = "isolated"


def log_ara(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] [ARA] {msg}"
    print(line)
    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(ARA_LOG_PATH, "a") as f:
        f.write(line + "\n")


class AutonomousResponseAgent:
    """
    Deterministic rule-based response agent.

    Rule set (intentionally simple and auditable, not a black box):
      1. If ANY flow at substation S in tick T is flagged (risk > dynamic
         threshold), and S is currently NORMAL -> transition to ISOLATED.
         Log the action with the triggering risk/confidence.
      2. If S is ISOLATED and has gone RESTORE_COOLDOWN_TICKS consecutive
         ticks with zero flagged flows -> transition back to NORMAL.
      3. Every transition is logged as a discrete action with a timestamp,
         substation, from/to state, and the evidence that triggered it.
    """

    def __init__(self, substations: list = None, cooldown: int = RESTORE_COOLDOWN_TICKS):
        self.substations = substations or SUBSTATIONS
        self.cooldown = cooldown
        self.state = {s: SubstationState.NORMAL for s in self.substations}
        self.clean_streak = {s: 0 for s in self.substations}
        self.actions = []

    def _record_action(self, tick, substation, from_state, to_state, reason, evidence):
        action = {
            "tick": tick,
            "substation": substation,
            "from_state": from_state.value,
            "to_state": to_state.value,
            "reason": reason,
            "evidence": evidence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.actions.append(action)
        log_ara(f"tick={tick:02d} {substation}: {from_state.value} -> {to_state.value} "
                 f"({reason})")

    def process_tick(self, tick: int, tick_events: list):
        """tick_events: list of twin event dicts for this tick, across all substations."""
        by_sub = {s: [] for s in self.substations}
        for ev in tick_events:
            by_sub.setdefault(ev["substation"], []).append(ev)

        for sub in self.substations:
            events = by_sub.get(sub, [])
            flagged = [e for e in events if e.get("flagged")]

            if flagged:
                self.clean_streak[sub] = 0
                if self.state[sub] == SubstationState.NORMAL:
                    top = max(flagged, key=lambda e: e["risk"])
                    self._record_action(
                        tick, sub, SubstationState.NORMAL, SubstationState.ISOLATED,
                        reason=f"risk {top['risk']:.3f} exceeded threshold "
                               f"{top['dynamic_threshold']:.3f}",
                        evidence={"predicted_class": top["predicted_class"],
                                  "confidence": top["confidence"], "risk": top["risk"]},
                    )
                    self.state[sub] = SubstationState.ISOLATED
            else:
                if self.state[sub] == SubstationState.ISOLATED:
                    self.clean_streak[sub] += 1
                    if self.clean_streak[sub] >= self.cooldown:
                        self._record_action(
                            tick, sub, SubstationState.ISOLATED, SubstationState.NORMAL,
                            reason=f"{self.cooldown} consecutive clean ticks",
                            evidence={"clean_streak": self.clean_streak[sub]},
                        )
                        self.state[sub] = SubstationState.NORMAL
                        self.clean_streak[sub] = 0

    def run(self, twin_events: list):
        by_tick = {}
        for ev in twin_events:
            by_tick.setdefault(ev["tick"], []).append(ev)
        for tick in sorted(by_tick):
            self.process_tick(tick, by_tick[tick])
        return self.actions


def main():
    if not os.path.exists(TWIN_EVENTS_PATH):
        raise FileNotFoundError(
            f"{TWIN_EVENTS_PATH} not found. Run digital_twin/simulator.py first — "
            "the ARA acts on the twin's real alert stream, it doesn't generate its own."
        )
    with open(TWIN_EVENTS_PATH) as f:
        twin_events = json.load(f)

    agent = AutonomousResponseAgent()
    actions = agent.run(twin_events)

    with open(ARA_ACTIONS_PATH, "w") as f:
        json.dump(actions, f, indent=2)

    isolations = [a for a in actions if a["to_state"] == "isolated"]
    restorations = [a for a in actions if a["to_state"] == "normal"]
    print(f"\n[ARA] {len(actions)} total actions: "
          f"{len(isolations)} isolations, {len(restorations)} restorations")
    print(f"[ARA] Saved -> {ARA_ACTIONS_PATH}")
    print(f"[ARA] Final substation states: "
          f"{ {s: st.value for s, st in agent.state.items()} }")


if __name__ == "__main__":
    main()
