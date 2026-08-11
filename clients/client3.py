# =============================================================================
# clients/client3.py
# FL Client — Substation C
# Run: python clients/client3.py
# =============================================================================
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clients.base_client import start_client

if __name__ == "__main__":
    print("[CLIENT] Substation C")
    start_client(client_id="3")
