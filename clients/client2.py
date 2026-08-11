# =============================================================================
# clients/client2.py
# FL Client — Substation B
# Run: python clients/client2.py
# =============================================================================
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clients.base_client import start_client

if __name__ == "__main__":
    print("[CLIENT] Substation B")
    start_client(client_id="2")
