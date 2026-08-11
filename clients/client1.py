# =============================================================================
# clients/client1.py
# FL Client — Substation A
# Run: python clients/client1.py
# =============================================================================
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clients.base_client import start_client

if __name__ == "__main__":
    print("[CLIENT] Substation A")
    start_client(client_id="1")
