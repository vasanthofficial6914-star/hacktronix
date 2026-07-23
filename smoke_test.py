"""Quick E2E smoke test via REST API."""
import httpx, json

base = "http://localhost:8000/api/v1"

# Run agent 5 steps
r = httpx.post(f"{base}/run", json={
    "objective": "Find and collect the EXIT GEM from the Throne Room",
    "max_steps": 5
}, timeout=30)
result = r.json()
print(f"Success: {result['success']}")
print(f"Steps taken: {result['steps_taken']}")
for log in result.get("reasoning_log", []):
    step = log.get("step", "?")
    room = log.get("room", "?")
    action = log.get("action", "?")
    res = log.get("result", "?")[:60]
    print(f"  Step {step}: [{room}] {action} -> {res}")

# World state
s = httpx.get(f"{base}/state", timeout=10).json()
print(f"\nWorld Model: {s['total_entities']} entities, {s['total_relationships']} relationships, {s['graph_nodes']} graph nodes")
