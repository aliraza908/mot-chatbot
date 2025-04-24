# agents/policy_agent.py

from tools.policy_tool import PolicySearchTool

policy_tool = PolicySearchTool()

def handle_policy_query(query: str) -> list:
    print(f"\n📜 [POLICY AGENT] Handling policy query: '{query}'")
    results = policy_tool.run(query)
    print(f"✅ [POLICY AGENT] Retrieved {len(results)} policy chunks.")
    return results
