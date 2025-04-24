from langgraph.graph import StateGraph, END
from graph.state import GraphState  # ✅ State schema

from graph.graph_builder import (
    planner_node,
    product_node,
    blog_node,
    policy_node,
    order_node,
    memory_node,
    final_node
)

def build_graph():
    workflow = StateGraph(GraphState)

    # Register all nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("product", product_node)
    workflow.add_node("blog", blog_node)
    workflow.add_node("policy", policy_node)
    workflow.add_node("order", order_node)
    workflow.add_node("memory", memory_node)
    workflow.add_node("final", final_node)

    # Set entry point
    workflow.set_entry_point("planner")

    # Define intelligent router
    def router(state: GraphState):
        plan = state.get("plan", {})
        intents = plan.get("intents", [])
        actions = plan.get("actions", [])
        resolved_entities = plan.get("resolved_entities", [])
        structured_memory = state.get("structured_memory", {})

        # ✅ Ensure all entities have actual chunks before skipping agents
        all_entities_have_chunks = all(
            ent in structured_memory and structured_memory[ent].get("chunks")
            for ent in resolved_entities
        )

        if resolved_entities and all_entities_have_chunks:
            print("🚀 [ROUTER] All entities have chunks in memory — skipping agent nodes.")
            return ["final"]

        return [a["type"] for a in actions] or ["final"]

    # Add conditional transitions
    workflow.add_conditional_edges(
        source="planner",
        path=router,
        path_map={
            "product": "product",
            "blog": "blog",
            "policy": "policy",
            "order": "order",
            "general": "final",
            "final": "final"
        }
    )

    # Connect agents to memory → final
    for node in ["product", "blog", "policy", "order"]:
        workflow.add_edge(node, "memory")

    workflow.add_edge("memory", "final")
    workflow.add_edge("final", END)

    return workflow.compile()

def run_graph(query, last_user=None, last_response=None, memory_summary="", last_chunks=None, structured_memory=None):
    graph = build_graph()

    initial_state = {
        "query": query,
        "last_user": last_user,
        "last_response": last_response,
        "memory_summary": memory_summary,
        "last_chunks": last_chunks or [],
        "structured_memory": structured_memory or {}
    }

    return graph.invoke(initial_state)
