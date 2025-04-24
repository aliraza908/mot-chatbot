from typing import TypedDict, Optional, List, Dict

class GraphState(TypedDict):
    query: str
    plan: Optional[dict]
    chunks: Optional[List[str]]
    order_details: Optional[dict]
    last_user: Optional[str]
    last_response: Optional[str]
    is_followup: Optional[bool]
    memory_summary: Optional[str]
    response: Optional[str]
    last_chunks: Optional[List[str]]
    structured_memory: Optional[Dict[str, dict]]  # <-- include this if you're using entity memory
