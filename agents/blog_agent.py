# agents/blog_agent.py

from tools.blog_tool import BlogSearchTool

blog_tool = BlogSearchTool()

def handle_blog_query(query: str) -> list:
    print(f"\n📝 [BLOG AGENT] Handling blog query: '{query}'")
    results = blog_tool.run(query)
    print(f"✅ [BLOG AGENT] Retrieved {len(results)} blog chunks.")
    return results
