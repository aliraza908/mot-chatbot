# agents/order_agent.py

from tools.order_tool import OrderTrackingTool

order_tool = OrderTrackingTool()

def handle_order_tracking(order_number: str) -> dict:
    print(f"\n📦 [ORDER AGENT] Looking up order number: {order_number}")
    result = order_tool.run(order_number)
    if "error" in result:
        print("⚠️ [ORDER AGENT] Order not found.")
    else:
        print("✅ [ORDER AGENT] Order found with details.")
    return result
