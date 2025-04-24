# tools/order_tool.py

import requests
from datetime import datetime
import os

SHOP_NAME = "malloftoys"
API_VERSION = "2024-01"
ACCESS_TOKEN = os.getenv("SHOPIFY_API_KEY")
BASE_URL = f"https://{SHOP_NAME}.myshopify.com/admin/api/{API_VERSION}"
HEADERS = {
    "Content-Type": "application/json",
    "X-Shopify-Access-Token": ACCESS_TOKEN
}

class OrderTrackingTool:
    def get_order_id(self, order_number):
        encoded = f"%23{order_number}"
        url = f"{BASE_URL}/orders.json?name={encoded}&status=any"
        res = requests.get(url, headers=HEADERS)
        if res.status_code == 200:
            orders = res.json().get("orders", [])
            return orders[0]["id"] if orders else None
        return None

    def get_order_details(self, order_id):
        url = f"{BASE_URL}/orders/{order_id}.json"
        res = requests.get(url, headers=HEADERS)
        if res.status_code != 200:
            return None

        order = res.json()["order"]

        # Extracting basic order details
        status = order.get("fulfillment_status", "In Progress")
        payment = order.get("financial_status", "Unpaid")
        currency = order.get("currency", "USD")
        total = order.get("total_price", "0.00")

        # Extracting customer details
        customer = order.get("customer", {})
        customer_email = customer.get("email", "No email")
        customer_name = f"{customer.get('first_name', '')} {customer.get('last_name', '')}".strip() or "Unknown"

        # Extracting dates
        created_at = order.get("created_at", "Unknown date")
        updated_at = order.get("updated_at", "Unknown date")

        # Extracting addresses (billing and shipping)
        shipping_address = order.get("shipping_address", {})
        billing_address = order.get("billing_address", {})

        shipping_address_str = f"{shipping_address.get('address1', '')}, {shipping_address.get('city', '')}, {shipping_address.get('province', '')}, {shipping_address.get('country', '')}"
        billing_address_str = f"{billing_address.get('address1', '')}, {billing_address.get('city', '')}, {billing_address.get('province', '')}, {billing_address.get('country', '')}"

        # Extracting products
        products = [
            {
                "product_name": item.get("title", "Unknown"),
                "quantity": item.get("quantity", 0),
                "price": f"{item.get('price', '0.00')} {currency}"
            } for item in order.get("line_items", [])
        ]

        # Returning the extended order details
        return {
            "order_number": order.get("name"),
            "status": status,
            "payment_status": payment,
            "total_price": f"{total} {currency}",
            "customer_email": customer_email,
            "customer_name": customer_name,
            "created_at": created_at,
            "updated_at": updated_at,
            "shipping_address": shipping_address_str,
            "billing_address": billing_address_str,
            "products": products
        }

    def run(self, order_number):
        order_id = self.get_order_id(order_number)
        if order_id:
            return self.get_order_details(order_id)
        return {"error": "Order not found"}
