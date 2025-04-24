# memory/memory_store.py

class MemoryStore:
    def __init__(self):
        self.memory_summary = ""
        self.last_user_query = ""
        self.last_assistant_response = ""

    def update(self, user_msg, assistant_msg):
        from memory.memory_summary import update_memory_summary
        self.memory_summary = update_memory_summary(self.memory_summary, user_msg, assistant_msg)
        self.last_user_query = user_msg
        self.last_assistant_response = assistant_msg

    def is_followup(self, current_msg):
        from memory.followup_detector import is_followup
        return is_followup(current_msg, self.last_user_query, self.last_assistant_response)
