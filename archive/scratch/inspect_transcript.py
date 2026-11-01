import json

log_path = r"C:\Users\Aryan Deo\.gemini\antigravity-ide\brain\fe82ba9b-a55b-41d7-b7e6-d16a58754613\.system_generated\logs\transcript.jsonl"
with open(log_path, "r", encoding="utf-8") as fi:
    for line in fi:
        obj = json.loads(line)
        if obj.get("step_index") == 4018:
            print(obj.get("content", ""))
