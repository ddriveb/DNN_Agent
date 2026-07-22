import sqlite3
import json
import os

db_path = r'C:\Users\User\AppData\Roaming\Code\User\globalStorage\state.vscdb'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 读取 OpenAI ChatGPT 扩展数据
cursor.execute("SELECT value FROM ItemTable WHERE key = 'openai.chatgpt'")
row = cursor.fetchone()
if row:
    data = row[0]
    print(f"openai.chatgpt data size: {len(data)} bytes")
    
    # 保存到文件
    output_path = r'C:\Users\User\.kimi_openclaw\workspace\dnn-agent\.ai-bridge\openai-chatgpt-data.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        try:
            parsed = json.loads(data)
            json.dump(parsed, f, indent=2, ensure_ascii=False)
            print(f"Saved parsed JSON to {output_path}")
        except:
            f.write(data)
            print(f"Saved raw data to {output_path}")

# 读取聊天会话索引
cursor.execute("SELECT value FROM ItemTable WHERE key = 'chat.ChatSessionStore.index'")
row = cursor.fetchone()
if row:
    data = row[0]
    output_path = r'C:\Users\User\.kimi_openclaw\workspace\dnn-agent\.ai-bridge\chat-session-index.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        try:
            parsed = json.loads(data)
            json.dump(parsed, f, indent=2, ensure_ascii=False)
            print(f"Saved chat session index to {output_path}")
        except:
            f.write(data)
            print(f"Saved raw chat session index to {output_path}")

conn.close()
