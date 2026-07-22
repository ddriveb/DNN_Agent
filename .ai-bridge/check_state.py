import sqlite3
import json

db_path = r'C:\Users\User\AppData\Roaming\Code\User\globalStorage\state.vscdb'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 看看有哪些表
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print('Tables:', [t[0] for t in tables])

# 看看 ItemTable 里有什么
cursor.execute('SELECT key, length(value) FROM ItemTable ORDER BY key')
rows = cursor.fetchall()
for row in rows:
    print(f'{row[0]}: {row[1]} bytes')

conn.close()
