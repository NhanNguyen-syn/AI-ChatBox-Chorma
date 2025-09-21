import sqlite3
conn=sqlite3.connect('chroma_chat.db')
cur=conn.cursor()
cur.execute("SELECT id, username, email, is_admin, is_active FROM users")
rows=cur.fetchall()
print('Users:', rows)

