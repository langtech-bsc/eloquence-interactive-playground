import sqlite3
import os

from settings import SQL_DB, USER_WORKSPACES

con = sqlite3.connect(SQL_DB)
cur = con.cursor()
cur.execute("CREATE TABLE users(username, password)")
for user, passwd in [
    ("vojta", "omilia"),
    ("test", "eloquence123")
]:
    os.makedirs(os.path.join(USER_WORKSPACES, user), exist_ok=True)
cur.execute(f"INSERT INTO users VALUES ('{user}', '{passwd}')")
con.commit()
con.close()
