import sqlite3

from settings import SQL_DB

con = sqlite3.connect(SQL_DB)
cur = con.cursor()
cur.execute("CREATE TABLE users(username, password)")
cur.execute("INSERT INTO users VALUES ('vojta', 'omilia')")
con.commit()
con.close()