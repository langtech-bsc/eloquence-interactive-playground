import sqlite3
import os

from settings import SQL_DB, USER_WORKSPACES

con = sqlite3.connect(SQL_DB)
cur = con.cursor()
cur.execute("CREATE TABLE users(username, password)")
for user, passwd in [
    ("vojta", "omilia"),
    ("test", "eloquence123"),
    ("aayushi", "test"),
    ("agnese.augello", "test"),
    ("aleix.santsavall", "test"),
    ("anastasiya", "test"),
    ("andraz", "test"),
    ("andreas", "test"),
    ("arvanitis", "test"),
    ("ben.malin", "test"),
    ("bourou", "test"),
    ("brdaric", "test"),
    ("brutti", "test"),
    ("cescolano3", "test"),
    ("cferles", "test"),
    ("dbajovic", "test"),
    ("djordje", "test"),
    ("elenif", "test"),
    ("elina", "test"),
    ("esau.villatoro", "test"),
    ("fanis", "test"),
    ("fisic", "test"),
    ("giuseppe.caggianese", "test"),
    ("h.mouratidis", "test"),
    ("helga.molbaek-steensig", "test"),
    ("info", "test"),
    ("iplchot", "test"),
    ("javier.garcia1.bsc", "test"),
    ("jordi.luque", "test"),
    ("jordi.luqueserrano", "test"),
    ("kocha78", "test"),
    ("luca.sabatucci", "test"),
    ("maite.melero", "test"),
    ("maite.melero", "test"),
    ("martin.scheinin", "test"),
    ("martin", "test"),
    ("matasso", "test"),
    ("nikolaos.boulgouris", "test"),
    ("nikolasimic", "test"),
    ("nstylianou", "test"),
    ("petr.motlicek", "test"),
    ("pietro.neroni", "test"),
    ("r.shekhar", "test"),
    ("secujski", "test"),
    ("sergio.burdisso", "test"),
    ("sinisa.suzic", "test"),
    ("tadic", "test"),
    ("tatiana.kalganova", "test"),
    ("tijana.nosek", "test"),
    ("tstafylakis", "test"),
    ("urska", "test"),
    ("vlado.delic", "test"),
    ("vukst", "test"),
    ("yuchen.zhang", "test"),
    ("yulia", "test")
]:
    os.makedirs(os.path.join(USER_WORKSPACES, user), exist_ok=True)
cur.execute(f"INSERT INTO users VALUES ('{user}', '{passwd}')")
con.commit()
con.close()
