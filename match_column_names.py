import sqlite3

con = sqlite3.connect(r'C:\Users\Luca\PycharmProjects\IntroToAI-Group5-TeamB(football)\database.sqlite')
cur = con.cursor()

cur.execute("SELECT * from Match")

names = [description[0] for description in cur.description]

print(names)