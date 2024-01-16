import wrds
db = wrds.Connection(wrds_username="twhittome")
#print(db.list_libraries())
print(db.raw_sql('SELECT date, dji FROM djones.djdaily'))