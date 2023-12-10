import wrds
db = wrds.Connection(wrds_username="twhittome")
print(db.list_libraries())