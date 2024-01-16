import wrds
db = wrds.Connection(wrds_username="twhittome")
#print(db.list_libraries())
print(db.list_tables('comp_execucomp'))
