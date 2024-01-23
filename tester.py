import wrds
db = wrds.Connection(wrds_username="twhittome")

print(db.list_libraries())

data_5rows = db.get_table(library='boardex', table='na_wrds_org_summary', columns = ['Ticker', 'AnnualReportdate'], obs=5)
print(data_5rows)
#OrgSummary = db.raw_sql("SELECT Ticker, Annualreportdate FROM boardex.na_wrds_org_summary")
#print(OrgSummary)

#query = "SELECT Ticker, Annualreportdate FROM boardex.na_wrds_org_summary"
#OrgSummary = db.get_table(library='boardex', table='na_wrds_org_summary', obs=10)


#query = "SELECT Ticker, NumberDirectors, GenderRatio, NationalityMix, Annualreportdate FROM boardex.na_wrds_org_summary"
#print(ps.sqldf(query, locals()))

#print(db.raw_sql("SELECT Ticker, NumberDirectors, GenderRatio, NationalityMix, Annualreportdate FROM boardex.na_wrds_org_summary"))

#con = wrds.connect('courses_database') 
#OrgSummary = pd.read_sql_query(f"SELECT Ticker, NumberDirectors, GenderRatio, NationalityMix, Annualreportdate FROM boardex.na_wrds_org_summary", con)