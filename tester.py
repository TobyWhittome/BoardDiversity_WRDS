import wrds
db = wrds.Connection(wrds_username="twhittome")

#print(db.list_libraries())


OrgSummary = db.raw_sql("SELECT Ticker, Annualreportdate FROM boardex.na_wrds_org_summary")
print(OrgSummary)






#data_5rows = db.get_table(library='boardex', table='na_wrds_org_summary', columns = ['Ticker', 'AnnualReportdate'], obs=5)
#print(data_5rows)

#query = "SELECT Ticker, Annualreportdate FROM boardex.na_wrds_org_summary"
#OrgSummary = db.get_table(library='boardex', table='na_wrds_org_summary', obs=10)


#query = "SELECT Ticker, NumberDirectors, GenderRatio, NationalityMix, Annualreportdate FROM boardex.na_wrds_org_summary"
#print(ps.sqldf(query, locals()))

#print(db.raw_sql("SELECT Ticker, NumberDirectors, GenderRatio, NationalityMix, Annualreportdate FROM boardex.na_wrds_org_summary"))

#con = wrds.connect('courses_database') 
#OrgSummary = pd.read_sql_query(f"SELECT Ticker, NumberDirectors, GenderRatio, NationalityMix, Annualreportdate FROM boardex.na_wrds_org_summary", con)


""" dataframe = pd.DataFrame(data=(self.db.raw_sql(f"SELECT TICKER, CLASSIFICATION, NUM_OF_SHARES, OWNLESS1, PCNT_CTRL_VOTINGPOWER FROM risk.rmdirectors WHERE YEAR BETWEEN '{self.lastyear}' AND '{self.thisyear}' AND TICKER IN {self.SP500List}")))
#Count if any directors have above ...% voting power. Director holds <1% Voting Power -- is ownless -- could use either
voting_power = dataframe.groupby('ticker')['pcnt_ctrl_votingpower'].apply(lambda x: (x >= 10).sum()).reset_index(name='high_voting_power')

#Count number of independent directors using CLASSIFICATION and make a percentage. -- Board affiliation (E-employee/insider; I-Independent; L-linked; NA-not ascertainable) (classification) -- I-NED = Independent Non-Exec director
num_independent_directors = dataframe.groupby('ticker')['classification'].apply(lambda x: round((x == ('I-NED' or 'I' or 'NI-NED')).sum() / len(x) *100, 1)).reset_index(name='percentage_NEDs')
 """