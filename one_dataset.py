import wrds
import pandas as pd
db = wrds.Connection(wrds_username="twhittome")


#Size (org summary) & Diversity
OrgSummary = pd.DataFrame(data=(db.raw_sql('SELECT Ticker, NumberDirectors, GenderRatio, NationalityMix FROM boardex.na_wrds_org_summary')))

#%Independent Directors & Shares held & CEO Duality & Number of committees & Voting type
DirectorsUS = pd.DataFrame(data=(db.raw_sql('SELECT TICKER, CLASSIFICATION, NUM_OF_SHARES, EMPLOYMENT_CEO, EMPLOYMENT_CHAIRMAN, AUDIT_MEMBERSHIP, CG_MEMBERSHIP, COMP_MEMBERSHIP, NOM_MEMBERSHIP, OWNLESS1, PCNT_CTRL_VOTINGPOWER FROM risk.rmdirectors')))

Governance = pd.DataFrame(data=(db.raw_sql('SELECT TICKER, MEETINGDATE, MTGMONTH, YEAR, DUALCLASS FROM risk.rmgovernance')))

#Meeting frequency
Execucomp = pd.DataFrame(data=(db.raw_sql('SELECT TICKER, NUMMTGS, YEAR FROM comp_execucomp.codirfin')))


#merged_df1 = pd.merge(OrgSummary, DirectorsUS, on='ticker', how='inner')
#merged_df2 = pd.merge(Governance, Execucomp, on='ticker', how='inner')
#merged_df3 = pd.merge(merged_df1, merged_df2, on='ticker', how='inner')
#print(merged_df3)
