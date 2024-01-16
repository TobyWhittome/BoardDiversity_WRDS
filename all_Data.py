import wrds
import pandas as pd
db = wrds.Connection(wrds_username="twhittome")

""" 
TO-DO:

-> Need to filter by only the S&P 500 companies currently.
-> Do the for loop stuff
-> Get rest of data in

 """


#Format is SELECT variable_name, FROM library.file
#All are shown in the variable descriptions section in WRDS


#Size (org summary) & Diversity
OrgSummary = pd.DataFrame(data=(db.raw_sql('SELECT Ticker, NumberDirectors, GenderRatio, NationalityMix FROM boardex.na_wrds_org_summary')))

#%Independent Directors & Shares held & CEO Duality
DirectorsUS = pd.DataFrame(data=(db.raw_sql('SELECT TICKER, CLASSIFICATION, NUM_OF_SHARES, EMPLOYMENT_CEO, EMPLOYMENT_CHAIRMAN FROM risk.rmdirectors')))

#Number of Committees
NumCommittees = pd.DataFrame(data=(db.raw_sql('SELECT TICKER, AUDIT_MEMBERSHIP, CG_MEMBERSHIP, COMP_MEMBERSHIP, NOM_MEMBERSHIP FROM risk.rmdirectors')))
#print(NumCommittees)
#for every employee on this ticker,  if audit it is not "None", then set AUD=True...... then repeat for other committees


#Meeting frequency
MeetFreq = pd.DataFrame(data=(db.raw_sql('SELECT TICKER, MEETINGDATE, MTGMONTH, YEAR FROM risk.rmgovernance')))
#print(MeetFreq)

#filter by company

#Voting Type
DualClass = pd.DataFrame(data=(db.raw_sql('SELECT TICKER, DUALCLASS FROM risk.rmgovernance')))
VoteType = pd.DataFrame(data=(db.raw_sql('SELECT TICKER, OWNLESS1, PCNT_CTRL_VOTINGPOWER FROM risk.rmdirectors')))
VotingType = pd.merge(DualClass, VoteType, on='ticker', how='inner')
#print(VotingType)


#Meeting frequency
MeetingFreq = pd.DataFrame(data=(db.raw_sql('SELECT TICKER, NUMMTGS FROM comp_execucomp.codirfin')))
print(MeetingFreq)


merged_df = pd.merge(OrgSummary, DirectorsUS, on='ticker', how='inner')
#print(merged_df)
