# BoardDiversity_WRDS

This repository contains the python scripts, collecting data from Wharton Research Data Services (WRDS) to retrive the corporate governance board structure variables and perform Multi-Criteria Decision Analysis, Factor Analysis and Cluster Analysis on them.

These scripts have also been automated through an Apache Airflow data pipeline, created by Yong Chen.

Project Structure:

Each file contains the relevant analyses, where DataCreation contains the python scripts to fetch and pre-process the dataset.
- MCDA performs the Multi-Criteria Decision Analysis
- Clustering performs both K-Means and Time-Series clustering
- FactorAnalysis performs the Factor Analysis
- Other Function includes scripts for the Linear regression analyses and clustering using the factors.


Dependencies:

Please ensure you have Python 3.10 or above installed.

Please ensure that you have major dependencies installed at specified below:
'''
Pandas: 1.2
MatPlotLib: 1.2
etc.

'''
Please refer to the example requirements.txt file for a full list of install dependencies.


Running the python scripts:

Simply navigate to the parent directory and run
'''
python DataCreation/create_dataset.py
'''
or 
'''
python3 DataCreation/create_dataset.py
'''

Then you can run the MCDA, Clustering or Factor Analysis scripts using the same format.