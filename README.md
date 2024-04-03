# BoardDiversity_WRDS

This repository contains the python scripts to collect data from Wharton Research Data Services (WRDS). It retrives the board structure variables and performs Multi-Criteria Decision Analysis, Factor Analysis and Cluster Analysis on them.
These scripts have also been automated through an Apache Airflow data pipeline, created by Yong Chen.

Project Structure:
Each file contains the relevant analyses, where the
- DataCreation file contains the python scripts to fetch and pre-process the dataset.
- MCDA performs the Multi-Criteria Decision Analysis
- Clustering performs both K-Means and Time-Series clustering
- FactorAnalysis performs the Factor Analysis
- Other Functions includes scripts for the Linear regression analyses and clustering using the factors.

Dependencies:
Please ensure that you have major dependencies installed, these can be found in the requirements.txt file

Running the python scripts:
Simply navigate to the parent directory and run: python DataCreation/create_dataset.py or python3 DataCreation/create_dataset.py

Then you can run the MCDA, Clustering or Factor Analysis scripts using the same format.
