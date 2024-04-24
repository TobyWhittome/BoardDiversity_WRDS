from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

# Define the default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
    'retries': 1,
}

# Define the DAG
with DAG(
    'data_processing_pipeline',
    default_args=default_args,
    description='A data processing pipeline with multiple steps',
    schedule_interval=None,
    catchup=False,
) as dag:
    
    # Task 1: Data Pull
    data_pull = DummyOperator(
        task_id='data_pull'
    )

    # Task 2: Cleaning and pre-processing steps
    cleaning_preprocessing = DummyOperator(
        task_id='cleaning_preprocessing'
    )

    # Task 3: Combine into a single dataset
    combine_dataset = DummyOperator(
        task_id='combine_dataset'
    )

    # Task 4: Output dataset in a single CSV
    output_csv = DummyOperator(
        task_id='output_csv'
    )

    # Task 5: Perform MCDA (Multi-Criteria Decision Analysis)
    perform_mcda = DummyOperator(
        task_id='perform_mcda'
    )

    # Task 6: Perform Factor Analysis
    perform_factor_analysis = DummyOperator(
        task_id='perform_factor_analysis'
    )

    # Task 7: Perform Cluster Analysis
    perform_cluster_analysis = DummyOperator(
        task_id='perform_cluster_analysis'
    )

    # Task 8: Graphs and Results Output
    results_output = DummyOperator(
        task_id='results_output'
    )

    # Define the flow of the DAG
    data_pull >> cleaning_preprocessing >> combine_dataset >> output_csv

    output_csv >> perform_mcda
    output_csv >> perform_factor_analysis
    output_csv >> perform_cluster_analysis

    perform_mcda >> results_output
    perform_factor_analysis >> results_output
    perform_cluster_analysis >> results_output
