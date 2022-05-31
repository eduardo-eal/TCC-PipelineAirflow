from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.dummy import DummyOperator
from airflow.operators.bash_operator import BashOperator
from airflow.utils.task_group import TaskGroup

pathScript = "~/airflow/dags/tcc_scripts"
pathTrue =  "~/airflow/dags/tcc_scripts/featurestore/True.csv"
pathFake = "~/airflow/dags/tcc_scripts/featurestore/Fake.csv"

default_args = {
   'owner': 'teste',
   'depends_on_past': False,
   'start_date': datetime(2019, 1, 1),
   'retries': 0,
   }

with DAG(
   'dag-tcc-Eduardo-Amaral-Lopes',
   schedule_interval=timedelta(minutes=60),
   catchup=False,
   default_args=default_args
   ) as dag:

    start = DummyOperator(task_id="start")
    
    with TaskGroup("preProcessar", tooltip="preProcessar") as preProcessar:
        t1 = BashOperator(
            dag=dag,
            task_id='preProcessar',
            bash_command="""
            cd {0}
            python tcc_preprocessing.py
            """.format(pathScript)
        )
        [t1]

    with TaskGroup("treinar_modelo", tooltip="treinar_modelo") as treinar_modelo:
        t2 = BashOperator(
            dag=dag,
            task_id='treinar',
            bash_command="""
            cd {0}
            python tcc_modelo.py {1} {2} {3} {4} {5} {6}
            """.format(pathScript, 6, 100, 0.8, 0.8, 10, 0.3)
        )
        [t2]

    with TaskGroup("selecionar_melhor", tooltip="selecionar_melhor") as selecionar_melhor:
        t3 = BashOperator(
            dag=dag,
            task_id='selecionar',
            bash_command="""
            cd {0}
            python tcc_validateModel.py
            """.format(pathScript)
        )
        [t3]

    end = DummyOperator(task_id='end')

    start >> preProcessar >> treinar_modelo >> selecionar_melhor >> end
