"""
This module defines the Airflow DAG for the Red Wine MLOps lifecycle. The DAG includes tasks
for various stages of the pipeline, including data reading, data processing, model training, 
and selecting the best model. 

The tasks are defined as functions and executed within the DAG. The execution order of the tasks 
is defined using task dependencies.

Note: The actual logic inside each task is not shown in the code, as it may reside in external 
script files.

The DAG is scheduled to run every day at 12:00 AM.

Please ensure that the necessary dependencies are installed and accessible for executing the tasks.

test
"""

from datetime import datetime
from airflow.decorators import dag, task
from kubernetes.client import models as k8s
from airflow.models import Variable

# define src path
src_path = '/git/clarus-llm-dag/src/llm_classificator'

@dag(
    description='LLMOps lifecycle',
    schedule_interval='0 12 * * *', 
    start_date=datetime(2024, 12, 12),
    catchup=False, 
    tags=['demo', 'llm'],
) 
def llm_training_dag_over_k8s():

    volume_mount = k8s.V1VolumeMount(
        name="dag-dependencies", mount_path="/git"
    )

    init_container_volume_mounts = [
        k8s.V1VolumeMount(mount_path="/git", name="dag-dependencies")
    ]
    
    volume = k8s.V1Volume(name="dag-dependencies", empty_dir=k8s.V1EmptyDirVolumeSource())

    init_container = k8s.V1Container(
        name="git-clone",
        image="alpine/git:latest",
        command=["sh", "-c", "mkdir -p /git && cd /git && git clone -b master --single-branch https://github.com/alejandrocalleja/clarus-llm-dag.git"],
        volume_mounts=init_container_volume_mounts
    )
    
    env_vars={
        "POSTGRES_USERNAME": Variable.get("POSTGRES_USERNAME"),
        "POSTGRES_PASSWORD": Variable.get("POSTGRES_PASSWORD"),
        "POSTGRES_DATABASE": Variable.get("POSTGRES_DATABASE"),
        "POSTGRES_HOST": Variable.get("POSTGRES_HOST"),
        "POSTGRES_PORT": Variable.get("POSTGRES_PORT"),
        "TRUE_CONNECTOR_EDGE_IP": Variable.get("CONNECTOR_EDGE_IP"),
        "TRUE_CONNECTOR_EDGE_PORT": Variable.get("IDS_EXTERNAL_ECC_IDS_PORT"),
        "TRUE_CONNECTOR_CLOUD_IP": Variable.get("CONNECTOR_CLOUD_IP"),
        "TRUE_CONNECTOR_CLOUD_PORT": Variable.get("IDS_PROXY_PORT"),
        "MLFLOW_ENDPOINT": Variable.get("MLFLOW_ENDPOINT"),
        "MLFLOW_TRACKING_USERNAME": Variable.get("MLFLOW_TRACKING_USERNAME"),
        "MLFLOW_TRACKING_PASSWORD": Variable.get("MLFLOW_TRACKING_PASSWORD")
    }

    # Define as many task as needed
    @task.kubernetes(
        image='clarusproject/dag-image:1.0.0-slim',
        name='read_data',
        task_id='read_data',
        namespace='airflow',
        init_containers=[init_container],
        volumes=[volume],
        volume_mounts=[volume_mount],
        do_xcom_push=True,
        env_vars=env_vars
    )
    def read_data_process_task():
        import sys
        import redis
        import uuid
        import pickle

        sys.path.insert(1, src_path)
        from Data.read_general_data import read_general_data
        from Data.read_specific_data import read_specific_data
        from Process.data_processing import data_processing
        from Process.create_dataloaders import create_dataloaders

        redis_client = redis.StrictRedis(
            host='redis-headless.redis.svc.cluster.local',
            port=6379,  # El puerto por defecto de Redis
            password='pass'
        )

        general_df = read_general_data()
        specific_df = read_specific_data()
        dp = data_processing(general_df, specific_df)
        dl = create_dataloaders(dp)

        read_id = str(uuid.uuid4())

        redis_client.set('data-' + read_id, pickle.dumps(dl))

        return read_id


    @task.kubernetes(
        image='clarusproject/dag-image:1.0.0-slim',
        name='xlNet_model_training',
        task_id='xlNet_model_training',
        namespace='airflow',
        get_logs=True,
        init_containers=[init_container],
        volumes=[volume],
        volume_mounts=[volume_mount],
        env_vars=env_vars
    )
    def xlNet_model_training_task(read_id=None):
        import sys
        import redis
        import pickle

        sys.path.insert(1, src_path)
        from Models.XLNet_model_training import xlNet_model_training

        redis_client = redis.StrictRedis(
            host='redis-headless.redis.svc.cluster.local',
            port=6379,  # El puerto por defecto de Redis
            password='pass'
        )

        data = redis_client.get('data-' + read_id)
        res = pickle.loads(data)
        return xlNet_model_training(res)

    @task.kubernetes(
        image='clarusproject/dag-image:1.0.0-slim',
        name='svc_model_training',
        task_id='svc_model_training',
        namespace='airflow',
        get_logs=True,
        init_containers=[init_container],
        volumes=[volume],
        volume_mounts=[volume_mount],
        env_vars=env_vars

    )
    def svc_model_training_result_task(read_id=None):
        import sys
        import redis
        import pickle

        sys.path.insert(1, src_path)
        from Models.SVC_model_training import svc_model_training

        redis_client = redis.StrictRedis(
            host='redis-headless.redis.svc.cluster.local',
            port=6379,  # El puerto por defecto de Redis
            password='pass'
        )

        data = redis_client.get('data-' + read_id)
        res = pickle.loads(data)

        return svc_model_training(res)
    
    @task.kubernetes(
        image='clarusproject/dag-image:1.0.0-slim',
        name='select_best_model',
        task_id='select_best_model',
        namespace='airflow',
        get_logs=True,
        init_containers=[init_container],
        volumes=[volume],
        volume_mounts=[volume_mount],
        env_vars=env_vars,
        do_xcom_push=True
    )
    def select_best_model_task(read_id):
        import redis
        import sys

        sys.path.insert(1, src_path)
        from Deployment.select_best_model import select_best_model

        redis_client = redis.StrictRedis(
            host='redis-headless.redis.svc.cluster.local',
            port=6379,  # El puerto por defecto de Redis
            password='pass'
        )

        redis_client.delete('data-' + read_id)

        return select_best_model()
    
    @task.kubernetes(
        image='clarusproject/dag-image:1.0.0-slim',
        name='register_experiment',
        task_id='register_experiment',
        namespace='airflow',
        get_logs=True,
        init_containers=[init_container],
        volumes=[volume],
        volume_mounts=[volume_mount],
        env_vars=env_vars
    )
    def register_experiment_task(best_model_res):
        import sys

        sys.path.insert(1, src_path)
        from Deployment.register_experiment import register_experiment_rds

        return register_experiment_rds(best_model_res)
    

    # Instantiate each task and define task dependencies
    processing_result = read_data_procces_task()
    elasticNet_model_training_result = elasticNet_model_training_task(processing_result)
    svc_model_training_result = svc_model_training_result_task(processing_result)
    select_best_model_result = select_best_model_task(processing_result)
    register_experiment_result = register_experiment_task(select_best_model_result)

    # Define the order of the pipeline
    processing_result >> [elasticNet_model_training_result, svc_model_training_result] >> select_best_model_result >> register_experiment_result

# Call the DAG 
redwine_training_dag_over_k8s()