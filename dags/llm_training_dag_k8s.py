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


@dag(
    description="LLMOps lifecycle",
    schedule_interval="0 12 * * *",
    start_date=datetime(2022, 1, 1),
    catchup=False,
    tags=["demo", "llm"],
)
def llm_training_dag_over_k8s():

    env_vars = {
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
        "MLFLOW_TRACKING_PASSWORD": Variable.get("MLFLOW_TRACKING_PASSWORD"),
    }

    volume_mount = k8s.V1VolumeMount(name="dag-dependencies", mount_path="/git")

    init_container_volume_mounts = [k8s.V1VolumeMount(mount_path="/git", name="dag-dependencies")]

    volume = k8s.V1Volume(name="dag-dependencies", empty_dir=k8s.V1EmptyDirVolumeSource())

    init_container = k8s.V1Container(
        name="git-clone",
        image="alpine/git:latest",
        command=[
            "sh",
            "-c",
            "mkdir -p /git && cd /git && git clone -b master --single-branch https://github.com/alejandrocalleja/clarus-llm-dag.git",
        ],
        volume_mounts=init_container_volume_mounts,
    )

    pod_spec = k8s.V1Pod(
        api_version="v1",
        kind="Pod",
        spec=k8s.V1PodSpec(
            runtime_class_name="nvidia",  # Establecer runtimeClassName a 'nvidia'
            containers=[
                k8s.V1Container(
                    name="base",
                )
            ],
        ),
    )

    # Define as many task as needed
    @task.kubernetes(
        image="alejandrocalleja/xlnet-dag:latest",
        name="read_data",
        task_id="read_data",
        namespace="airflow",
        init_containers=[init_container],
        volumes=[volume],
        volume_mounts=[volume_mount],
        do_xcom_push=True,
        env_vars=env_vars,
    )
    def read_data_process_task():
        import sys
        import redis
        import uuid
        import pickle

        sys.path.insert(1, "/git/clarus-llm-dag/src/llm_classificator")
        from Data.read_data import read_data
        from Process.data_processing import data_processing
        from Process.create_dataloaders import create_dataloaders

        redis_client = redis.StrictRedis(
            host="redis-headless.redis.svc.cluster.local",
            port=6379,  # El puerto por defecto de Redis
            password="pass",
        )

        general_df, specific_df = read_data()
        dp = data_processing(general_df, specific_df)
        dl = create_dataloaders(dp)
        print(xlNet_model_training_task(dl))
        read_id = str(uuid.uuid4())

        redis_client.set("data-" + read_id, pickle.dumps(dl))

        return read_id

    @task.kubernetes(
        image="alejandrocalleja/xlnet-dag:latest",
        name="xlNet_model_training",
        task_id="xlNet_model_training",
        namespace="airflow",
        init_containers=[init_container],
        image_pull_policy="Always",
        volumes=[volume],
        volume_mounts=[volume_mount],
        full_pod_spec=pod_spec,
        do_xcom_push=True,
        container_resources=k8s.V1ResourceRequirements(
            requests={"cpu": "1", "nvidia.com/gpu": "1"}, limits={"cpu": "1.5", "nvidia.com/gpu": "1"}
        ),
        priority_class_name="medium-priority",
        env_vars=env_vars,
    )
    def xlNet_model_training_task(read_id=None):
        import sys
        import redis
        import pickle

        sys.path.insert(1, "/git/clarus-llm-dag/src/llm_classificator")
        from Models.XLNet_model_training import xlNet_model_training

        redis_client = redis.StrictRedis(
            host="redis-headless.redis.svc.cluster.local",
            port=6379,  # El puerto por defecto de Redis
            password="pass",
        )

        data = redis_client.get("data-" + read_id)
        res = pickle.loads(data)
        redis_client.delete("data-" + read_id)

        return xlNet_model_training(res)

    # Instantiate each task and define task dependencies
    processing_result = read_data_process_task()
    xlNet_model_training_result = xlNet_model_training_task(processing_result)

    # Define the order of the pipeline
    (processing_result >> xlNet_model_training_result)


# Call the DAG
llm_training_dag_over_k8s()
