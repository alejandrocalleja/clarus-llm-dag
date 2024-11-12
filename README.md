**Table of Contents**

1. Entrenamiento en Jupyter
2. Crear el repositorio base en GitHub
3. Crear el DAG


# 1 Entrenamiento en Jupyter
El proyecto parte de la idea de un científico de datos que trabaja en el entrenamiento de LLMs. En esta primera fase, el entorno más común para el entrenamiento son los notebooks de Jupyter. En el ejemplo actual se querrá **entrenar una XLNet para clasificación**.

[NLP_Clasificador.ipynb](https://github.com/alejandrocalleja/clarus-llm-dag/blob/master/src/NLP_Clasificador.ipynb "NLP_Clasificador.ipynb")


# 2 Crear el repositorio base en GitHub

El siguiente es un ejemplo de repositorio para la creación del DAG en CLARUS. Este repositorio debe mantener un orden muy concreto para que todas las automatizaciones funcionen correctamente.

```
├── dags
│ ├── XX_XX_dag_k8s.py
├── src
│ ├── Data
│ ├── Models
│ ├── Process
│ ├── config.py
│ ├── Dockerfile
│ ├── requirements.txt
```

Para este caso de uso, partiendo del Jupyter notebook se han creado las siguientes carpetas y archivos `.py`. 

```
├── dags
│ ├── llm_training_dag_k8s.py
├── src
│ ├── Data
│ | ├── general_questions.csv
│ | ├── specific_questions.csv
│ | ├── read_data.py
│ ├── Models
│ | ├── utils.py
│ | ├── XLNet_model_training.py
│ ├── modules
│ | ├── __init__.py
│ | ├── questions_dataset.py
│ ├── Process
│ | ├── create_dataloaders.py
│ | ├── data_processing.py
│ ├── config.py
│ ├── Dockerfile
│ ├── requirements.txt
```

El flujo que después será ejecutado por el DAG, es el siguiente:

```
read_data.py >> data_processing.py >> 
create_dataloaders.py >> XLNet_model_training.py
```

El resto de archivos serán utilizados por algunos de los principales mostrados en el flujo anterior.


# 3 Crear el DAG

En este casó, se crearán 2 tareas:
- **read_data**
	- Se leerán los datos de los 2 CSV y se generarán Pandas Dataframes.
	- Se procesarán los Dataframes y se transformarán en preguntas de entrenamiento y evaluación (dividido entre preguntas y categoría)
- **xlNet_model_training**
	- Se crearán los Dataloaders. Es decir, en este paso los datos de entrenamiento y evaluación se transforman para poder ser usados correctamente en la XLNet.
	- Se procederá a entrenar la XLNet y guardar los datos en MLFlow.


> [!WARNING] Alternativa
> Es posible que se puedan dividir las 2 tareas previas en más subtareas. Sin embargo, habría que realizar diferentes pruebas, ya que en un primer momento la generación de los Dataloaders en la tarea de lectura de datos y la posterior transferencia mediante Redis era problemática.

---

> [!ERROR] AVISO
> El Dockerfile que se utilice deberá incluir las librerías necesarias, además de la versión de Python 3.11.9 y la instalación de sus `requirements.txt`.
> 
> Esta imagen de docker deberá estar en un repositorio público y accesible desde Airflow en todo momento. Esta imagen se deberá especificar en cada tarea creada.



