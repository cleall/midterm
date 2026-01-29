# Cpu turbo frequency predictor

I wanted to practice Multiple Linear Regression and Random Forests Machine
Learning models as I had curiosity to know how I could use them and compare
their results as well as spot some of their differences in the process

I started with a dataset that had cpu, gpu data, basic video settings to play a few games and the achieved fps on each game with each configuration.
I spent a lot of time investigating about Multiple Linear Regression, cpu, gpu features that I had to select a subset of the dataset as I was not going to finish on time so I chose the cpu data

What I want to accomplish is to estimate the potential turbo frequency of a cpu given some of its features, the ones that you can find over the internet on each manufacturer site, a datasheet or data from enthusiast blogs.
So using features like core count, tdp, process node and so on use them to predict potential cpu turbo frequency

This way you can have a starting point to compare cpus only by looking at their potential turbo frequency.
Please keep in mind that in modern cpu industry there are features that are not considered in the dataset but this could serve as a starting point to gather useful and significant features from modern designs and architectures
which are used for different purposes not only for gaming

## Notebooks

Go to data folder and decompress the zip file make sure the name of unzipped file is fps_videogames.csv

I uploaded two notebooks to : https://github.com/cleall/ml_zmcmp_hwwrk/tree/prime/midterm_07

#### Random Forests
```
    midterm_07_dt_optn_cpu_units_pl contains all the steps to create the model deployed to Render
```

#### Multiple Linear Regression
```
    midterm_mlr_cpu_turbo_pl contains all the steps to create the Linear Regression model 
```
They contain the steps I followed to do EDA, create models, tune them and evaluate them.

## System prerequisites

Python3, version 3.10.12

```
    https://www.python.org/downloads/
```

uv, version 0.9.5

```
    https://docs.astral.sh/uv/getting-started/installation/
```

Docker

```
    https://docs.docker.com/engine/install/
```

Available local port (Optional)

In case you want to test on your system check that you have port 4444 available otherwise you have to edit client.py, predict.py and Dockerfile to match your system configuration with the port of your choice


## It works on my machine ¯\\_(ツ)_/¯ (Run Locally)

Create a folder to clone there

```bash
    mkdir midterm
  
    cd midterm
```

Clone project to midterm/

```bash
    git clone https://github.com/cleall/midterm.git
```

Navigate to project folder

```bash
    cd midterm
```

Install python version for project using uv

```bash
    uv python install 3.10
```

Create venv using previously installed python version

```bash
    uv venv --python 3.10
```

Activate the environment

```bash
    source .venv/bin/activate
```

Install project dependencies

```bash
    uv sync --locked
```

## Local usage from uv

#### Train model and save it
Go to data folder and decompress the zip file make sure the name of unzipped file is fps_videogames.csv

Go to midterm/ and do

```bash
    uv run -m train_pipeline.train
    .
    .
    Saved pipeline check cwd for: cpu_tf_rfpl_v1.bin
```
The previous command trains the model and saves it to the root of midterm. Model is included with the project so in case it does not work (for whatever reason) you still have it available

#### Start server to listen to make model available
Go to midterm/ and do

```bash
    uv run predict.py
    .
    .
    INFO:     Uvicorn running on http://0.0.0.0:4444 (Press CTRL+C to quit)
```
The previous command uses uv to start uvicorn for all interfaces in specified port

When you want to stop the server simply do CTRL+C

#### Send a request using client.py
Go to midterm/ and do

```bash
    uv run predict.py
    .
    .
    INFO:     Uvicorn running on http://0.0.0.0:4444 (Press CTRL+C to quit)
```
The previous command sends a request to test the model, by default it points to the local machine.

In case you want to try out other parameters you can use the form accessing the localhost:4444 and below you can see the same features used in [client.py](client.py) just for guidance.

```python
    request = {
        "cpu_cores": 6,
        "cpu_threads": 12,
        "cpu_frq": 3.6,
        "cpu_multiplier": 35,
        "cpu_tdp": 65,
        "cpu_prcss": 14,
        "cpu_die": 160,
        "cpu_has_oc": "N"
    }
```
Once form has been filled click the Predict Cpu Turbo Frequency button.

## Local usage from Docker

Make sure you are not running the uvicorn server before you create the Dockerimage

To create Dockerimage go to midterm/ and do

```bash
    sudo docker build -t midterm .
```

The previous command creates the Dockerimage midterm:latest

To run Dockerimage go to midterm/ and do

```bash
    sudo docker run -p 4444:4444 --env PORT=4444 midterm
```

The previous command runs the Dockerimage midterm:latest in interactive mode

When you want to stop the server simply do CTRL+C


## Test Docker image

#### Send a request using client.py
Go to midterm/ and do

```bash
    uv run client.py
```
The previous command sends a request to test the model, by default it points to the local machine, this time deployed using the docker image

#### Send a request using html form

In case you want to try out other parameters you can use the form accessing the localhost:4444 and below you can see the same features used in [client.py](client.py) just for guidance.

```python
    request = {
        "cpu_cores": 6,
        "cpu_threads": 12,
        "cpu_frq": 3.6,
        "cpu_multiplier": 35,
        "cpu_tdp": 65,
        "cpu_prcss": 14,
        "cpu_die": 160,
        "cpu_has_oc": "N"
    }
```

Once form has been filled click the Predict Cpu Turbo Frequency button.

# Cloud Deployment

Deployed to Render (access html form): https://predict-cpu-turbo-frequency.onrender.com

When you access the url it loads the form ready to use as in the local tests
at this point you can choose either form or [client.py](client.py) file.

#### Send a request using client.py to deployed service

Send post requests: https://predict-cpu-turbo-frequency.onrender.com/predict_cpu_tf

Go to midterm/ and edit [client.py](client.py)

```python
    #Uncomment url to point to Deployed service on Render and comment the url for local tests

    url = "https://predict-cpu-turbo-frequency.onrender.com/predict_cpu_tf"

    Save and do
    
    uv run client.py
```
The previous command sends a request to test the model to the deployed service on render.

#### Video interacting with Render (first deployment)

I recoded a small video interacting with Render and running it locally.

Video: [Deployment and local test](https://www.youtube.com/watch?v=5FRstzQELxU)

#### Video interacting with Render (rev1)

I recoded a small video interacting with Render after rev1.

Video: [Render interaction rev1](https://www.youtube.com/watch?v=TY3g0jDsu-w)

#### Deployment to Render

In my first deployment I uploaded docker image to DockerHub and pulled the image from there to render.

This time I pushed the docker image to Github Container Registry (ghcr) and pulled the image from there to render.

I liked the ghcr over the dockerhub because the docker image now requires an access token with read permissions only to be pulled by render, instead of having the image publicly available in dockerhub where surprisingly it is downloaded almost immediately either automated systems or people who like to take advantage of anything at their reach.

The only thing that is a bit misleading in ghcr are the statistics of the package as when render first establish a connection and then deploys the service, each time, it pulls the image and that counts as a download, so if the service restarts or is re-deployed by any other reason that is going to increase the download count.

Check this for steps: [Deploy Docker image to render](https://render.com/docs/deploying-an-image)

Thanks for reviewing my project.