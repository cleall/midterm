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
The previous command sends a request to test the model

#### Send a request using dashboard
You can use

127.0.0.1:4444/docs

0.0.0.0:4444/docs

localhost:4444/docs

just to name a few to load the dashboard using your preferred browser
```
    Look for the POST button, go to the end of the line and click the arrow pointing down
    
    Click the Try it out button

    Set values of each attribute, do not remove curly braces e.g.

    {
        "cpu_cores": 4,
        "cpu_threads": 8,
        "cpu_frq": 3.5,
        "cpu_clock": 0.1,
        "cpu_multiplier": 35,
        "cpu_tdp": 77,
        "cpu_prcss": 22,
        "cpu_die": 160,
        "cpu_has_oc": "N"
    }

    Brief explanation
        cpu_cores, cpu_threads, cpu_multiplier, cpu_tdp, cpu_prcss are int positive values
        cpu_frq, cpu_clock, are int positive values represented in Ghz
        cpu_die is int positive value represented in mm²
        cpu_has_oc is a str "Y" or "N" indicates if multiplier is unlocked or not
    
    Once ready click the Execute button

    Below you can review the server response e.g.

    {
        "turbo_frequency": 3.8850742063492056
    }
```

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


## Test Dockerimage

#### Send a request using client.py
Go to midterm/ and do

```bash
    uv run client.py
```
The previous command sends a request to test the model

#### Send a request using dashboard
You can use

127.0.0.1:4444/docs

0.0.0.0:4444/docs

localhost:4444/docs

just to name a few to load the dashboard using your preferred browser
```
    Look for the POST button, go to the end of the line and click the arrow pointing down
    
    Click the Try it out button

    Set values of each attribute, do not remove curly braces e.g.

    {
        "cpu_cores": 4,
        "cpu_threads": 8,
        "cpu_frq": 3.5,
        "cpu_clock": 0.1,
        "cpu_multiplier": 35,
        "cpu_tdp": 77,
        "cpu_prcss": 22,
        "cpu_die": 160,
        "cpu_has_oc": "N"
    }

    Brief explanation
        cpu_cores, cpu_threads, cpu_multiplier, cpu_tdp, cpu_prcss are int positive values
        cpu_frq, cpu_clock, are int positive values represented in Ghz
        cpu_die is int positive value represented in mm²
        cpu_has_oc is a str "Y" or "N" indicates if multiplier is unlocked or not
    
    Once ready click the Execute button

    Below you can review the server response e.g.

    {
        "turbo_frequency": 3.8850742063492056
    }
```
You sent a request using the dashboard or docs


# Cloud Deployment

Deployed to Render: https://predict-cpu-turbo-frequency.onrender.com/predict_cpu_tf_rf

Note: It is going to be running when peer review starts for a few days it is going to be suspended

#### Send a request using client.py
Go to midterm/ and edit client.py

```bash
    Change url to point to Deployed service on Render

    url = "https://predict-cpu-turbo-frequency.onrender.com/predict_cpu_tf_rf"

    Save and do
    
    uv run client.py
```
The previous command sends a request to test the model

#### Send a request using dashboard
Go to: https://predict-cpu-turbo-frequency.onrender.com/predict_cpu_tf_rf/docs
```
    Look for the POST button, go to the end of the line and click the arrow pointing down
    
    Click the Try it out button

    Set values of each attribute, do not remove curly braces e.g.

    {
        "cpu_cores": 4,
        "cpu_threads": 8,
        "cpu_frq": 3.5,
        "cpu_clock": 0.1,
        "cpu_multiplier": 35,
        "cpu_tdp": 77,
        "cpu_prcss": 22,
        "cpu_die": 160,
        "cpu_has_oc": "N"
    }

    Brief explanation
        cpu_cores, cpu_threads, cpu_multiplier, cpu_tdp, cpu_prcss are int positive values
        cpu_frq, cpu_clock, are int positive values represented in Ghz
        cpu_die is int positive value represented in mm²
        cpu_has_oc is a str "Y" or "N" indicates if multiplier is unlocked or not
    
    Once ready click the Execute button

    Below you can review the server response e.g.

    {
        "turbo_frequency": 3.8850742063492056
    }
```
You sent a request using the dashboard or docs

#### Video interacting with Render

I recoded a small video interacting with Render and running it locally

Video: [Deployment and local test](https://www.youtube.com/watch?v=5FRstzQELxU)
