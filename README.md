# fakeNewsClassifier
This repository uses tensorflow and keras for fake news classification

# Project template

To setup the project structure, we create a python file called templates.py.
In this file we have to setup two things:

1. Basic logging string to capture INFO level messages
2. Setup the project name
3. Create a list with the name of files and folders that needs to be created for the complete project 

![Alt text](<Screenshot 2023-12-04 at 2.20.46 PM.png>)

We created .gitkeep file because we need atleast one file to be created in our project setup. Once we setup our deployment pipeline, we will delete that file.

# Setting environment and setup.py file

1. Create requirements.txt file
2. Update the ReadMe.md file
3. Modify the setup.py file so that we can deploy this project as package

# Setting a custom logger and exception.

In the textClassifier, we will create the custom logger. The logger will be created in __init__.py file This will able to capture all the logs that will be generated while running our project.

Now we will create our exception. For this we will be using box-exception library and this library will help us in creating the exception errors that might come across while executing the project. This will be created in the utils folder in common.py file.

# Project workflow
In this section, we will highlight the file that will be updated regularly in our project.

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the dvc.yaml
10. app.py

# Data ingestion

We are going to take the data and compress it and store it in google drive. Post that we will use gdown library in python, download it on our project and then unzip the file. We will then going to create an artifacts folder and then create the data ingestion pipeline.

# Prepare base model

After working on the data ingestion, prepare the model and save it in h5 model format in the artifacts folder.

Follow the same project workflow.

![Alt text](<Screenshot 2023-12-19 at 4.50.31 PM.png>)

# Model training

We create a model training module that will read the data and train the model that we created in the previous step. The trained model is then saved in the artifacts folder, inside training folder.

# Model Training and evaluation through MLFLow

MLflow is an open source platform for managing the end-to-end machine learning lifecycle. We can use it to track text classification models.

First, log the text classification model as an MLflow artifact when training the model. Then, load the logged model in the MLflow model registry and deploy it to make predictions.

Finally, monitor the deployed model performance using the MLflow UI and retrain models as needed to maintain accuracy over time.

# Pipeline tracking with DVC

1. Install dvc
2. Install dvc extension on VS Code
3. in the terminal type, dvc init to initiate the project
4. Prepare the YAML file
5. Execute dvc repro
