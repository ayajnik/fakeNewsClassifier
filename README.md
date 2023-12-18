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


