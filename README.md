# APIDeploymentOfBostonHousePrediction
Deploying Boston House Prediction as an API on AWS

Birds view of step involved in creation of ML model as REST API

1) Created sklearn pipeline for imputation, scaling and OHE involving categorical and numerical attributes

2) Perfomed hyperparameter tuning using gridsearch cv and saved the model as pickle life

3) Loading the saved model in predict.py and performing predictions on incoming test data. This is wrapped with Flask microweb framework for python

4) Used gunicorn to get production ready webserver capabilities

5) Develop dockerfile for the application and build the image

6) Run the image locally to test your application within docker container.

7) On successful run, push the image to your dockerhub account.

8) Setup aws account and develop json file, which specifies the image to be pulled from dockerhub

9) Start Amazon Beanstalk service (which hosts nginx by default) and upload you json file.

10) Amazon will throw the url and use postman to test you REST API.

Inspired from https://towardsdatascience.com/deploy-a-machine-learning-model-as-an-api-on-aws-43e92d08d05b

Happy deploying !!!
