# Car Price Prediction - Building ML Pipeline using Apache Airflow

I built an ML pipeline to automate the workflow of my Car Price Prediction (which makes predict Car Prices using Linear, Polynomial, Lasso and Support Vector Regressions to compare the results of each and choose the best model) task by using Apache AirFlow which is a platform that helps to programmatically author, schedule and monitor workflows.

Once you have Apache Airflow installed on your machine, you can simply start it by using the following commands,

### In one terminal
- airflow db init
- airflow webserver

### In another terminal 
- airflow scheduler

Then you can connect to the port which is the default localhost:8080 that can be changed from the airflow.cfg file.

After you see the main home page of Apache Airflow, you should be sure that you have a dags file in the airflow folder which should contain your pipeline file (which is ml_regression_pipeline.py in my case), I also have a variables_config file which is also imported in the Admin > Variables part on Apache Airflow.

![dags](https://user-images.githubusercontent.com/84930400/130382829-c2d17405-de4f-4102-bff8-766b7e736dd0.PNG)

Then you can start/run your dag and monitor the process by clicking on the ml_regression_pipeline dag, there are many options to monitor the progress such as tree view and graph view:

## Tree View

![tree_view](https://user-images.githubusercontent.com/84930400/130381800-cf80e05b-a12e-4e8a-90e6-6af1c3afc6e6.PNG)

## Graph View

![graph_view](https://user-images.githubusercontent.com/84930400/130381803-f3c81f28-92db-4253-bc1c-b3a28ca82f10.PNG)

If one of your job fails in your pipeline you can simply check the logs and see where the pipeline fails and fix the issues/bugs from the .py file,

![log](https://user-images.githubusercontent.com/84930400/130383267-19017a89-8ca3-4e04-baeb-dd66359a2daf.PNG)

As a result, my pipeline was successful and I received the output as I expected on the Apache Airflow as well!

![output](https://user-images.githubusercontent.com/84930400/130383833-3832eef0-7509-4d44-9f63-fd475e542818.PNG)

- ASA

