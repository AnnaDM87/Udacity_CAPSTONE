*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Employee Retain

This project is done as a part of Udacity's 'Machine Learning Engineer with Microsoft Azure' nanodegree course. I have to train, deploy and consume the model endpoint. I have used Azure Machine Learning SDK for Python to build and run machine learning workflows. Training the dataset is done using 2 methods:

    * Optimize the hyperparameters of a standard Scikit-learn Logistic Regression using HyperDrive.
    * AutoML run.


### Overview
I used the Employee dataset provided by kaggle. https://www.kaggle.com/datasets/tawfikelmetwally/employee-dataset
The Dataset dataset contains information about employees in a company, including their educational backgrounds, work history, demographics, and employment-related factors. It has been anonymized to protect privacy while still providing valuable insights into the workforce.

#### Columns:

Education: The educational qualifications of employees, including degree, institution, and field of study.

Joining Year: The year each employee joined the company, indicating their length of service.

City: The location or city where each employee is based or works.

Payment Tier: Categorization of employees into different salary tiers.

Age: The age of each employee, providing demographic insights.

Gender: Gender identity of employees, promoting diversity analysis.

Ever Benched: Indicates if an employee has ever been temporarily without assigned work.

Experience in Current Domain: The number of years of experience employees have in their current field.

Leave or Not: a target column

### Task
TThe dataset was used for supervised learning binary classification with LeaveOrNot column being the target. It is a boolean value:
* 0: the employee left the company
* 1: the empoyee still works for the company
The aim is train a model to predict if a employee will leave or not the company. 

### Access
To easily access the dataset and have control on it, I uploaded the dataset in my github account. In both notebooks the dataset was read by using pandas fucntion " read_csv()"
[Employee Raw Data](https://raw.githubusercontent.com/AnnaDM87/Udacity_CAPSTONE/main/starter_file/Employee.csv)

#### AutoML Access
As mentioned before, I imported the data by using the pandas function "read_csv()". The data are imported as a dataframe. The dataframe is then passed to the fucntion clean_data from the python file [train.py](https://github.com/AnnaDM87/Udacity_CAPSTONE/blob/main/starter_file/train.py).
The training set is then written to a file and read into a tabular dataset to send to the experiment using Dataset.Tabular.from_delimited_files. The training set is used for automl training. After training the test set is split into x_test and y_test for testing. Finally during deployment, the first two rows of the test set, excluding LeaveOrNot, are sent to the endpoint for prediction.
#### Hyperdrive tuning
For hyperdrive, the [train.py](https://github.com/AnnaDM87/Udacity_CAPSTONE/blob/main/starter_file/train.py) file reads the data in using the url, as was originally done before registering. Then the dataset is converted to a Pandas dataframe, LeaveOrNot is popped off and the dataset is split into x and y training and test sets using the same random seed as was used in the automl notebook.
The x_train and y_train are used to fit the logistic regression model and the x_test and y_test are used to compute accuracy.

## Automated ML
[automl.ipynb](https://github.com/AnnaDM87/Udacity_CAPSTONE/blob/main/starter_file/automl.ipynb)
Automated machine learning, also referred to as automated ML or AutoML, is the process of automating the time-consuming, iterative tasks of machine learning model development. It allows data scientists, analysts, and developers to build ML models with high scale, efficiency, and productivity all while sustaining model quality.
The following options were set:
    * experiment_timeout_minutes : 20,
    * max_concurrent_iterations : 5,
    * primary_metric  : 'AUC_weighted',The metric that Automated Machine Learning will optimize for model selection.
    * task = "classification",A classification job is used to train a model that best predict the class of a data sample. 
    * label_column_name="LeaveOrNot",   
    * enable_early_stopping= True,
    * featurization= 'auto',
[Reference](https://learn.microsoft.com/en-us/python/api/azure-ai-ml/azure.ai.ml.automl?view=azure-python)
### Results

As can be seen in the Run Details screenshot, the top performing model was a VotingEnsemble with 87% accuracy.
_Run Deatils Screenshot_
![Deatils Screenshot](https://github.com/AnnaDM87/Udacity_CAPSTONE/blob/main/starter_file/screenshot/run_details_automl.png?raw=true)
![best metric](https://github.com/AnnaDM87/Udacity_CAPSTONE/blob/main/starter_file/screenshot/autml_metric_chart.png?raw=true)
_Best Model Id Scrrenshot_
![best_run](https://github.com/AnnaDM87/Udacity_CAPSTONE/blob/main/starter_file/screenshot/best_run_automl.png?raw=true)

![best_model](https://github.com/AnnaDM87/Udacity_CAPSTONE/blob/main/starter_file/screenshot/best_model_automl.png?raw=true)
![best_model2](https://github.com/AnnaDM87/Udacity_CAPSTONE/blob/main/starter_file/screenshot/best_model_png_automl.png?raw=true)

## Hyperparameter Tuning
[hyperparameter_tuning.ipynb](https://github.com/AnnaDM87/Udacity_CAPSTONE/blob/main/starter_file/hyperparameter_tuning.ipynb)  
As the task is binary classification, I chose to use Scikit-Learn's logistic regression for hyperparameter tuning.
The parameters I chose to tune are C: the inverse of the regularization strength; solver: what algorithm is used for optimization; and, max_iter: max number of iterations to take for the solver to converge.
C has a default value of 1. For the sampler I used the discrete option 'choice' with 1 multiplied and divided by two powers of 10. Choice was used instead of a continuous sampler to ensure a large variation in the possible regularization strengths used.
Max_iter has a default value of 100. For the sample space I used choice from 50 to 300 in incriments of 50 from 50 to 100, then by 100.




### Results
The best hypedrive model scored an accuracy of 72 %, C =50, and max_iterations=200
_Run Deatils Screenshot_
![run details](https://github.com/AnnaDM87/Udacity_CAPSTONE/blob/main/starter_file/screenshot/run_details_hyper.png?raw=true)

![run_details2](https://github.com/AnnaDM87/Udacity_CAPSTONE/blob/main/starter_file/screenshot/run_details_hyper_2.png?raw=true)


![metric](https://github.com/AnnaDM87/Udacity_CAPSTONE/blob/main/starter_file/screenshot/results_hyper.png?raw=true)
![immagine](https://github.com/AnnaDM87/Udacity_CAPSTONE/assets/22540529/e5dc9f45-fa20-4986-b9af-b3bd2351bcf0)
_Best Model Id Scrrenshot_

_hypedrive summary_ 
![run_sumamry](https://github.com/AnnaDM87/Udacity_CAPSTONE/blob/main/starter_file/screenshot/hyperdrive_sumamry.png?raw=true)
## Model Deployment
Deployment is about delivering a trained model into production so that it can be consumed by others. In Azure, deploying the best model will allow it to interact with the HTTP API service and interact with the model by sending data over POST requests. In this project, I have deployed the model into a production environment using Azure Container Instance (ACI). ACI offers the fastest and simplest way to run a container without having to manage any virtual machines and without having to adopt a higher-level service. Also, authentication is enabled to prevent unauthorized access.
Following you can see the model deployed in azure. I enbaled the application insigths by checking the option while deploying the model


_Deployed model_
![deployed model](https://github.com/AnnaDM87/Udacity_CAPSTONE/blob/main/starter_file/screenshot/deploymodel.png?raw=true)

To test the model I used the script [endpoint.py](https://github.com/AnnaDM87/Udacity_CAPSTONE/blob/main/starter_file/endpoint.py). In the file, I used as example the first two records in teh test set.

_Example_
![example](https://github.com/AnnaDM87/Udacity_CAPSTONE/assets/22540529/8edf7013-5dc2-4fdc-8c87-479108fc8261)
_Result_
![endpoint](https://github.com/AnnaDM87/Udacity_CAPSTONE/blob/main/starter_file/screenshot/endpoint.png?raw=true)


## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
