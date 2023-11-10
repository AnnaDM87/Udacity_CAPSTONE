from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory


def clean_data(data):

    
    # label encoder
    data.head()
    x_df=data
    x_df=pd.get_dummies(data=x_df, columns=['Education','Gender','City','EverBenched'], drop_first=True)
    y_df=x_df.pop('LeaveOrNot')

    return x_df, y_df

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run = Run.get_context()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    # TODO: Create TabularDataset using TabularDatasetFactory
    # Data is located at:
    # "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

    ds = pd.read_csv('https://raw.githubusercontent.com/AnnaDM87/Udacity_CAPSTONE/main/starter_file/Employee.csv')
    
    x, y = clean_data(ds)

    # TODO: Split data into train and test sets.
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

    ### YOUR CODE HERE ###a

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)
    folder='outputs'
    os.makedirs(folder, exist_ok=True)
    joblib.dump(value=model,filename=os.path.join(folder,'model.joblib'))

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()

