# set up the environment
1. create python environment 
conda create -n py39 python=3.9  
2. install required packages
pip install -r requirements.txt


# Data Preprocessing
1. preprocess_{datasetname}.py: used for preprocessing and obtaining necessary features
2. dataset_{datasetname}.py: used to create train, val, test data


# Run
python run.py

after training, the metrics, model, and forecast results will be saved in ./output/

