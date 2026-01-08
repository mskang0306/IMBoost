# IMBoost
Inlier-Memorization-Guided Active Outlier Detection (IMBoost)

### Code files description
* base: Pre-processing steps for the data
* datasets: Define a loader for the dataset and split it into train and test sets
* networks: Structure of an MLP that includes the encoder and decoder
* IMBoost: Define the model for the IMBoost to function
* IMBoost_tr: Train IMBoost and calculate TrainAUC, TrainAP, TestAUC, TestAP of datasets
* utils: Code for loading dataset 

### Run the Experiments
In this experiments, you can calculate TrainAUC, TrainAP(AveragePrecision), TestAUC, TestAP(AveragePrecision) of datasets using IMBoost.

### Implementation example
python IMBoost_tr.py --gpu_num 0 
