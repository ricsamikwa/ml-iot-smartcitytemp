# Machine Learning-based Energy Optimisation in Smart City Internet of Things

## Usage

The pre-trained LSTM, 1D-CNN, and FFNN models are in `models` folder.

### Running pre-trained models 

To run inference on the pre-trained models using sample inputs: 
```
python3 run_model.py 
```
The output shows the input shape for each model and makes a prediction using sample inputs. The actual input samples are defined in input_formats.py. For LSTM, there are two input types; without and with padding. The inputs are scaled with MinMaxScaler(feature_range=(0, 1)) over the dataset, and then the predictions are scaled back, as the steps in this file model_evaluation1.py

## Model Training 

NB: This will retrain the models and overwrite the existing models in the `models` folder.

### Data processing

To download unstructured dataset into `dataset` folder:
```
python3 data_processing1.py 
```
The result is a downloaded `dataset.pkl` file in `dataset` folder AND a restructured dataset without padding `normal_tran_dataset.npy` and `normal_final_temp.npy` representing the multivariate temperature and humidity time series, and the end temperature values respectively.

To create a dataset with variable pre-padding for the multivariate time series :
```
python3 data_processing2.py 
```
The result is a restructured dataset with pre-padding `exp_final_temp1.npy` and `exp_tran_dataset1.npy` representing the multivariate temperature and humidity time series and the end temperature values, respectively. 

### Training

To retrain and save the trained LSTM model:
```
python3 model_training1.py 
```
The files `cnn_model_training2.npy` and `dense_model_training2.npy` retrain and save the CNN and FFNN models, respectively.

## Model Evaluation

The folder `results` contain the results of the model evaluation for the LSTM (`model_evaluation2.py`), 1D-CNN (`cnn_model_evaluation2.py`), and FFNN (`dense_model_evaluation2.py`).

### evaluation scripts 

NB: This will re-evaluate the models and append the results to the existing files in the `results` folder.

The results will be added at the end of the evaluation file if already available.

To evaluate the saved trained LSTM model without padding:
```
python3 model_evaluation1.py
```

To evaluate the saved trained LSTM model with padding:
```
python3 model_evaluation2.py
```

To evaluate the saved trained 1D CNN model with padding:

```
python3 cnn_model_evaluation2.py
```

To evaluate the saved trained Dense model with padding:
```
python3 dense_model_evaluation2.py
```
