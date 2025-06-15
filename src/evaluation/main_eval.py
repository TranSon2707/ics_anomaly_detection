import importlib.util
import os

util_path = os.path.join(os.path.dirname(__file__), "..", "data_loader", "__pycache__", "util.pyc")
util_path = os.path.abspath(util_path)
spec = importlib.util.spec_from_file_location("util", util_path)
util = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(util)

from src.hyperparameter_tuning import tuning_ML, tuning_DL
from src.data_loader.data_loader import load_test_data

# from ..hyperparameter_tuning import tuning_ML, tuning_DL
# from ..data_loader.data_loader import load_test_data
from sklearn.metrics import f1_score, classification_report

import numpy as np
import pandas as pd

def test_evaluation(model_name, number) :

    X_test_scaled, Y_test = load_test_data('data/BATADAL', number)
    X_test_orig = X_test_scaled.copy()  # Preserve original 2D data

    model_list = None
    best_f1, best_threshold = 0.0, 0.0
    report = None

    if model_name == 'ridge' :
        model_list = tuning_ML.tuning_threshold_ridge_MOR()
    elif model_name == 'svr' :
        model_list = tuning_ML.best_model_retrive_SVR()
    elif model_name == 'adaboost' :
        model_list = tuning_ML.best_model_retrive_adaboost()
    elif model_name == 'cnn_ae' :
        model_list = tuning_DL.best_model_retrive_cnn_ae()
    elif model_name == 'lstm_ae' :
        model_list = tuning_DL.best_model_retrive_lstm_ae()
    elif model_name == 'random_forest' :
        model_list = tuning_ML.best_model_retrive_RF()
    elif model_name == 'xgboost' :
        model_list = tuning_ML.best_model_retrive_XGB()
    for model in model_list:
        threshold = model['threshold']
        X_test_hat = None

        if model_name == 'cnn_ae':
            # Reshape for CNN: (samples, 1, features)
            X_test_3d = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
            X_test_hat = model['model'].predict(X_test_3d)
            X_test_hat = np.squeeze(X_test_hat, axis=1)  # Output to (samples, features)
            
        elif model_name == 'lstm_ae':
            # Reshape for LSTM: (samples, timesteps, features) = (samples, 43, 1)
            X_test_3d = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
            X_test_hat = model['model'].predict(X_test_3d)
            X_test_hat = np.squeeze(X_test_hat, axis=-1)  # Output to (samples, 43)
            
        else:
            # Other models use 2D data directly
            X_test_hat = model['model'].predict(X_test_scaled)

        # Calculate reconstruction errors using ORIGINAL 2D data
        errors = np.mean((X_test_orig - X_test_hat) ** 2, axis=1)
    
    for model in model_list :

        threshold = model['threshold']
        X_test_hat = None

        if model_name == 'cnn_ae':
            # Reshape for CNN: (samples, 1, features)
            X_test_3d = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
            X_test_hat = model['model'].predict(X_test_3d)
            X_test_hat = np.squeeze(X_test_hat, axis=1)  # Output to (samples, features)
            
        elif model_name == 'lstm_ae':
            # Reshape for LSTM: (samples, timesteps, features) = (samples, 43, 1)
            X_test_3d = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
            X_test_hat = model['model'].predict(X_test_3d)
            X_test_hat = np.squeeze(X_test_hat, axis=-1)  # Output to (samples, 43)
            
        else:
            # Other models use 2D data directly
            X_test_hat = model['model'].predict(X_test_scaled)

        # Tính lỗi tái tạo (MSE từng điểm)
        errors = np.mean((X_test_scaled - X_test_hat) ** 2, axis=1)

        # Phát hiện bất thường: nếu lỗi vượt ngưỡng thì gán là bất thường (1)
        Y_pred = (errors > threshold).astype(int)

        # Tính F1-score (giữa dự đoán và nhãn thật)
        f1 = f1_score(Y_test, Y_pred)

        if f1 > best_f1 :
            best_threshold = threshold
            # report = util.classification_metric(classification_report(Y_test, Y_pred, output_dict=True))
            report = classification_report(Y_test, Y_pred, output_dict=True)           
            best_f1 = report['1.0']['f1-score']
            report_cleaned = pd.DataFrame(report).transpose().to_string(float_format="%.2f")


    if model_name == 'ridge' :
        print(f"[{model_name.upper()}] F1-score on test_{number} = {best_f1:.4f} with threshold = {best_threshold:.6f}\n")
        print(f'------------------- REPORT FOR RIDGE REGRESSION FOR DATASET {number} -------------------')
        print(report_cleaned)
    elif model_name == 'svr' :
        print(f"[{model_name.upper()}] F1-score on test_{number} = {best_f1:.4f} with threshold = {best_threshold:.6f}\n")
        print(f'------------------- REPORT FOR SVR FOR DATASET {number} -------------------')
        print(report_cleaned)
    elif model_name == 'adaboost' :
        print(f"[{model_name.upper()}] F1-score on test_{number} = {best_f1:.4f} with threshold = {best_threshold:.6f}\n")
        print(f'------------------- REPORT FOR ADABOOST FOR DATASET {number} -------------------')
        print(report_cleaned)
    elif model_name == 'cnn_ae' :
        print(f"[{model_name.upper()}] F1-score on test_{number} = {best_f1:.4f} with threshold = {best_threshold:.6f}\n")
        print(f'------------------- REPORT FOR CNN + AE FOR DATASET {number} -------------------')
        print(report_cleaned)
    elif model_name == 'lstm_ae' :
        print(f"[{model_name.upper()}] F1-score on test_{number} = {best_f1:.4f} with threshold = {best_threshold:.6f}\n")
        print(f'------------------- REPORT FOR LSTM + AE FOR DATASET {number} -------------------')
        print(report_cleaned)
    elif model_name == 'random_forest' :
        print(f"[{model_name.upper()}] F1-score on test_{number} = {best_f1:.4f} with threshold = {best_threshold:.6f}\n")
        print(f'------------------- REPORT FOR RANDOM FOREST FOR DATASET {number} -------------------')
        print(report_cleaned)
    elif model_name == 'xgboost' :
        print(f"[{model_name.upper()}] F1-score on test_{number} = {best_f1:.4f} with threshold = {best_threshold:.6f}\n")
        print(f'------------------- REPORT FOR XGBOOST FOR DATASET {number} -------------------')
        print(report_cleaned)
    
    return f1

def main_eval() :
    model_name = input("Select the model (ridge / svr / adaboost / cnn_ae): ").strip().lower()
    dataset_number = input("Number of testing dataset (1 or 2): ").strip()

    for dataset_number in ['1', '2']:
        
        valid_models = {'ridge', 'svr', 'adaboost', 'cnn_ae', 'lstm_ae', 'random_forest', 'xgboost'}
        if model_name in valid_models and dataset_number in {'1', '2'}:
            test_evaluation(model_name, int(dataset_number))
        else:
            print("Invalid input. Please try again by typing the valid model name (ridge, svr, adaboost, cnn_ae) and testing dataset is 1 or 2.")


'''Program'''

if __name__ == "__main__":
    #main_eval()
    model_name = 'lstm_ae'

    for dataset_number in ['1', '2']:
        
        valid_models = {'ridge', 'svr', 'adaboost', 'cnn_ae', 'lstm_ae', 'random_forest', 'xgboost'}
        if model_name in valid_models and dataset_number in {'1', '2'}:
            test_evaluation(model_name, int(dataset_number))
        else:
            print("Invalid input. Please try again by typing the valid model name (ridge, svr, adaboost, cnn_ae) and testing dataset is 1 or 2.")

'''-------'''