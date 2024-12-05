"""
This script trains models with different paranoia levels and penalties, 
using grid search and cross-validation for hyperparameter tuning.
It visualizes model performance before saving the best model in the `models` directory.
"""

import toml
import os
import sys
import joblib
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader
from src.extractor import ModSecurityFeaturesExtractor
from src.models import InfSVM2 as InfSVM
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Set to True if you want to train the models on the WAFAMOLE dataset
DS_WAFAMOLE = False

if __name__ == '__main__':
    settings         = toml.load('config.toml')
    crs_dir          = settings['crs_dir']
    crs_ids_path     = settings['crs_ids_path']
    models_path      = settings['models_path'] if not DS_WAFAMOLE else settings['models_wafamole_path']
    dataset_path     = settings['dataset_path'] if not DS_WAFAMOLE else settings['dataset_wafamole_path']
    paranoia_levels  = settings['params']['paranoia_levels']
    models           = list(filter(lambda model: model != 'modsec', settings['params']['other_models']))
    models          += settings['params']['models']
    penalties        = settings['params']['penalties']
    
    # ----------------------
    # LOADING DATASET PHASE
    # ----------------------
    print('[INFO] Loading dataset...')

    loader = DataLoader(
        malicious_path  = os.path.join(
            dataset_path, 
            f'malicious_train.{"pkl" if DS_WAFAMOLE else "json"}'
        ),
        legitimate_path = os.path.join(
            dataset_path, 
            f'legitimate_train.{"pkl" if DS_WAFAMOLE else "json"}'
        )
    )  
    
    training_data = loader.load_data_pkl() if DS_WAFAMOLE else loader.load_data()

    # ---------------------
    # STARTING EXPERIMENTS
    # ---------------------
    for pl in paranoia_levels:
        print(f'[INFO] Extracting features for PL {pl}...')
        
        extractor = ModSecurityFeaturesExtractor(
            crs_ids_path = crs_ids_path,
            crs_path     = crs_dir,
            crs_pl       = pl
        )
    
        xtr, ytr = extractor.extract_features(training_data)

        for model_name in models:
            print(f'[INFO] Hyperparameter tuning and training for {model_name} model at PL {pl}...')
            
            # Initialize the parameter grid and model for each type
            if model_name == 'infsvm':
                param_grid = {'t': [0.1, 0.5, 1.0]}
                model      = InfSVM()
            
            elif model_name == 'svc':
                param_grid = {'C': [0.1, 0.5, 1.0], 'penalty': penalties}
                model = LinearSVC(
                    dual          = 'auto',
                    class_weight  = 'balanced',
                    random_state  = 77,
                    fit_intercept = False
                )
            
            elif model_name == 'rf':
                param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20]}
                model = RandomForestClassifier(
                    class_weight = 'balanced',
                    random_state = 77,
                    n_jobs       = -1
                )
            
            elif model_name == 'log_reg':
                param_grid = {'C': [0.1, 0.5, 1.0], 'penalty': penalties}
                model = LogisticRegression(
                    class_weight = 'balanced',
                    random_state = 77,
                    n_jobs       = -1,
                    max_iter     = 1000,
                    solver       = 'saga'
                )

            # Run GridSearchCV
            grid_search = GridSearchCV(
                model,
                param_grid,
                cv      = 5,
                scoring = 'f1',
                n_jobs  = -1
            )
            grid_search.fit(xtr, ytr)

            # Get cross-validation results and plot them
            results = grid_search.cv_results_
            mean_scores = results['mean_test_score']
            param_names = list(param_grid.keys())

            # Save the best model after visualization
            best_model = grid_search.best_estimator_
            print(best_model)
            joblib.dump(
                best_model, 
                os.path.join(models_path, f'{model_name}_pl{pl}.joblib')
            )
