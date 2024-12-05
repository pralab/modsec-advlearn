"""
This script is used to train the models with different paranoia levels and penalties.
The trained models are saved as joblib files in the `models` directory.
"""

import toml
import os
import sys
import joblib
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader
from src.extractor import ModSecurityFeaturesExtractor
from src.models import InfSVM2 as InfSVM
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Set to True if you want to train the models on the WAFAMOLE dataset
DS_WAFAMOLE = False

if __name__ == '__main__':
    settings         = toml.load('config.toml')
    crs_dir          = settings['crs_dir']
    crs_ids_path     = settings['crs_ids_path']
    models_path      = settings['models_path'] if not DS_WAFAMOLE else settings['models_wafamole_path']
    dataset_path     = settings['dataset_path'] if not DS_WAFAMOLE else settings['dataset_wafamole_path']
    figures_path     = settings['figures_path']
    paranoia_levels  = settings['params']['paranoia_levels']
    models           = list(filter(lambda model: model != 'modsec', settings['params']['other_models']))
    models          += settings['params']['models']
    penalties        = settings['params']['penalties']
    t                = [0.5]
    
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
    
    if DS_WAFAMOLE:
        training_data = loader.load_data_pkl()
    else:
        training_data = loader.load_data()

    models_weights = dict()
    
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
            print(f'[INFO] Training {model_name} model for PL {pl}...')
            
            if model_name == 'infsvm':
                for number in t: 
                    model = InfSVM(number)
                    model.fit(xtr, ytr)
                    joblib.dump(
                        model, 
                        os.path.join(models_path, f'inf_svm_pl{pl}_t{number}.joblib')
                    )
                    
            if model_name == 'svc':
                for penalty in penalties:
                    model = LinearSVC(
                        C             = 0.5,
                        penalty       = penalty,
                        dual          = 'auto',
                        class_weight  = 'balanced',
                        random_state  = 77,
                        fit_intercept = False,
                    )
                    model.fit(xtr, ytr)
                    joblib.dump(
                        model, 
                        os.path.join(models_path, f'linear_svc_pl{pl}_{penalty}.joblib')
                    )
                        
            elif model_name == 'rf':
                model = RandomForestClassifier(
                    class_weight = 'balanced',
                    random_state = 77,
                    n_jobs       = -1
                )
                model.fit(xtr, ytr)
                joblib.dump(
                    model, 
                    os.path.join(models_path, f'rf_pl{pl}.joblib'))

            elif model_name == 'log_reg':
                for penalty in penalties:
                    model = LogisticRegression(
                        C            = 0.5,
                        penalty      = penalty,
                        class_weight = 'balanced',
                        random_state = 77,
                        n_jobs       = -1,
                        max_iter     = 1000,
                        solver       = 'saga'
                    )
                    model.fit(xtr, ytr)
                    joblib.dump(
                        model, 
                        os.path.join(models_path, f'log_reg_pl{pl}_{penalty}.joblib')
                    )