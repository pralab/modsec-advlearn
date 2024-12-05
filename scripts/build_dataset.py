"""
This script is used to build a dataset for the training/testing phase composed
by 25k samples of malicious and 25k samples of legitimate payloads. 
The training and testing are splitted in 80% and 20% respectively. 
The dataset is built starting from the full dataset available in the following
repository https://github.com/pralab/http-traffic-dataset.
"""

import os
import toml
import sys
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    settings        = toml.load('config.toml')
    crs_dir         = settings['crs_dir']
    crs_ids_path    = settings['crs_ids_path']
    malicious_path  = settings['malicious_path']
    legitimate_path = settings['legitimate_path']
    dataset_path    = settings['dataset_path']

    loader = DataLoader(
        legitimate_path = legitimate_path,
        malicious_path  = malicious_path,
    )    
    
    df = loader.load_data()

    legitimate_data = shuffle(
        df[df['label'] == 0],
        random_state = 77,
        n_samples    = 25_000
    )
    malicious_data  = shuffle(
        df[df['label'] == 1],
        random_state = 77,
        n_samples    = 25_000
    )

    # LEGITIMATE DATA
    xtr, xts, _, _ = train_test_split(
        legitimate_data['payload'],
        legitimate_data['label'],
        test_size    = 0.2,
        random_state = 77,
        shuffle      = True
    )

    with open(os.path.join(dataset_path, 'legitimate_train.json'), 'w') as file:
        json.dump(xtr.tolist(), file, indent=4)

    with open(os.path.join(dataset_path, 'legitimate_test.json'), 'w') as file:
        json.dump(xts.tolist(), file, indent=4)

    # MALICIOUS DATA
    xtr, xts, _, _ = train_test_split(
        malicious_data['payload'],
        malicious_data['label'],
        test_size    = 0.2,
        random_state = 77,
        shuffle      = True
    )

    with open(os.path.join(dataset_path, 'malicious_train.json'), 'w') as file:
        json.dump(xtr.tolist(), file, indent=4)

    with open(os.path.join(dataset_path, 'malicious_test.json'), 'w') as file:
        json.dump(xts.tolist(), file, indent=4)