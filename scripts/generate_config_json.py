"""
A script to generate JSON configuration files for the WAF-A-MoLE tool.
"""

import json
import os

# Directory paths
base_dir   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
config_dir = os.path.join(base_dir, "wafamole_models_config")
# NOTE: This should be change, accordin to which dataset you are using 
models_dir = os.path.join(base_dir, "data/models_wafamole")
rules_dir  = os.path.join(base_dir, "coreruleset", "rules")
ids_path   = os.path.join(base_dir, "data/crs_sqli_ids_4.0.0.json")

# Model types and their file extensions
model_types = {
    "inf_svm"      : "inf_svm_pl{pl}_t0.5.joblib",
    "linear_svc_l1": "linear_svc_pl{pl}_l1.joblib",
    "linear_svc_l2": "linear_svc_pl{pl}_l2.joblib",
    "log_reg_l1"   : "log_reg_pl{pl}_l1.joblib",
    "log_reg_l2"   : "log_reg_pl{pl}_l2.joblib",
    "rf"           : "rf_pl{pl}.joblib"
}

# PL values and threshold
pl_values     = range(1, 5)
crs_threshold = 5.0

# Generate and save JSON files
for pl in pl_values:
    for model_name, model_filename in model_types.items():
        config = {
            "sklearn_clf_path"  : os.path.join(models_dir, model_filename.format(pl=pl)),
            "rules_path"        : rules_dir,
            "crs_rules_ids_path": ids_path,
            "crs_pl"            : pl,
            "crs_threshold"     : crs_threshold
        }
        
        # JSON file path
        json_file_path = os.path.join(config_dir, f"{model_name}_crs_pl{pl}_config.json")
        
        # Save JSON to a file
        with open(json_file_path, 'w') as json_file:
            json.dump(config, json_file, indent=4)
            
        print(f"Generated JSON file: {json_file_path}")