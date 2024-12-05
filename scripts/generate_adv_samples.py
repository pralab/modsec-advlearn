"""
The script is used to generate adversarial examples using WAF-a-MoLE.
"""

import shlex
import subprocess
import os
import sys
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


base_path   = os.path.join(os.path.dirname(__file__), '..')
config_path = os.path.join(base_path, 'wafamole_config')

# NOTE: This should be change, accordin to which dataset you are using 
dataset_path    = os.path.join(base_path, 'data/dataset')
output_base_dir = os.path.join(base_path, 'modsec_results/results_dataset')

# NOTE: The .json files are the modsec-learn dataset, while the .pkl files are the WAF-a-MoLE dataset
dataset_path_test     = os.path.join(dataset_path, 'malicious_test.json')
dataset_path_advtrain = os.path.join(dataset_path, 'malicious_train.json')

crs_path          = os.path.join(base_path, 'coreruleset')
wafamole_cli_path = os.path.join(base_path, 'scripts/run_wafamole.py')

target_wafs       = {
    # MS CRS vanilla
    'ms_pl1': ('modsecurity_pl1', crs_path),
    'ms_pl2': ('modsecurity_pl2', crs_path),
    'ms_pl3': ('modsecurity_pl3', crs_path),
    'ms_pl4': ('modsecurity_pl4', crs_path),
    
    # ModSec-Learn - Inf SVM
    'inf_svm_pl1': ('ml_model_crs', os.path.join(config_path, 'inf_svm_crs_pl1_config.json')),
    'inf_svm_pl2': ('ml_model_crs', os.path.join(config_path, 'inf_svm_crs_pl2_config.json')),
    'inf_svm_pl3': ('ml_model_crs', os.path.join(config_path, 'inf_svm_crs_pl3_config.json')),
    'inf_svm_pl4': ('ml_model_crs', os.path.join(config_path, 'inf_svm_crs_pl4_config.json')),
    
    # ModSec-Learn - Logistic Regression
    'log_reg_l1_pl1': ('ml_model_crs', os.path.join(config_path, 'log_reg_l1_crs_pl1_config.json')),
    'log_reg_l1_pl2': ('ml_model_crs', os.path.join(config_path, 'log_reg_l1_crs_pl2_config.json')),
    'log_reg_l1_pl3': ('ml_model_crs', os.path.join(config_path, 'log_reg_l1_crs_pl3_config.json')),
    'log_reg_l1_pl4': ('ml_model_crs', os.path.join(config_path, 'log_reg_l1_crs_pl4_config.json')),
    'log_reg_l2_pl1': ('ml_model_crs', os.path.join(config_path, 'log_reg_l2_crs_pl1_config.json')),
    'log_reg_l2_pl2': ('ml_model_crs', os.path.join(config_path, 'log_reg_l2_crs_pl2_config.json')),
    'log_reg_l2_pl3': ('ml_model_crs', os.path.join(config_path, 'log_reg_l2_crs_pl3_config.json')),
    'log_reg_l2_pl4': ('ml_model_crs', os.path.join(config_path, 'log_reg_l2_crs_pl4_config.json')),
    
    # ModSec-Learn - SVM
    'svm_linear_l1_pl1': ('ml_model_crs', os.path.join(config_path, 'linear_svc_l1_crs_pl1_config.json')),
    'svm_linear_l1_pl2': ('ml_model_crs', os.path.join(config_path, 'linear_svc_l1_crs_pl2_config.json')),
    'svm_linear_l1_pl3': ('ml_model_crs', os.path.join(config_path, 'linear_svc_l1_crs_pl3_config.json')),
    'svm_linear_l1_pl4': ('ml_model_crs', os.path.join(config_path, 'linear_svc_l1_crs_pl4_config.json')),
    'svm_linear_l2_pl1': ('ml_model_crs', os.path.join(config_path, 'linear_svc_l2_crs_pl1_config.json')),
    'svm_linear_l2_pl2': ('ml_model_crs', os.path.join(config_path, 'linear_svc_l2_crs_pl2_config.json')),
    'svm_linear_l2_pl3': ('ml_model_crs', os.path.join(config_path, 'linear_svc_l2_crs_pl3_config.json')),
    'svm_linear_l2_pl4': ('ml_model_crs', os.path.join(config_path, 'linear_svc_l2_crs_pl4_config.json')),
    
    # ModSec-Learn - Random Forest
    'rf_pl1': ('ml_model_crs', os.path.join(config_path, 'rf_crs_pl1_config.json')),
    'rf_pl2': ('ml_model_crs', os.path.join(config_path, 'rf_crs_pl2_config.json')),
    'rf_pl3': ('ml_model_crs', os.path.join(config_path, 'rf_crs_pl3_config.json')),
    'rf_pl4': ('ml_model_crs', os.path.join(config_path, 'rf_crs_pl4_config.json')),
    
    # ModSec-AdvLearn
    'svm_linear_l1_pl4_advtrain': ('ml_model_crs', os.path.join(config_path, 'linear_svc_l1_crs_pl4_adv_config.json')),
    'svm_linear_l2_pl4_advtrain': ('ml_model_crs', os.path.join(config_path, 'linear_svc_l2_crs_pl4_adv_config.json')),
    'rf_pl4_advtrain'           : ('ml_model_crs', os.path.join(config_path, 'rf_crs_pl4_adv_config.json')),
    'log_reg_l1_pl4_advtrain'   : ('ml_model_crs', os.path.join(config_path, 'log_reg_l1_crs_pl4_adv_config.json')),
    'log_reg_l2_pl4_advtrain'   : ('ml_model_crs', os.path.join(config_path, 'log_reg_l2_crs_pl4_adv_config.json')),
    'inf_svm_pl4_advtrain'      : ('ml_model_crs', os.path.join(config_path, 'inf_svm_crs_pl4_adv_config.json')),
}

thresholds          = { waf: -200.0 for waf in target_wafs }
rand_seed           = 0
round_size_default  = 20
max_queries_default = 2000
use_multiproc       = True


def run_experiments(test_cases, out_dir, dataset_path):
    cmd_base = "python {wafamole_cli} -w {waf_type} --threshold {thr} " + \
               "--round-size {round_size} --max-rounds {max_rounds} --random-seed {seed} " + \
               "-o {out_path}  {waf_path} {dataset_path}"

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    for tc in test_cases:
        round_size  = tc.get('round_size', round_size_default)
        max_queries = tc.get('max_queries', max_queries_default)
        max_rounds  = int(max_queries // round_size)

        waf_name  = tc['model']
        waf_type, waf_path = target_wafs[waf_name]
        model_thr = float(1e-6)

        savename = waf_name[:-9] if waf_name.endswith('_advtrain') else waf_name
        out_path = os.path.join(out_dir, 'output_{}_rs{}_{}rounds.json'.format(savename, round_size, max_rounds))

        print("> WAF-a-MoLE configured with the following settings: WAF={} ({}), conf={}, max_queries={}, "
            "threshold={}, round_size={}, rand_seed={}, ".format(
            waf_name, waf_type, waf_path, max_queries, model_thr, round_size, rand_seed))

        cmd = cmd_base.format(
            wafamole_cli  = wafamole_cli_path,
            waf_type      = waf_type,
            waf_path      = waf_path,
            thr           = model_thr,
            out_path      = out_path,
            round_size    = round_size,
            max_rounds    = max_rounds,
            use_multiproc = use_multiproc,
            seed          = rand_seed,
            dataset_path  = dataset_path
        )
        
        subprocess.Popen(shlex.split(cmd))


if __name__ == "__main__":
    if not os.path.isdir(base_path):
        raise Exception("Please set the base path. Invalid path specified : {}".format(base_path))
    
    parser = argparse.ArgumentParser(description='Generate adversarial SQLi examples using WAF-a-MoLE.')
    parser.add_argument(
        'type', 
        metavar = 'target',
        type    = str,
        help    = 'Type of the target: test-adv (test ModSec and ModSec-Learn), advtrain (adv training of ModSec-Learn), retrained-test-adv (test ModSec-AdvLearn)'
    )
    args = parser.parse_args()

    if args.type not in ['test-adv', 'advtrain', 'advmodsec-test-adv']:
        raise Exception("Invalid type of dataset")
    
    if args.type == 'test-adv':
        test_cases = (
            [{'round_size': 20, 'model': 'ms_pl{}'.format(pl), 'max_queries': 2000} for pl in range(1, 5)]
        )
        test_cases.extend(
            [{'round_size': 20, 'model': 'inf_svm_pl{}'.format(pl), 'max_queries': 2000} for pl in range(1, 5)]
        )
        test_cases.extend(
            [{'round_size': 20, 'model': 'svm_linear_l1_pl{}'.format(pl), 'max_queries': 2000} for pl in range(1, 5)]
        )
        test_cases.extend(
            [{'round_size': 20, 'model': 'svm_linear_l2_pl{}'.format(pl), 'max_queries': 2000} for pl in range(1, 5)]
        )
        test_cases.extend(
            [{'round_size': 20, 'model': 'log_reg_l1_pl{}'.format(pl), 'max_queries': 2000}
                for pl in range(1, 5)])
        test_cases.extend(
            [{'round_size': 20, 'model': 'log_reg_l2_pl{}'.format(pl), 'max_queries': 2000}
                for pl in range(1, 5)])
        test_cases.extend(
            [{'round_size': 20, 'model': 'rf_pl{}'.format(pl), 'max_queries': 2000} for pl in range(1, 5)]
        )

        out_dir = os.path.join(
            output_base_dir,
            'adv_examples_test'
        )

        run_experiments(test_cases, out_dir, dataset_path_test)

    elif args.type == 'advtrain':
        test_cases_advtrain = [
            {'round_size': 20, 'model': 'inf_svm_pl4', 'max_queries': 2000},
        ]
        test_cases_advtrain.extend(
            [{'round_size': 20, 'model': 'svm_linear_l1_pl4', 'max_queries': 2000}]
        )
        test_cases_advtrain.extend(
            [{'round_size': 20, 'model': 'svm_linear_l2_pl4', 'max_queries': 2000}]
        )
        test_cases_advtrain.extend(
            [{'round_size': 20, 'model': 'ms_pl4', 'max_queries': 2000}]
        )
        test_cases_advtrain.extend(
            [{'round_size': 20, 'model': 'log_reg_l1_pl4', 'max_queries': 2000}
                for pl in range(1, 5)])
        test_cases_advtrain.extend(
            [{'round_size': 20, 'model': 'log_reg_l2_pl4', 'max_queries': 2000}]
        )
        test_cases_advtrain.extend(
            [{'round_size': 20, 'model': 'rf_pl4', 'max_queries': 2000}]
        )

        out_dir_advtrain = os.path.join(
            output_base_dir,
            'adv_examples_advtrain'
        )

        run_experiments(test_cases_advtrain, out_dir_advtrain, dataset_path_advtrain)

    else:
        test_cases_advmodsec = [
            {'round_size': 20, 'model': 'inf_svm_pl4_advtrain', 'max_queries': 2000}      
        ]
        test_cases_advmodsec.extend(
            [{'round_size': 20, 'model': 'svm_linear_l1_pl4_advtrain', 'max_queries': 2000}]
        )
        test_cases_advmodsec.extend(
            [{'round_size': 20, 'model': 'svm_linear_l2_pl4_advtrain', 'max_queries': 2000}]
        )
        test_cases_advmodsec.extend(
            [{'round_size': 20, 'model': 'ms_pl4_advtrain', 'max_queries': 2000}]
        )
        test_cases_advmodsec.extend(
            [{'round_size': 20, 'model': 'log_reg_l1_pl4_advtrain', 'max_queries': 2000}
                for pl in range(1, 5)])
        test_cases_advmodsec.extend(
            [{'round_size': 20, 'model': 'log_reg_l2_pl4_advtrain', 'max_queries': 2000}]
        )
        test_cases_advmodsec.extend(
            [{'round_size': 20, 'model': 'rf_pl4_advtrain', 'max_queries': 2000}]
        )

        out_dir_advmodsec = os.path.join(
            output_base_dir,
            'adv_examples_retrained_test'
        )
        
        run_experiments(test_cases_advmodsec, out_dir_advmodsec, dataset_path_test)