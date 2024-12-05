"""
This script is used as interface to the WAFamole evasion engine. 
It is used in conjunction with the `generate_adv_samples.py` script to generate adversarial
examples using WAF-a-MoLE.
"""

import click
import json
import numpy as np
import random
import pickle
import re
import toml
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import PyModSecurityWafamole
from wafamole.evasion import EvasionEngine
from wafamole.exceptions.models_exceptions import UnknownModelError
from src.models.sklearn_modsecurity_ml_waf import SklearnModSecurityMlWaf


def _load_json_conf(json_conf_path):
    try:
        with open(json_conf_path, 'r') as fp:
            json_conf = json.load(fp)
    except (json.JSONDecodeError, OSError) as e:
        raise Exception("Error loading the JSON config options for the target Web WAF:\n{}".format(e))
    else:
        return json_conf


@click.group()
def wafamole():
    pass

@wafamole.command()
@click.option(
    "--waf-type", "-w", default="token", help="Target WAF", type=str
)
@click.option(
    "--timeout", "-t", default=14400, help="Timeout when evading the WAF", type=int
)
@click.option(
    "--max-rounds", "-r", default=1000, help="Maximum number of fuzzing rounds", type=int
)
@click.option(
    "--round-size", "-s", default=20, help="Fuzzing step size for each round (parallel fuzzing steps)", type=int
)
@click.option(
    "--threshold", default=0.5, help="Classification threshold of the target WAF (default 0.5)", type=float
)
@click.option(
    "--random-seed", default=0, help="Random seed to use (default: 0)", type=int
)
@click.option(
    "--output-path", "-o", default="output.json", help="Location were to save the adversarial examples found (JSON file).", type=str
)
@click.option(
    "--use-multiproc", "-m", default=False, help="Whether to enable multiprocessing for fuzzing", type=bool
)
@click.argument("waf-path", default="")
@click.argument("dataset-path")
def run_wafamole(
    waf_path,
    dataset_path,
    waf_type,
    timeout,
    max_rounds,
    round_size,
    threshold,
    random_seed,
    output_path,
    use_multiproc
):
    settings         = toml.load('config.toml')
    crs_dir          = settings['crs_dir']
    n_adv_samples    = 2000
    
    np.random.seed(random_seed)
    random.seed(random_seed)

    print("[INFO] Loading the target WAF...")

    if re.match(r"modsecurity_pl[1-4]", waf_type):
        pl = int(waf_type[-1]) 
        waf = PyModSecurityWafamole(
            rules_dir = crs_dir,
            threshold = 5.0,
            pl        = pl
        )
    elif waf_type == "ml_model_crs":
        waf = SklearnModSecurityMlWaf(**_load_json_conf(waf_path))
    else:
        raise UnknownModelError("Unsupported WAF type: {}".format(waf_type))

    engine = EvasionEngine(waf, use_multiproc)

    print("[INFO] Loading the dataset...")
    try:
        with open(dataset_path, 'r') as fp:
            dataset = json.load(fp)
    except Exception as error:
        raise SystemExit("Error loading the dataset: {}".format(error))
    
    print("[INFO] Number of attack payloads: {}".format(len(dataset)))

    with open(output_path, 'w') as out_file:
        for sample in dataset[:n_adv_samples]:
            best_score, adv_sample, scores_trace, _, _, _, _ = engine.evaluate(
                sample, 
                max_rounds, 
                round_size, 
                timeout, 
                threshold
            )
            
            if isinstance(best_score, np.ndarray):
                best_score = best_score.tolist()[0]

            info = {
                'payload'     : sample,
                'adv_payload' : adv_sample,
                'best_score'  : best_score,
                'scores_tarce': scores_trace
            }
            
            # Save the adversarial example to the output file
            out_file.write(json.dumps(info, indent=4) + '\n')


if __name__ == "__main__":
    run_wafamole()