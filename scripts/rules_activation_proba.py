"""
This script is used to plot the activation probability and the weights of the CRS rules for the ML models.
"""

import matplotlib.pyplot as plt
import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import toml
import sys
import joblib
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader
from src.extractor import ModSecurityFeaturesExtractor

# Set to True if you want to train the models on the WAFAMOLE dataset
DS_WAFAMOLE = False

def analyze_rules_importance(
    crs_rules_ids_file : str,
    figs_save_path     : str,
    benign_load_path   : str,
    attacks_load_path  : str,
    adv_payloads_path  : str,
    model_name         : str,
    model_path         : str,
    model_advtrain_path: str,
    pl               = 4,
    rules_selector   = None,
    legend_fontsize  = 13,
    axis_labels_size = 20,
    tick_labels_size = 14
):
    # ------
    # SETUP
    # ------
    with open(crs_rules_ids_file, 'r') as fp:
        data = json.load(fp)
        owasp_crs_ids = sorted(data['rules_ids'])

    info_rules               = [942011, 942012, 942013, 942014, 942015, 942016, 942017, 942018]
    crs_rules_ids_pl1_unique = [942100, 942140, 942151, 942160, 942170, 942190, 942220, 942230, 942240, 942250, 942270, 942280, 942290, 942320, 942350, 942360, 942500, 942540, 942560, 942550]
    crs_rules_ids_pl3_unique = [942251, 942490, 942420, 942431, 942460, 942511, 942530]
    crs_rules_ids_pl4_unique = [942421, 942432]
    crs_rules_ids_pl2_unique = list(set([int(rule) for rule in owasp_crs_ids]) - set(crs_rules_ids_pl1_unique + crs_rules_ids_pl3_unique + crs_rules_ids_pl4_unique + info_rules))

    crs_rules_ids_pl1 = crs_rules_ids_pl1_unique
    crs_rules_ids_pl2 = crs_rules_ids_pl1 + crs_rules_ids_pl2_unique
    crs_rules_ids_pl3 = crs_rules_ids_pl2 + crs_rules_ids_pl3_unique
    crs_rules_ids_pl4 = crs_rules_ids_pl3 + crs_rules_ids_pl4_unique

    # -----------------------------------
    # COMPUTING MODEL WEIGHTS AND SCORES
    # -----------------------------------
    num_samples = 2000

    # Load default dataset
    loader = DataLoader(
        malicious_path  = attacks_load_path,
        legitimate_path = benign_load_path
    )
    
    # Load adversarial dataset
    loader_adv = DataLoader(
        malicious_path  = adv_payloads_path,
        legitimate_path = benign_load_path
    )

    if DS_WAFAMOLE:
        dataset  = loader.load_data_pkl()
        data_adv = loader_adv.load_data_pkl()
    else:
        dataset  = loader.load_data()
        data_adv = loader_adv.load_data()
        
    extractor = ModSecurityFeaturesExtractor(
        crs_ids_path = crs_ids_path,
        crs_path     = crs_dir,
        crs_pl       = pl
    )

    benign_datset = dataset[dataset['label'] == 0][:num_samples]
    xts_benign, _ = extractor.extract_features(benign_datset)
    
    malicious_dataset = dataset[dataset['label'] == 1][:num_samples]
    xts_attack, _     = extractor.extract_features(malicious_dataset)
    
    adv_dataset = data_adv[data_adv['label'] == 1][:num_samples]
    xts_adv, _  = extractor.extract_features(adv_dataset)
    
    df_benign = pd.DataFrame(
        data    = xts_benign,
        index   = list(range(num_samples)),
        columns = owasp_crs_ids
    )
    df_attack = pd.DataFrame(
        data    = xts_attack,
        index   = list(range(num_samples)),
        columns = owasp_crs_ids
    )
    df_adv    = pd.DataFrame(
        data    = xts_adv,
        index   = list(range(num_samples)),
        columns = owasp_crs_ids
    )

    # Load ML models
    model          = joblib.load(model_path)
    model_advtrain = joblib.load(model_advtrain_path)

    # -------------------
    # DATA PREPROCESSING
    # -------------------

    # Rules to be removed from the plots: rules that are not triggered by any benign, attack and adv. samples
    rules_to_remove = list()
    for rule in owasp_crs_ids:
        if df_attack[rule].sum() == 0 and df_benign[rule].sum() == 0 and df_adv[rule].sum(): 
            rules_to_remove.append(rule)
            print(f'SUM {df_adv[rule].sum()} of rule {rule} is zero')
            print("RULES NEVER TRIGGERED: {}".format(rules_to_remove))

    # Select rules related to target PL, sort them alphabetically
    if pl == 1:
        select_rules = sorted([rule for rule in owasp_crs_ids if (int(rule) in crs_rules_ids_pl1) and (rule not in rules_to_remove)])
    elif pl == 2:
        select_rules = sorted([rule for rule in owasp_crs_ids if (int(rule) in crs_rules_ids_pl2) and (rule not in rules_to_remove)])
    elif pl == 3:
        select_rules = sorted([rule for rule in owasp_crs_ids if (int(rule) in crs_rules_ids_pl3) and (rule not in rules_to_remove)])
    elif pl == 4:
        select_rules = sorted([rule for rule in owasp_crs_ids if (int(rule) in crs_rules_ids_pl4) and (rule not in rules_to_remove)])

    # Computing delta between adversarial and attack samples
    delta = (df_adv[select_rules] - df_attack[select_rules]).mean().values
    assert delta.flatten().shape[0] == len(select_rules)

    rules_delta  = {r: s for r, s in zip(select_rules, delta.tolist())}
    rules_delta  = dict(sorted(rules_delta.items(), key=lambda item: item[1], reverse=False))
    sorted_rules = list(rules_delta.keys())

    # Split rules into positive, negative and same slope
    pos_rules, neg_rules, same_rules = list(), list(), list()
    for rule, slope in rules_delta.items():
        if slope > 0:
            pos_rules.append(rule)
        elif slope < 0:
            neg_rules.append(rule)
        else:
            same_rules.append(rule)

    # Retrieve weights of the model and the adversarial trained model
    weights          = model.coef_.flatten()
    weights_advtrain = model_advtrain.coef_.flatten()
    
    # Sort weights according to the order of select_rules:
    weights          = np.array([weights[owasp_crs_ids.index(rule)] for rule in sorted_rules])
    weights_advtrain = np.array([weights_advtrain[owasp_crs_ids.index(rule)] for rule in sorted_rules])
    assert tuple(weights.shape) == tuple(delta.shape) and tuple(weights_advtrain.shape) == tuple(delta.shape)

    # Normalize weights
    weights          = weights / np.linalg.norm(weights)
    weights_advtrain = weights_advtrain / np.linalg.norm(weights_advtrain)

    # Filter rules according to the rules_selector (if provided)
    if rules_selector is not None and isinstance(rules_selector, list):
        weights                   = np.array([w for w, rule in zip(weights.tolist(), sorted_rules) if rule if rule[3:] in rules_selector])
        weights_advtrain          = np.array([w for w, rule in zip(weights_advtrain.tolist(), sorted_rules) if rule if rule[3:] in rules_selector])
        sorted_rules              = [rule for rule in sorted_rules if rule[3:] in rules_selector]

    adv_prob    = df_adv[sorted_rules].mean().values.tolist()
    attack_prob = df_attack[sorted_rules].mean().values.tolist() 
    
    # -----------------------------------------
    # PLOT ACTIVATION PROBABILITY OF THE RULES
    # -----------------------------------------
    fig_prob, ax_prob = plt.subplots(1, 1)
    
    df_plot = pd.DataFrame({
        'rules': sorted_rules * 2,
        'prob' : adv_prob + attack_prob,
        'type' : (['adversarial'] * len(sorted_rules)) + (['malicious'] * len(sorted_rules))
    })

    sns.barplot(
        data    = df_plot,
        x       = 'rules',
        y       = 'prob',
        hue     = 'type',
        dodge   = True,
        palette = {'adversarial': 'orange', 'malicious': 'deepskyblue'},
        ax      = ax_prob
    )
    rule_positions_short = {rule[3:]: idx for idx, rule in enumerate(sorted_rules)}

    # Get the indices for rules '101' and '550'
    idx_101 = rule_positions_short.get('101')
    idx_550 = rule_positions_short.get('550')

    # If '101' is not found, try '100' as a close substitute
    if idx_101 is None:
        idx_101 = rule_positions_short.get('100')

    # Add vertical lines to divide the plot into three parts
    if idx_101 is not None:
        ax_prob.axvline(x=idx_101 - 0.5, color='black', linestyle='-')
    if idx_550 is not None:
        ax_prob.axvline(x=idx_550 + 0.5, color='black', linestyle='-')

    # Set the x-ticks and labels
    ax_prob.set_xticks(range(len(sorted_rules)))
    ax_prob.set_xticklabels(
        [rule[3:] for rule in sorted_rules],
        rotation      = 75,
        ha            = 'right',
        rotation_mode = 'anchor'
    )
    
    legend = ax_prob.get_legend()
    ax_prob.legend(
        legend.legendHandles, 
        [t.get_text() for t in legend.texts], 
        loc='upper right', 
        fancybox=True, 
        shadow=False, 
        fontsize=legend_fontsize
    )

    ax_prob.set_xlabel('CRS SQLi Rules', fontsize=axis_labels_size, labelpad=10)
    ax_prob.set_ylabel('Activation probability', fontsize=axis_labels_size, labelpad=10)
    ax_prob.set_xmargin(0.05)
    ax_prob.set_ymargin(0.05)
    ax_prob.xaxis.set_tick_params(labelsize=tick_labels_size)
    ax_prob.yaxis.set_tick_params(labelsize=tick_labels_size)
    ax_prob.grid(visible=True, axis='both', color='gray', linestyle='dotted')
    ax_prob.set_title(model_name.upper().replace('_', ' '), fontsize=axis_labels_size)
    
    fig_prob.set_size_inches(16, 4)
    fig_prob.tight_layout()
    fig_prob.savefig(
        os.path.join(
            figs_save_path, 
            'comparison_attack_adv_{}{}.pdf'.format(model_name, '_wafamole' if DS_WAFAMOLE else '')
        ), 
        dpi         = 600,
        format      = 'pdf',
        bbox_inches = "tight"
    )

    # ------------------------ 
    # PLOT WEIGHTS COMPARISON
    # ------------------------
    fig_weights_cmp, ax_weights = plt.subplots(1, 1)
    
    df_plot = pd.DataFrame({
        'rules': sorted_rules * 2,
        'weight': weights_advtrain.tolist() + weights.tolist(),
        'type': (['ModSec-AdvLearn'] * len(sorted_rules)) + (['ModSec-Learn'] * len(sorted_rules))
    })
    
    sns.barplot(
        data    = df_plot,
        x       = 'rules',
        y       = 'weight',
        hue     = 'type',
        dodge   = True,
        palette = {'ModSec-AdvLearn': 'orange', 'ModSec-Learn': 'deepskyblue'},
        ax      = ax_weights
    )
    rule_positions_short = {rule[3:]: idx for idx, rule in enumerate(sorted_rules)}

    # Get the indices for rules '101' and '550'
    idx_101 = rule_positions_short.get('101')
    idx_550 = rule_positions_short.get('550')

    # If '101' is not found, try '100' as a close substitute
    if idx_101 is None:
        idx_101 = rule_positions_short.get('100')

    # Add vertical lines to divide the plot into three parts
    if idx_101 is not None:
        ax_weights.axvline(x=idx_101 - 0.5, color='black', linestyle='-')
    if idx_550 is not None:
        ax_weights.axvline(x=idx_550 + 0.5, color='black', linestyle='-')

    # Set the x-ticks and labels
    ax_weights.set_xticks(range(len(sorted_rules)))
    
    ax_weights.set_xticklabels(
        [rule[3:] for rule in sorted_rules], 
        rotation      = 75,
        ha            = 'right',
        rotation_mode = 'anchor'
    )
    
    legend = ax_weights.get_legend()
    
    ax_weights.legend(
        legend.legendHandles, 
        [t.get_text() for t in legend.texts], 
        loc      = 'lower left',
        fancybox = True,
        shadow   = False,
        fontsize = legend_fontsize
    )
    ax_weights.set_xlabel('CRS SQLi Rules', fontsize=axis_labels_size, labelpad=10)
    ax_weights.set_ylabel('Weight', fontsize=axis_labels_size, labelpad=0)
    ax_weights.set_xmargin(0.05)
    ax_weights.set_ymargin(0.05)
    ax_weights.xaxis.set_tick_params(labelsize=tick_labels_size)
    ax_weights.yaxis.set_tick_params(labelsize=tick_labels_size)
    ax_weights.grid(visible=True, axis='both', color='gray', linestyle='dotted')
    ax_weights.set_title(model_name.upper().replace('_', ' '), fontsize=axis_labels_size)
    
    fig_weights_cmp.set_size_inches(16, 4)
    fig_weights_cmp.tight_layout()
    fig_weights_cmp.savefig(
        os.path.join(
            figs_save_path, 
            'weights_comparison_{}{}.pdf'.format(model_name, '_wafamole' if DS_WAFAMOLE else '')
        ), 
        dpi         = 600,
        format      = 'pdf',
        bbox_inches = "tight"
    )


if __name__ == '__main__':
    settings          = toml.load('config.toml')
    crs_ids_path      = settings['crs_ids_path']
    figures_path      = settings['figures_path']
    crs_dir           = settings['crs_dir']

    if DS_WAFAMOLE:
        dataset_path     = settings['dataset_wafamole_path']
        models_path      = settings['models_wafamole_path']
    else:
        dataset_path     = settings['dataset_path']
        models_path      = settings['models_path']

    adv_payloads_path = os.path.join(
        dataset_path, 
        f'adv_train_test_inf_svm_pl4_rs20_100rounds.{"pkl" if DS_WAFAMOLE else "json"}'
    )

    legitimate_test_path = os.path.join(
        dataset_path, 
        f'legitimate_test.{"pkl" if DS_WAFAMOLE else "json"}'
    )    
    malicious_test_path = os.path.join(
        dataset_path, 
        f'malicious_test.{"pkl" if DS_WAFAMOLE else "json"}'
    )

    analyze_rules_importance(
        crs_rules_ids_file  = crs_ids_path,
        figs_save_path      = figures_path,
        benign_load_path    = legitimate_test_path,
        attacks_load_path   = malicious_test_path,
        adv_payloads_path   = adv_payloads_path,
        model_name          = 'inf_svm_t1',
        model_path          = os.path.join(
            models_path, 
            'inf_svm_pl4_t0.5.joblib'
        ),
        model_advtrain_path = os.path.join(
            models_path, 
            'adv_inf_svm_pl4_t1.joblib'
        ),
        legend_fontsize     = 12,
        axis_labels_size    = 16,
        tick_labels_size    = 14
    )
    
    
    