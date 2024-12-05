"""
This script is used to evaluate the performance of the models trained with adversarial training.
"""

import os
import sys

import joblib
import matplotlib.pyplot as plt
import toml

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import DataLoader
from src.extractor import ModSecurityFeaturesExtractor
from src.utils.plotting import plot_roc


if __name__ == "__main__":
    settings     = toml.load("config.toml")
    crs_dir      = settings["crs_dir"]
    crs_ids_path = settings["crs_ids_path"]
    figures_path = settings["figures_path"]
    models       = settings["params"]["models"]
    other_models = settings["params"]["other_models"]
    penalties    = settings["params"]["penalties"]
    pl           = 4

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    datasets = ["wafamole", "modsec-learn"]

    for dataset_idx, dataset in enumerate(datasets):
        ds_wafamole  = dataset == "wafamole"
        models_path  = settings["models_wafamole_path"] if ds_wafamole else settings["models_path"]
        dataset_path = settings["dataset_wafamole_path"] if ds_wafamole else settings["dataset_path"]
        ext          = "pkl" if ds_wafamole else "json"

        # Preparing path templates for the adversarial dataset
        adv_infsvm_pl_path = os.path.join(
            dataset_path, f"adv_test_inf_svm_pl%s_rs20_100rounds.{ext}"
        )
        ms_adv_pl_path = os.path.join(
            dataset_path, f"adv_test_ms_pl%s_rs20_100rounds.{ext}"
        )
        adv_log_reg_l1_path = os.path.join(
            dataset_path, f"adv_test_log_reg_l1_pl%s_rs20_100rounds.{ext}"
        )
        adv_log_reg_l2_path = os.path.join(
            dataset_path, f"adv_test_log_reg_l2_pl%s_rs20_100rounds.{ext}"
        )
        adv_svm_linear_l1_path = os.path.join(
            dataset_path, f"adv_test_svm_linear_l1_pl%s_rs20_100rounds.{ext}"
        )
        adv_svm_linear_l2_path = os.path.join(
            dataset_path, f"adv_test_svm_linear_l2_pl%s_rs20_100rounds.{ext}"
        )
        adv_rf_path = os.path.join(
            dataset_path, f"adv_test_rf_pl%s_rs20_100rounds.{ext}"
        )
        train_test_adv_infsvm_pl_path = os.path.join(
            dataset_path, f"adv_train_test_inf_svm_pl%s_rs20_100rounds.{ext}"
        )
        train_test_adv_log_reg_l1_path = os.path.join(
            dataset_path, f"adv_train_test_log_reg_l1_pl%s_rs20_100rounds.{ext}"
        )
        train_test_adv_log_reg_l2_path = os.path.join(
            dataset_path, f"adv_train_test_log_reg_l2_pl%s_rs20_100rounds.{ext}"
        )
        train_test_adv_svm_linear_l1_path = os.path.join(
            dataset_path, f"adv_train_test_svm_linear_l1_pl%s_rs20_100rounds.{ext}"
        )
        train_test_adv_svm_linear_l2_path = os.path.join(
            dataset_path, f"adv_train_test_svm_linear_l2_pl%s_rs20_100rounds.{ext}"
        )
        train_test_adv_rf_path = os.path.join(
            dataset_path, f"adv_train_test_rf_pl%s_rs20_100rounds.{ext}"
        )
        malicious_path       = os.path.join(dataset_path, f"malicious_test.{ext}")
        legitimate_test_path = os.path.join(dataset_path, f"legitimate_test.{ext}")

        print(f"[INFO] Loading dataset for {dataset}...")

        loader = DataLoader(
            legitimate_path = legitimate_test_path,
            malicious_path  = malicious_path
        )

        if ds_wafamole:
            test_data = loader.load_data_pkl()
        else:
            test_data = loader.load_data()

        # ---------------------
        # STARTING EXPERIMENTS
        # ---------------------
        # Loading the adversarial datasets
        adv_inf_svm_loader = DataLoader(
            malicious_path  = adv_infsvm_pl_path % pl,
            legitimate_path = legitimate_test_path
        )
        adv_log_reg_l1_loader = DataLoader(
            malicious_path  = adv_log_reg_l1_path % pl,
            legitimate_path = legitimate_test_path,
        )
        adv_log_reg_l2_loader = DataLoader(
            malicious_path  = adv_log_reg_l2_path % pl,
            legitimate_path = legitimate_test_path,
        )
        adv_svm_linear_l1_loader = DataLoader(
            malicious_path  = adv_svm_linear_l1_path % pl,
            legitimate_path = legitimate_test_path,
        )
        adv_svm_linear_l2_loader = DataLoader(
            malicious_path  = adv_svm_linear_l2_path % pl,
            legitimate_path = legitimate_test_path,
        )
        adv_rf_loader = DataLoader(
            malicious_path  = adv_rf_path % pl,
            legitimate_path = legitimate_test_path
        )
        train_test_adv_inf_svm_loader = DataLoader(
            malicious_path  = train_test_adv_infsvm_pl_path % pl,
            legitimate_path = legitimate_test_path,
        )
        train_test_adv_log_reg_l1_loader = DataLoader(
            malicious_path  = train_test_adv_log_reg_l1_path % pl,
            legitimate_path = legitimate_test_path,
        )
        train_test_adv_log_reg_l2_loader = DataLoader(
            malicious_path  = train_test_adv_log_reg_l2_path % pl,
            legitimate_path = legitimate_test_path,
        )
        train_test_adv_svm_linear_l1_loader = DataLoader(
            malicious_path  = train_test_adv_svm_linear_l1_path % pl,
            legitimate_path = legitimate_test_path,
        )
        train_test_adv_svm_linear_l2_loader = DataLoader(
            malicious_path  = train_test_adv_svm_linear_l2_path % pl,
            legitimate_path = legitimate_test_path,
        )
        train_test_adv_rf_loader = DataLoader(
            malicious_path  = train_test_adv_rf_path % pl,
            legitimate_path = legitimate_test_path,
        )

        if ds_wafamole:
            adv_test_inf_svm_data             = adv_inf_svm_loader.load_data_pkl()
            adv_test_log_reg_l1_data          = adv_log_reg_l1_loader.load_data_pkl()
            adv_test_log_reg_l2_data          = adv_log_reg_l2_loader.load_data_pkl()
            adv_test_svm_linear_l1_data       = adv_svm_linear_l1_loader.load_data_pkl()
            adv_test_svm_linear_l2_data       = adv_svm_linear_l2_loader.load_data_pkl()
            adv_test_rf_data                  = adv_rf_loader.load_data_pkl()
            adv_train_test_inf_svm_data       = train_test_adv_inf_svm_loader.load_data_pkl()
            adv_train_test_log_reg_l1_data    = train_test_adv_log_reg_l1_loader.load_data_pkl()
            adv_train_test_log_reg_l2_data    = train_test_adv_log_reg_l2_loader.load_data_pkl()
            adv_train_test_svm_linear_l1_data = train_test_adv_svm_linear_l1_loader.load_data_pkl()
            adv_train_test_svm_linear_l2_data = train_test_adv_svm_linear_l2_loader.load_data_pkl()
            adv_train_test_rf_data            = train_test_adv_rf_loader.load_data_pkl()
        else:
            adv_test_inf_svm_data             = adv_inf_svm_loader.load_data()
            adv_test_log_reg_l1_data          = adv_log_reg_l1_loader.load_data()
            adv_test_log_reg_l2_data          = adv_log_reg_l2_loader.load_data()
            adv_test_svm_linear_l1_data       = adv_svm_linear_l1_loader.load_data()
            adv_test_svm_linear_l2_data       = adv_svm_linear_l2_loader.load_data()
            adv_test_rf_data                  = adv_rf_loader.load_data()
            adv_train_test_inf_svm_data       = train_test_adv_inf_svm_loader.load_data()
            adv_train_test_log_reg_l1_data    = train_test_adv_log_reg_l1_loader.load_data()
            adv_train_test_log_reg_l2_data    = train_test_adv_log_reg_l2_loader.load_data()
            adv_train_test_svm_linear_l1_data = train_test_adv_svm_linear_l1_loader.load_data()
            adv_train_test_svm_linear_l2_data = train_test_adv_svm_linear_l2_loader.load_data()
            adv_train_test_rf_data            = train_test_adv_rf_loader.load_data()

        print(f"[INFO] Extracting features for PL {pl} in {dataset} dataset...")

        extractor = ModSecurityFeaturesExtractor(
            crs_ids_path=crs_ids_path, crs_path=crs_dir, crs_pl=pl
        )

        xts                             , yts                              = extractor.extract_features(test_data)
        adv_inf_svm_xts                 , adv_inf_svm_yts                  = extractor.extract_features(adv_test_inf_svm_data)
        adv_log_reg_l1_xts              , adv_log_reg_l1_yts               = extractor.extract_features(adv_test_log_reg_l1_data)
        adv_log_reg_l2_xts              , adv_log_reg_l2_yts               = extractor.extract_features(adv_test_log_reg_l2_data)
        adv_svm_linear_l1_xts           , adv_svm_linear_l1_yts            = extractor.extract_features(adv_test_svm_linear_l1_data)
        adv_svm_linear_l2_xts           , adv_svm_linear_l2_yts            = extractor.extract_features(adv_test_svm_linear_l2_data)
        adv_rf_xts                      , adv_rf_yts                       = extractor.extract_features(adv_test_rf_data)
        train_test_adv_inf_svm_xts      , train_test_adv_inf_svm_yts       = extractor.extract_features(adv_train_test_inf_svm_data)
        train_test_adv_log_reg_l1_xts   , train_test_adv_log_reg_l1_yts    = extractor.extract_features(adv_train_test_log_reg_l1_data)
        train_test_adv_log_reg_l2_xts   , train_test_adv_log_reg_l2_yts    = extractor.extract_features(adv_train_test_log_reg_l2_data)
        train_test_adv_svm_linear_l1_xts, train_test_adv_svm_linear_l1_yts = extractor.extract_features(adv_train_test_svm_linear_l1_data)
        train_test_adv_svm_linear_l2_xts, train_test_adv_svm_linear_l2_yts = extractor.extract_features(adv_train_test_svm_linear_l2_data)
        train_test_adv_rf_xts           , train_test_adv_rf_yts            = extractor.extract_features(adv_train_test_rf_data)

        for model_name in other_models:
            print(
                f"[INFO] Evaluating {model_name} model for PL {pl} in {dataset} dataset..."
            )

            if model_name == "infsvm":
                label_legend     = "SecSVM"
                normal_settings  = {"color": "darkmagenta"}
                adv_settings     = {"color": "darkmagenta", "linestyle": "dashed"}
                model            = joblib.load(
                    os.path.join(models_path, f"inf_svm_pl{pl}_t0.5.joblib")
                )
                adv_model = joblib.load(
                    os.path.join(models_path, f"adv_inf_svm_pl{pl}_t1.joblib")
                )
                y_scores                = model.decision_function(xts)
                y_scores_adv_train      = adv_model.decision_function(xts)
                adv_y_scores            = model.decision_function(adv_inf_svm_xts)
                train_test_adv_y_scores = adv_model.decision_function(
                    train_test_adv_inf_svm_xts
                )

                plot_roc(
                    dataset_idx,
                    yts,
                    y_scores,
                    label_legend       = label_legend ,
                    ax                 = axs[dataset_idx, 0] ,
                    settings           = normal_settings ,
                    plot_rand_guessing = False ,
                    log_scale          = True ,
                    update_roc_values  = True if pl == 1 else False,
                    include_zoom       = False ,
                    pl                 = pl ,
                )
                plot_roc(
                    dataset_idx,
                    adv_inf_svm_yts,
                    adv_y_scores,
                    ax                 = axs[dataset_idx, 0] ,
                    settings           = adv_settings ,
                    plot_rand_guessing = False ,
                    log_scale          = True ,
                    update_roc_values  = True if pl == 1 else False,
                    include_zoom       = False ,
                    pl                 = pl ,
                )
                plot_roc(
                    dataset_idx,
                    yts,
                    y_scores_adv_train,
                    label_legend       = label_legend ,
                    ax                 = axs[dataset_idx, 1] ,
                    settings           = normal_settings ,
                    plot_rand_guessing = False ,
                    log_scale          = True ,
                    update_roc_values  = True if pl == 1 else False,
                    include_zoom       = False ,
                    pl                 = pl ,
                )
                plot_roc(
                    dataset_idx,
                    train_test_adv_inf_svm_yts,
                    train_test_adv_y_scores,
                    label_legend       = label_legend ,
                    ax                 = axs[dataset_idx, 1] ,
                    settings           = adv_settings ,
                    plot_rand_guessing = False ,
                    log_scale          = True ,
                    update_roc_values  = True if pl == 1 else False,
                    include_zoom       = False ,
                    pl                 = pl ,
                )

            elif model_name == "rf":
                label_legend     = "RF"
                normal_settings  = {"color": "green"}
                adv_settings     = {"color": "green", "linestyle": "dashed"}
                model            = joblib.load(os.path.join(models_path, f"rf_pl{pl}.joblib"))
                adv_model        = joblib.load(
                    os.path.join(models_path, f"adv_rf_pl{pl}.joblib")
                )
                y_scores                = model.predict_proba(xts)[:, 1]
                y_scores_adv_train      = adv_model.predict_proba(xts)[:, 1]
                adv_y_scores            = model.predict_proba(adv_rf_xts)[:, 1]
                train_test_adv_y_scores = adv_model.predict_proba(
                    train_test_adv_rf_xts
                )[:, 1]

                plot_roc(
                    dataset_idx,
                    yts,
                    y_scores,
                    label_legend       = label_legend ,
                    ax                 = axs[dataset_idx, 0] ,
                    settings           = normal_settings ,
                    plot_rand_guessing = False ,
                    log_scale          = True ,
                    update_roc_values  = True if pl == 1 else False,
                    include_zoom       = False ,
                    pl                 = pl ,
                )
                plot_roc(
                    dataset_idx,
                    adv_rf_yts,
                    adv_y_scores,
                    ax                 = axs[dataset_idx, 0] ,
                    settings           = adv_settings ,
                    plot_rand_guessing = False ,
                    log_scale          = True ,
                    update_roc_values  = True if pl == 1 else False,
                    include_zoom       = False ,
                    pl                 = pl ,
                )
                plot_roc(
                    dataset_idx,
                    yts,
                    y_scores_adv_train,
                    label_legend       = label_legend ,
                    ax                 = axs[dataset_idx, 1] ,
                    settings           = normal_settings ,
                    plot_rand_guessing = False ,
                    log_scale          = True ,
                    update_roc_values  = True if pl == 1 else False,
                    include_zoom       = False ,
                    pl                 = pl ,
                )
                plot_roc(
                    dataset_idx,
                    train_test_adv_rf_yts,
                    train_test_adv_y_scores,
                    label_legend       = label_legend ,
                    ax                 = axs[dataset_idx, 1] ,
                    settings           = adv_settings ,
                    plot_rand_guessing = False ,
                    log_scale          = True ,
                    update_roc_values  = True if pl == 1 else False,
                    include_zoom       = False ,
                    pl                 = pl ,
                )

        for model_name in models:
            print(
                f"[INFO] Evaluating {model_name} model for PL {pl} in {dataset} dataset..."
            )

            for penalty in penalties:
                if model_name == "svc":
                    label_legend = f"SVM – $\\ell_{{{penalty[1]}}}$"
                    normal_settings = {
                        "color"    : "blue" if penalty == "l1" else "aqua",
                        "linestyle": "solid",
                    }
                    adv_settings = {
                        "color"    : "blue" if penalty == "l1" else "aqua",
                        "linestyle": "dashed",
                    }
                    model = joblib.load(
                        os.path.join(models_path, f"linear_svc_pl{pl}_{penalty}.joblib")
                    )
                    adv_model = joblib.load(
                        os.path.join(
                            models_path, f"adv_linear_svc_pl{pl}_{penalty}.joblib"
                        )
                    )

                    y_scores           = model.decision_function(xts)
                    y_scores_adv_train = adv_model.decision_function(xts)

                    if penalty == "l1":
                       adv_y_scores             = model.decision_function(adv_svm_linear_l1_xts)
                       adv_yts                  = adv_svm_linear_l1_yts
                       train_test_adv_yts       = train_test_adv_svm_linear_l1_yts
                       train_test_adv_y_scores  = adv_model.decision_function(
                            train_test_adv_svm_linear_l1_xts
                        )
                    else:
                        adv_y_scores            = model.decision_function(adv_svm_linear_l2_xts)
                        adv_yts                 = adv_svm_linear_l2_yts
                        train_test_adv_yts      = train_test_adv_svm_linear_l2_yts
                        train_test_adv_y_scores = adv_model.decision_function(
                            train_test_adv_svm_linear_l2_xts
                        )

                    plot_roc(
                        dataset_idx,
                        yts,
                        y_scores,
                        label_legend       = label_legend ,
                        ax                 = axs[dataset_idx, 0] ,
                        settings           = normal_settings ,
                        plot_rand_guessing = False ,
                        log_scale          = True ,
                        update_roc_values  = True if pl == 1 else False,
                        include_zoom       = False ,
                        pl                 = pl ,
                    )
                    plot_roc(
                        dataset_idx,
                        adv_yts,
                        adv_y_scores,
                        ax                 = axs[dataset_idx, 0] ,
                        settings           = adv_settings ,
                        plot_rand_guessing = False ,
                        log_scale          = True ,
                        update_roc_values  = True if pl == 1 else False,
                        include_zoom       = False ,
                        pl                 = pl ,
                    )
                    plot_roc(
                        dataset_idx,
                        yts,
                        y_scores_adv_train,
                        label_legend       = label_legend ,
                        ax                 = axs[dataset_idx, 1] ,
                        settings           = normal_settings ,
                        plot_rand_guessing = False ,
                        log_scale          = True ,
                        update_roc_values  = True if pl == 1 else False,
                        include_zoom       = False ,
                        pl                 = pl ,
                    )
                    plot_roc(
                        dataset_idx,
                        train_test_adv_yts,
                        train_test_adv_y_scores,
                        label_legend       = label_legend ,
                        ax                 = axs[dataset_idx, 1] ,
                        settings           = adv_settings ,
                        plot_rand_guessing = False ,
                        log_scale          = True ,
                        update_roc_values  = True if pl == 1 else False,
                        include_zoom       = False ,
                        pl                 = pl ,
                    )

                elif model_name == "log_reg":
                    label_legend     = f"LR – $\\ell_{{{penalty[1]}}}$"
                    normal_settings  = {
                        "color"    : "orange" if penalty == "l1" else "chocolate",
                        "linestyle": "solid",
                    }
                    adv_settings = {
                        "color"    : "orange" if penalty == "l1" else "chocolate",
                        "linestyle": "dashed",
                    }
                    model = joblib.load(
                        os.path.join(models_path, f"log_reg_pl{pl}_{penalty}.joblib")
                    )
                    adv_model = joblib.load(
                        os.path.join(
                            models_path, f"adv_log_reg_pl{pl}_{penalty}.joblib"
                        )
                    )

                    y_scores           = model.predict_proba(xts)[:, 1]
                    y_scores_adv_train = adv_model.predict_proba(xts)[:, 1]

                    if penalty == "l1":
                        adv_y_scores             = model.predict_proba(adv_log_reg_l1_xts)[:, 1]
                        adv_yts                  = adv_log_reg_l1_yts
                        train_test_adv_yts       = train_test_adv_log_reg_l1_yts
                        train_test_adv_y_scores  = adv_model.predict_proba(
                            train_test_adv_log_reg_l1_xts
                        )[:, 1]
                    else:
                        adv_y_scores            = model.predict_proba(adv_log_reg_l2_xts)[:, 1]
                        adv_yts                 = adv_log_reg_l2_yts
                        train_test_adv_yts      = train_test_adv_log_reg_l2_yts
                        train_test_adv_y_scores = adv_model.predict_proba(
                            train_test_adv_log_reg_l2_xts
                        )[:, 1]

                    plot_roc(
                        dataset_idx,
                        yts,
                        y_scores,
                        label_legend       = label_legend ,
                        ax                 = axs[dataset_idx, 0] ,
                        settings           = normal_settings ,
                        plot_rand_guessing = False ,
                        log_scale          = True ,
                        update_roc_values  = True if pl == 1 else False,
                        include_zoom       = False ,
                        pl                 = pl ,
                    )
                    plot_roc(
                        dataset_idx,
                        adv_yts,
                        adv_y_scores,
                        ax                 = axs[dataset_idx, 0] ,
                        settings           = adv_settings ,
                        plot_rand_guessing = False ,
                        log_scale          = True ,
                        update_roc_values  = True if pl == 1 else False,
                        include_zoom       = False ,
                        pl                 = pl ,
                    )
                    plot_roc(
                        dataset_idx,
                        yts,
                        y_scores_adv_train,
                        label_legend       = label_legend ,
                        ax                 = axs[dataset_idx, 1] ,
                        settings           = normal_settings ,
                        plot_rand_guessing = False ,
                        log_scale          = True ,
                        update_roc_values  = True if pl == 1 else False,
                        include_zoom       = False ,
                        pl                 = pl ,
                    )
                    plot_roc(
                        dataset_idx,
                        train_test_adv_yts,
                        train_test_adv_y_scores,
                        label_legend       = label_legend ,
                        ax                 = axs[dataset_idx, 1] ,
                        settings           = adv_settings ,
                        plot_rand_guessing = False ,
                        log_scale          = True ,
                        update_roc_values  = True if pl == 1 else False,
                        include_zoom       = False ,
                        pl                 = pl ,
                    )

    # Final global settings for the figure
    axs[0, 0].set_title("ModSec-Learn", fontsize=16)
    axs[0, 1].set_title("ModSec-AdvLearn", fontsize=16)
    axs[0, 0].set_ylabel("True Positive Rate (TPR)", fontsize=16, labelpad=10)
    axs[1, 0].set_ylabel("True Positive Rate (TPR)", fontsize=16, labelpad=10)
    axs[0, 1].yaxis.set_ticklabels([])
    axs[1, 1].yaxis.set_ticklabels([])

    for ax_row in axs:
        for ax in ax_row:
            ax.xaxis.set_tick_params(labelsize=14)
            ax.yaxis.set_tick_params(labelsize=14)
            ax.xaxis.label.set_size(16)
            ax.yaxis.label.set_size(16)
            ax.set_ylim([0.8, 1.0])

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc            = "upper center",
        bbox_to_anchor = (0.54, -0.01),
        fancybox       = True,
        shadow         = True,
        ncol           = 7,
        fontsize       = 13,
    )
    fig.tight_layout(pad=1.4)
    fig.savefig(
        os.path.join(figures_path, "roc_curves_adv_training.pdf"),
        dpi         = 600,
        format      = "pdf",
        bbox_inches = "tight",
    )
