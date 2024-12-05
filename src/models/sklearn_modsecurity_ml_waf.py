"""
A wrapper for the ModSecurity CRS WAF, used by WAF-A-MoLE during the 
generation of adversarial examples.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from wafamole.models import SklearnModelWrapper
from wafamole.utils.check import type_check


class SklearnModSecurityMlWaf(SklearnModelWrapper):
    def __init__(
        self, 
        sklearn_clf_path, 
        crs_rules_ids_path, 
        rules_path, 
        crs_threshold, 
        crs_pl
    ):
        super(SklearnModSecurityMlWaf, self).__init__()
        super(SklearnModSecurityMlWaf, self).load(sklearn_clf_path)
        
        type_check(crs_rules_ids_path, str, "crs_rules_ids_path")
        type_check(rules_path, str, "rules_path")
        
        from src.extractor import ModSecurityFeaturesExtractor

        self._feat_builder = ModSecurityFeaturesExtractor(
            crs_rules_ids_path, 
            rules_path,
            crs_threshold, 
            crs_pl
        )
        
    def features_extraction(self, value: str):
        type_check(value, str, "value")
        feature_vector = self._feat_builder.extract_features_wafamole(value)
        return feature_vector

    def classify(self, value):
        feature_vector = self.features_extraction(value)
        try:
            if getattr(self._sklearn_classifier, "predict_proba", None) is not None:
                y_pred = self._sklearn_classifier.predict_proba(feature_vector)[0, 1]
            else:
                y_pred = self._sklearn_classifier.decision_function(feature_vector)[0]
            return y_pred
        except Exception:
            raise SystemExit("Sklearn internal error")