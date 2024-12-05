"""
A wrapper for the ModSecurity CRS WAF compliant with the payload
format by WAF-A-MoLE dataset.
"""

import os
import re

from urllib.parse import quote_plus
from ModSecurity import ModSecurity, RulesSet, Transaction, LogProperty
from wafamole.models import Model
from utils import type_check


class PyModSecurityWafamole(Model):
    """PyModSecurity WAF wrapper"""

    _SELECTED_RULES_FILES = [
        'REQUEST-901-INITIALIZATION.conf',
        'REQUEST-942-APPLICATION-ATTACK-SQLI.conf'
    ]

    def __init__(
        self, 
        rules_dir, 
        threshold = 5.0, 
        pl = 4
    ):
        """
        Constructor of PyModsecurity class
        
        Arguments
        ---------
            rules_dir: str
                Path to the directory containing the CRS rules.
            threshold: float
                The threshold to use for the ModSecurity CRS.
            pl: int from 1 to 4
                The paranoia level to use for the ModSecurity CRS.
        """
        type_check(rules_dir, str, 'rules_dir')
        type_check(threshold, float, 'threshold')
        type_check(pl, int, 'pl'),

        # Check if the paranoia level is valid
        if not 1 <= pl <= 4:
            raise ValueError(
                "Invalid value for pl input param: {}. Valid values are: [1, 2, 3, 4]"
                    .format(pl)
            )
            
        self._modsec                = ModSecurity()
        self._rules                 = RulesSet()
        self._threshold             = threshold
        self._debug                 = False
        self._rules_logger_callback = None

        # Load the ModSecurity CRS configuration files
        for conf_file in ['modsecurity.conf', f'crs-setup-pl{pl}.conf']:
            config_path = os.path.join('./modsec_config', conf_file)
            assert os.path.isfile(config_path)
            self._rules.loadFromUri(config_path)
    
        # Load the WAF rules
        for filename in __class__._SELECTED_RULES_FILES:
            rule_path = os.path.join(os.path.abspath(rules_dir), filename)
            assert os.path.isfile(rule_path)
            self._rules.loadFromUri(rule_path)

        if self._debug:
            print("[INFO] Using ModSecurity CRS with PL = {} and INBOUND THRESHOLD = {}"
                    .format(pl, threshold)
            )

    def extract_features(self, value):
        return value

    def classify(self, value: str):
        """
        Predict the class of the provided payload using the ModSecurity CRS WAF.
        
        Arguments:
        ----------
            value: str
                The payload to classify.
        
        Returns:
        --------
            score: float
                The score of the response if the output type is 'score', 0.0 if the
                output type is 'binary' and the response is good, 1.0 if the response
                is bad.
        """
        self._process_query(value)
        return self._process_response()

    def _process_query(self, payload: str):
        """
        Process the provided payload using the ModSecurity CRS WAF.

        Arguments:
        ----------
            payload: str
                The payload to process. 
        """
        # Create the rules logger
        rules_logger_cb = RulesLogger(
            threshold=self._threshold,
            debug=self._debug
        )
        # Set the rules logger callback to the ModSecurity CRS
        self._modsec.setServerLogCb2(
            rules_logger_cb, 
            LogProperty.RuleMessageLogProperty,
        )

        self._rules_logger_cb = rules_logger_cb

        # Remove encoding from the payload
        payload = quote_plus(payload)
        
        # Process the payload using the ModSecurity CRS
        transaction = Transaction(self._modsec, self._rules)
        transaction.processURI(
            "http://127.0.0.1/test?q={}".format(payload), 
            "GET", 
            "HTTP/1.1"
        )
        transaction.processRequestHeaders()
        transaction.processRequestBody()

    def _process_response(self) -> float:
        """
        Processes the HTTP response received from the ModSecurity CRS

        Returns:
        --------
            score: float
                The score of the response if the output type is 'score', 0.0 if the
                output type is 'binary' and the response is good, 1.0 if the response
                is bad.
        """
        if self._rules_logger_cb is not None:
            return self._rules_logger_cb.get_score()
        else:
            raise SystemExit("Callback to process rules not initialized")

    def _get_triggered_rules(self):
        """
        Returns the list of the triggered rules.

        Returns:
        --------
            list
                The list of the triggered rules.
        """
        return self._rules_logger_cb.get_triggered_rules()
    

class RulesLogger:
    _SEVERITY_SCORE = {
            2: 5,   # CRITICAL
            3: 4,   # ERROR
            4: 3,   # WARNING
            5: 2    # NOTICE
        }
    
    def _severity2score(self, severity):
        """
        Convert a severity level to a score.

        Parameters:
        ----------
            severity: int
                The severity of the rule.
        
        Returns:
        --------
            score: float
                The score of the severity.
        """
        return self._SEVERITY_SCORE[severity]
    
    def __init__(self, threshold=5.0, regex_rules_filter=None, debug=False):
        """
        Constructor of RulesLogger class

        Parameters:
        ----------
            threshold: float
                The threshold to use
            regex_rules_filter: str
                The regular expression to filter the rules.
            debug: bool
                Flag to enable the debug mode.
        """
        self._rules_triggered = []
        self._debug           = debug
        self._rules_filter    = re.compile(regex_rules_filter) if regex_rules_filter is not None \
                                    else re.compile('^.*')
        self._score           = 0.0
        self._threshold       = threshold
        self._status          = 200

    def __call__(self, data, rule_message):
        """
        Callback function to log the ModSecurity rules triggered

        Parameters:
        ----------
            data: object
                The data to log.
            rule_message: object
                The message of the rule.
        """
        if self._debug:
            print('[DEBUG] PyModSecurity rule logger callback')
            print("[DEBUG] ID: {}, Message: {}, Phase: {}, Severity: {}".format(
                rule_message.m_ruleId, 
                rule_message.m_message, 
                rule_message.m_phase,
                rule_message.m_severity
            ))
 
        elif re.match(self._rules_filter, str(rule_message.m_ruleId)) and \
                (str(rule_message.m_ruleId) not in self._rules_triggered):
            self._rules_triggered.append(str(rule_message.m_ruleId))

        # Update the score
        self._score += self._severity2score(rule_message.m_severity)
        
        if self._score >= self._threshold:
            self._status = 403

    def get_triggered_rules(self):
        """
        Get the rules triggered
        
        Returns:
        --------
            rules: list
                The list of rules triggered.
        """
        return self._rules_triggered

    def get_score(self):
        """
        Get the score of the request
        
        Returns:
        --------
            score: float
                The score of the request.
        """
        return self._score
    
    def get_status(self):
        """
        Get the status of the request

        Returns:
        --------
            request_status: int
                The status of the request.
        """
        return self._status