"""
The script extracts the rule IDs from the CRS ruleset file REQUEST-942-APPLICATION-ATTACK-SQLI.conf,
and save them into a JSON file.
"""

import os
import re
import toml
import json


if __name__ == '__main__':
    config       = toml.load('config.toml')
    rules_dir    = config['crs_dir']
    crs_ids_path = config['crs_ids_path']
    exclude_ids  = [f'9420{num}' for num in range(11, 19)] # rules only used for logging purpose

    sqli_rule_path = os.path.join(rules_dir, 'REQUEST-942-APPLICATION-ATTACK-SQLI.conf')
    with open(sqli_rule_path) as rule_file:
        content = rule_file.read()

    sqli_rule_ids = re.findall(r'id:(\d{6})', content)
    sqli_rule_ids = [id for id in sqli_rule_ids if id not in exclude_ids]
    
    # Dump on file
    sqli_rule_ids.sort()
    data = {"rules_ids": sqli_rule_ids}
        
    with open(crs_ids_path, 'w') as file:
        json.dump(data, file, indent=4)