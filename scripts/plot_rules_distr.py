"""
This script is used to plot the distribution of the OWASP CRS rules for each paranoia level.
"""

import matplotlib.pyplot as plt
import os
import json
import numpy as np
import toml
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def plot_rules_distr_ms(
    crs_ids_path: str,
    figures_path: str,
):
    def fmt_label_pie(pct, allvals):
        absolute = int(np.round(pct/100.*np.sum(allvals)))
        ws = '\n' if absolute > 5 else ' '
        return f"{pct:.0f}%{ws}({absolute:d})"
    
    owasp_crs_rules_ids_filepath = crs_ids_path
    figs_save_path               = figures_path

    with open(owasp_crs_rules_ids_filepath, 'r') as fp:
        data = json.load(fp)
        owasp_crs_ids = data['rules_ids']

    info_rules               = [942011, 942012, 942013, 942014, 942015, 942016, 942017, 942018]
    crs_rules_ids_pl1_unique = [942100, 942140, 942151, 942160, 942170, 942190, 942220, 942230, 942240, 942250, 942270, 942280, 942290, 942320, 942350, 942360, 942500, 942540, 942560, 942550]
    crs_rules_ids_pl3_unique = [942251, 942490, 942420, 942431, 942460, 942511, 942530]
    crs_rules_ids_pl4_unique = [942421, 942432]
    crs_rules_ids_pl2_unique = list(
        set([int(rule) for rule in owasp_crs_ids]) - set(crs_rules_ids_pl1_unique + crs_rules_ids_pl3_unique + crs_rules_ids_pl4_unique + info_rules)
    )

    crs_rules_unique = dict()
    crs_rules_unique['pl1'] = crs_rules_ids_pl1_unique
    crs_rules_unique['pl2'] = crs_rules_ids_pl2_unique
    crs_rules_unique['pl3'] = crs_rules_ids_pl3_unique
    crs_rules_unique['pl4'] = crs_rules_ids_pl4_unique

    crs_rules_activated = dict()
    crs_rules_activated['pl1'] = crs_rules_ids_pl1_unique
    crs_rules_activated['pl2'] = crs_rules_activated['pl1'] + crs_rules_ids_pl2_unique
    crs_rules_activated['pl3'] = crs_rules_activated['pl2'] + crs_rules_ids_pl3_unique
    crs_rules_activated['pl4'] = crs_rules_activated['pl3'] + crs_rules_ids_pl4_unique

    warn_rules = [942430, 942420, 942431, 942460, 942421, 942432]

    fig_rules_weights, ax_rules_weights = plt.subplots()
    size = 0.3
    rules_weights_pl = []

    for _, rules in crs_rules_unique.items():
        data = {'critical': [], 'warn': []}
        for rule in rules:
            if rule in warn_rules:
                data['warn'].append(rule)
            else:
                data['critical'].append(rule)
        rules_weights_pl.append([len(pl_values) for pl_values in data.values()])
    
    rules_weights_pl = np.array(rules_weights_pl)
    labels_outer     = ['PL1', 'PL2', 'PL3', 'PL4']
    labels_inner     = ['CRITICAL', 'WARN'] * len(labels_outer)

    # outer slices
    wedges_outer = ax_rules_weights.pie(rules_weights_pl.sum(axis=1), radius=1, 
    colors      = ['mediumseagreen', 'springgreen', 'lightblue', 'deepskyblue'],
    wedgeprops  = dict(width=size, edgecolor='w'),
    autopct     = lambda pct: fmt_label_pie(pct, rules_weights_pl.sum(axis=1)),
    pctdistance = (1-size/2))

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw         = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center")

    for _, (p, label) in enumerate(zip(wedges_outer[0], labels_outer)):
        ang                 = (p.theta2 - p.theta1)/2. + p.theta1
        y                   = np.sin(np.deg2rad(ang))
        x                   = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle     = f"angle,angleA=0,angleB={ang}"
        
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax_rules_weights.annotate(
            label, 
            xy                  = (x, y),
            xytext              = (1.35*np.sign(x), 1.4*y),
            fontsize            = 12,
            horizontalalignment = horizontalalignment,      **kw
        )

    # inner slices
    type_to_color = {
        'CRITICAL': 'tomato',
        'WARN'    : 'limegreen'
    }

    inner_values, inner_colors = list(), list()
    for val, label in zip(rules_weights_pl.flatten().tolist(), labels_inner):
        if val > 0:
            inner_values.append(val)
            inner_colors.append(type_to_color[label])
    
    inner_labels = list()
    for row in rules_weights_pl:
        for val in row:
            if val > 0:
                ws = '\n'
                str_val = '{:.0f}%{}({:d})'.format(val / np.sum(row) * 100, ws, val)
                inner_labels.append(str_val)

    wedges, labels_list = ax_rules_weights.pie(
        inner_values, 
        radius        = 1-size,
        colors        = inner_colors,
        wedgeprops    = dict(width=size, edgecolor='w'),
        labels        = inner_labels,
        labeldistance = 0.76,
        rotatelabels  = False
    )

    for _, label in enumerate(labels_list):
        label.update({"horizontalalignment": "center",  "verticalalignment": "center"})

    ax_rules_weights.legend(
        wedges[1:3], 
        ['CRITICAL (5)', 'WARNING (3)'],
        loc            = 'upper center',
        bbox_to_anchor = (0.5, -0.005),
        fancybox       = True,
        shadow         = True,
        ncol           = 4,
        fontsize       = 12
    )

    fig_rules_weights.set_size_inches(7, 7.5)
    fig_rules_weights.tight_layout()
    fig_rules_weights.savefig(
        os.path.join(figs_save_path, 'ms_weights_analysis_pl_weights.pdf'), 
        dpi         = 600,
        format      = 'pdf',
        bbox_inches = "tight"
    )

if __name__ == '__main__':
    settings         = toml.load('config.toml')
    crs_ids_path     = settings['crs_ids_path']
    figures_path     = settings['figures_path']

    plot_rules_distr_ms(
        crs_ids_path = crs_ids_path,
        figures_path = figures_path
    )