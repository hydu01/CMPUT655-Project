#!/bin/bash
python sarsa.py tabular_opt_emptyenv_config.json
python sarsa_plots.py tabular_opt_emptyenv_config.json
python sarsa_plots.py tabular_opt_distshift_config.json
python sarsa.py tabular_opt_distshift_config.json
python sarsa_plots.py tabular_opt_distshift_config.json
python sarsa.py tabular_opt_lavagap_config.json
python sarsa_plots.py tabular_opt_lavagap_config.json
