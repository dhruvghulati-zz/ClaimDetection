#!/bin/bash

vw --csoaa 24 data/output/cost/openCostClassifier.dat -f data/output/cost/csoaa.model
vw -t -i data/output/cost/csoaa.model data/output/cost/openCostClassifier.dat -p data/output/cost/csoaa.predict

vw --csoaa 16 data/output/cost/closedCostClassifier.dat -f data/output/cost/csoaa.model
vw -t -i data/output/cost/csoaa.model data/output/cost/closedCostClassifier.dat -p data/output/cost/csoaa_closed.predict

vw --csoaa 24 data/output/cost/openCostClassifier_1.dat -f data/output/cost/csoaa.model
vw -t -i data/output/cost/csoaa.model data/output/cost/openCostClassifier_1.dat -p data/output/cost/csoaa_1.predict

vw --csoaa 16 data/output/cost/closedCostClassifier_1.dat -f data/output/cost/csoaa.model
vw -t -i data/output/cost/csoaa.model data/output/cost/closedCostClassifier_1.dat -p data/output/cost/csoaa_closed_1.predict

vw --csoaa 24 data/output/cost/openCostClassifier_2.dat -f data/output/cost/csoaa.model
vw -t -i data/output/cost/csoaa.model data/output/cost/openCostClassifier_2.dat -p data/output/cost/csoaa_2.predict

vw --csoaa 16 data/output/cost/closedCostClassifier_2.dat -f data/output/cost/csoaa.model
vw -t -i data/output/cost/csoaa.model data/output/cost/closedCostClassifier_2.dat -p data/output/cost/csoaa_closed_2.predict
