#!/bin/bash

vw --csoaa 24 data/output/cost/open_cost_1.dat -f data/output/cost/csoaa.model
vw -t -i data/output/cost/csoaa.model data/output/cost/open_cost_1.dat -p data/output/cost/open_cost_1.predict

vw --csoaa 24 data/output/cost/open_cost_2.dat -f data/output/cost/csoaa.model
vw -t -i data/output/cost/csoaa.model data/output/cost/open_cost_2.dat -p data/output/cost/open_cost_2.predict

vw --csoaa 24 data/output/cost/open_cost_3.dat -f data/output/cost/csoaa.model
vw -t -i data/output/cost/csoaa.model data/output/cost/open_cost_3.dat -p data/output/cost/open_cost_3.predict

vw --csoaa 16 data/output/cost/closed_cost_1.dat -f data/output/cost/csoaa.model
vw -t -i data/output/cost/csoaa.model data/output/cost/closed_cost_3.dat -p data/output/cost/closed_cost_1.predict

vw --csoaa 16 data/output/cost/closed_cost_2.dat -f data/output/cost/csoaa.model
vw -t -i data/output/cost/csoaa.model data/output/cost/closed_cost_3.dat -p data/output/cost/closed_cost_2.predict

vw --csoaa 16 data/output/cost/closed_cost_3.dat -f data/output/cost/csoaa.model
vw -t -i data/output/cost/csoaa.model data/output/cost/closed_cost_3.dat -p data/output/cost/closed_cost_3.predict
