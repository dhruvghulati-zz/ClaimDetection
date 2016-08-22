#!/bin/bash

# python src/main/costSensitiveClassifier.py data/output/predictedProperties.json data/output/devLabels.json data/featuresKept.json data/output/cost/

# END=2

# vw --csoaa_ldf=mc --loss_function=logistic -d data/output/zero/cost_test/closed_cost_1closed_ld.dat -f data/output/zero/cost_test/csoaa_ldf.model --probabilities
# vw -t -i data/output/zero/cost_test/csoaa_ldf.model -d data/output/zero/cost_test/ldf_closed_test.dat -p data/output/zero/cost_test/probs.predict

vw --csoaa 16 -d data/output/zero/cost_test/closed_cost_1.dat -f data/output/zero/cost_test/csoaa.model
vw -t -i data/output/zero/cost_test/csoaa.model -d data/output/zero/cost_test/test.dat --raw_predictions data/output/zero/cost_test/closed_cost_1_raw.dat

# for i in $(seq 1 $END); do
#   # vw --csoaa 24 data/output/zero/cost_test/open_cost_$i.dat -f data/output/zero/cost_test/open_csoaa_$i.model
#   # vw -t -i data/output/zero/cost_test/open_csoaa_$i.model data/output/zero/cost_test/test.dat -p data/output/zero/cost_test/open_cost_$i.predict
#
#   # vw --csoaa 24 data/output/zero/cost_test/open_cost_$i.dat -f data/output/zero/cost_test/open_cost_$i.model
#   # vw -t -i data/output/zero/cost_test/open_cost_$i.model data/output/zero/cost_test/test.dat -p data/output/zero/cost_test/open_cost_$i.predict
#   #
#   # # vw --csoaa 16 data/output/zero/cost_test/closed_cost_$i.dat -f data/output/zero/cost_test/closed_csoaa_$i.model
#   # # vw -t -i data/output/zero/cost_test/closed_csoaa_$i.model data/output/zero/cost_test/test.dat -p data/output/zero/cost_test/closed_cost_$i.predict
#   #
#   # vw --csoaa 16 data/output/zero/cost_test/closed_cost_$i.dat -f data/output/zero/cost_test/closed_cost_$i.model --link=logistic
#   #
#   # vw -t -i data/output/zero/cost_test/closed_cost_$i.model data/output/zero/cost_test/test.dat -p data/output/zero/cost_test/closed_cost_$i.predict
#
#   # vw --csoaa_ldf=mc --loss_function=logistic -d data/output/zero/cost_test/closed_cost_1closed_ld.dat -f data/output/zero/cost_test/csoaa_ldf.model --probabilities
#   # vw -t -i data/output/zero/cost_test/csoaa_ldf.model -d data/output/zero/cost_test/ldf_closed_test.dat -p data/output/zero/cost_test/probs.predict --probabilities
#
#   # vw --csoaa 16 data/output/cost/single_open_cost_$i.dat -f data/output/cost/csoaa.model
#   # vw -t -i data/output/cost/csoaa.model data/output/cost/test.dat -p data/output/cost/single_open_cost_$i.predict
#   #
#   # vw --csoaa 16 data/output/cost/single_closed_cost_$i.dat -f data/output/cost/csoaa.model
#   # vw -t -i data/output/cost/csoaa.model data/output/cost/test.dat -p data/output/cost/single_closed_cost_$i.predict
# done

# python src/main/costPredictions.py data/output/cost/models.txt data/output/cost/open_label_mapping.txt data/output/cost/closed_label_mapping.txt data/output/devLabels.json data/output/cost/ data/output/cost/summaryEvaluationCost.csv

# vw --csoaa 24 data/output/cost/open_cost_1.dat -f data/output/cost/csoaa.model
# vw -t -i data/output/cost/csoaa.model data/output/cost/test.dat -p data/output/cost/open_cost_1.predict
#
# # vw --csoaa 24 data/output/cost/open_cost_2.dat -f data/output/cost/csoaa.model
# # vw -t -i data/output/cost/csoaa.model data/output/cost/open_cost_2.dat -p data/output/cost/open_cost_2.predict
# #
# # vw --csoaa 24 data/output/cost/open_cost_3.dat -f data/output/cost/csoaa.model
# # vw -t -i data/output/cost/csoaa.model data/output/cost/open_cost_3.dat -p data/output/cost/open_cost_3.predict
#
# vw --csoaa 16 data/output/cost/closed_cost_1.dat -f data/output/cost/csoaa.model
# vw -t -i data/output/cost/csoaa.model data/output/cost/test.dat -p data/output/cost/closed_cost_1.predict

# vw --csoaa 16 data/output/cost/closed_cost_2.dat -f data/output/cost/csoaa.model
# vw -t -i data/output/cost/csoaa.model data/output/cost/closed_cost_3.dat -p data/output/cost/closed_cost_2.predict
#
# vw --csoaa 16 data/output/cost/closed_cost_3.dat -f data/output/cost/csoaa.model
# vw -t -i data/output/cost/csoaa.model data/output/cost/closed_cost_3.dat -p data/output/cost/closed_cost_3.predict
