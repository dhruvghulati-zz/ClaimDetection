#!/bin/bash

properties=(gni_per_capita_in_ppp_dollars gdp_nominal internet_users_percent_population cpi_inflation_rate health_expenditure_as_percent_of_gdp gdp_growth_rate fertility_rate consumer_price_index prevalence_of_undernourisment gni_in_ppp_dollars population_growth_rate diesel_price_liter life_expectancy population gdp_nominal_per_capita renewable_freshwater_per_capita)

c=(8 0.03125 2 2 2 1 0.5 1 32 16 0.0078125 2 1 0.03125 16 8)

for ((i = 0; i < ${#properties[@]}; i++)); do
  python src/main/factChecker.py data/freebaseTriples.json data/mainMatrixFiltered.json ${properties[$i]} ${c[$i]} data/output/cleanFullLabelsDistant.json data/locationNames data/aliases.json data/output/zero/andreas2np/${properties[$i]}.tsv

done

python src/utils/precisionRecall.py data/output/zero/andreas2np
