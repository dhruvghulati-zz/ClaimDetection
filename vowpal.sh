#!/bin/bash

declare -a threshold=(0.0001 0.0005 0.001 0.0025 0.0050 0.01 0.02 0.05 0.075 0.1 0.15 0.30 0.40 0.50 0.75 1.0);

# 0.0001 0.0005 0.001 0.0025 0.0050 0.01 0.02 0.05 0.075 0.1

declare -a probThreshold=(0.0001 0.001 0.0050 0.01 0.02 0.04 0.05 0.06 0.075 0.1 0.15 0.30 0.40 0.50 0.75 1.0);

# 0.0001 0.001 0.0050 0.01 0.02 0.04 0.05 0.06 0.075 0.1 0.15 0.30 0.40 0.50 0.75 1.0

declare -a costThreshold=(0.0001 0.001 0.0050 0.01 0.02 0.04 0.05 0.06 0.075 0.1 0.15 0.30 0.40 0.50 0.75 1.0);

FILES=data/output/zero/arow_test/

for f in ${FILES}*.dat
do
    if [[ $f == *"bigrams"* ]]
    then
      echo "It's there!";
    fi
done


for var in "${threshold[@]}";
do
      python src/main/propertyPredictor.py data/freebaseTriples.json data/sentenceMatrixFilteredZero.json data/output/predictedPropertiesZero.json $var data/featuresKept.json
      python src/main/testFeatures.py data/featuresKept.json data/output/testLabels.json data/output/hyperTestLabels.json $var data/freebaseTriples.json data/output/devLabels.json
      # Create different files with different thresholds and probability thresholds. We have models that do not care about this cost threshold though.
      for cost in "${costThreshold[@]}";
      do
          python src/main/costSensitiveClassifier.py data/freebaseTriples.json data/output/predictedPropertiesZero.json data/output/devLabels.json data/featuresKept.json data/output/zero/cost_test/ data/output/zero/arow_test/ data/output/zero/cost_test/costMatrices.json $cost
      done
      # for f in ${FILES}*.pdf
      # do
      #     if [[ $string == *"My long"* ]]
      #     then
      #       echo "It's there!";
      #     fi
      # done



      # Now do the predictions with all these files outputting them somewhere, including some precision F1 scores
      # Now make different probability predictions with the same input files and output these also somewhere in a tuning version for precision recall

done

# python src/main/costPredictions.py data/output/cost/models.txt data/output/cost/open_label_mapping.txt data/output/cost/closed_label_mapping.txt data/output/devLabels.json data/output/cost/ data/output/cost/summaryEvaluationCost.csv
