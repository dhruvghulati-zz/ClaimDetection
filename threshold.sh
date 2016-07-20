#!/bin/bash

declare -a threshold=(0.0001 0.0005 0.001 0.0025 0.0050 0.01 0.02 0.05 0.075 0.1 0.15 0.30 0.40 0.50 0.75 1.0);

for var in "${threshold[@]}"; do
      python src/main/propertyPredictor.py data/freebaseTriples.json data/sentenceMatrixFiltered.json data/output/predictedProperties.json $var data/featuresKept.json
      python src/main/testFeatures.py data/featuresKept.json data/output/testLabels.json data/output/hyperTestLabels.json $var data/freebaseTriples.json data/output/devLabels.json
      python src/main/logisticBagOfWords.py data/output/predictedProperties.json data/output/devLabels.json data/featuresKept.json data/output/ data/output/test/summaryEval.csv
      # echo "python src/main/propertyPredictor.py data/freebaseTriples.json data/sentenceMatrixFiltered.json data/output/predictedProperties.json $var"
      # echo "python src/main/testFeatures.py data/featuresKept.json data/output/testLabels.json data/output/hyperTestLabels.json $var data/freebaseTriples.json data/output/devLabels.json"
      # echo "python src/main/logisticBagOfWords.py data/output/predictedProperties.json data/output/devLabels.json data/featuresKept.json data/output/evaluation.txt data/output/ data/output/summaryEval.csv"
done
