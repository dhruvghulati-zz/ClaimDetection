#!/bin/bash
#
declare -a threshold=(0.0001 0.0005 0.001 0.0025 0.0050 0.01 0.02 0.05 0.075 0.1 0.15 0.30 0.40 0.50 0.75 1.0);

# 0.0001 0.0005 0.001 0.0025 0.0050 0.01 0.02 0.05 0.075 0.1

declare -a probThreshold=(0.0001 0.001 0.0050 0.01 0.02 0.04 0.05 0.06 0.075 0.1 0.15 0.30 0.40 0.50 0.75 1.0);

for var in "${threshold[@]}";
do
      python src/main/propertyPredictor.py data/freebaseTriples.json data/sentenceMatrixFiltered.json data/output/predictedProperties.json $var data/featuresKept.json
      python src/main/testFeatures.py data/featuresKept.json data/output/testLabels.json data/output/hyperTestLabels.json $var data/freebaseTriples.json data/output/devLabels.json
      python src/main/logisticBagOfWords2.py data/output/predictedProperties.json data/output/devLabels.json data/featuresKept.json data/output/test_zero/ data/output/test_zero/summaryEvaluationTest.csv
      for prob in "${probThreshold[@]}";
      do
          python src/main/probPredictor.py data/output/test_zero/ data/output/test/summaryEvaluationTest.csv $prob
      done
done

# python src/main/propertyPredictor.py data/freebaseTriples.json data/sentenceMatrixFiltered.json data/output/predictedProperties.json 0.05 data/featuresKept.json
# python src/main/testFeatures.py data/featuresKept.json data/output/testLabels.json data/output/hyperTestLabels.json 0.05 data/freebaseTriples.json data/output/devLabels.json
# python src/main/naiveBaselines.py data/output/predictedProperties.json data/output/hyperTestLabels.json data/featuresKept.json data/output/final/ data/output/final/summaryEvaluation.csv 0.05
# python src/main/probPredictor.py data/output/final/ data/output/final/summaryEvaluation.csv 0.05
