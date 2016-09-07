#!/bin/bash

declare -a threshold=(0.0001 0.0005 0.001 0.0025 0.0050 0.01 0.02 0.05 0.075 0.1 0.15 0.30 0.40 0.50 0.75 1.0);

declare -a probThreshold=(0.0001 0.001 0.0050 0.01 0.02 0.04 0.05 0.06 0.075 0.1 0.15 0.30 0.40 0.50 0.75 1.0);

# This is for the pure naive baseline

python src/main/propertyPredictor.py data/freebaseTriples.json data/sentenceMatrixFilteredZero.json data/output/predictedPropertiesZero.json 0.05 data/featuresKept.json
python src/main/testFeatures.py data/featuresKept.json 0.05 data/freebaseTriples.json data/output/devLabels.json data/output/testLabels.json data/output/fullTestLabels.json
python src/main/naiveBaselines.py data/output/predictedPropertiesZero.json data/output/devLabels.json data/featuresKept.json data/output/zero/test2/ data/output/zero/test2/summaryEvaluation.csv

for prob in "${probThreshold[@]}";
do
    python src/main/naiveProbPredictor.py data/output/zero/test2/ $prob data/output/zero/test2/models.json data/output/zero/test2/summaryEvaluation.csv
done



for var in "${threshold[@]}";
do
      python src/main/propertyPredictor.py data/freebaseTriples.json data/sentenceMatrixFilteredZero.json data/output/predictedPropertiesZero.json $var data/featuresKept.json
      python src/main/testFeatures.py data/featuresKept.json $var data/freebaseTriples.json data/output/devLabels.json data/output/testLabels.json data/output/fullTestLabels.json
      python src/main/logisticBagOfWords.py data/output/predictedPropertiesZero.json data/output/devLabels.json data/featuresKept.json data/output/zero/test2/ data/output/zero/test2/summaryEvaluation.csv
      for prob in "${probThreshold[@]}";
      do
          python src/main/probPredictor.py data/output/zero/test2/ $prob data/output/zero/test2/models.json data/output/zero/test2/summaryEvaluation.csv
      done
done

# python src/main/propertyPredictor.py data/freebaseTriples.json data/sentenceMatrixFilteredZero.json data/output/predictedPropertiesZero.json 0.05 data/featuresKept.json
# python src/main/testFeatures.py data/featuresKept.json 0.05 data/freebaseTriples.json data/output/devLabels.json data/output/hyperTestLabels.json data/output/testLabels.json data/output/fullTestLabels.json
# python src/main/logisticBagOfWords.py data/output/predictedPropertiesZero.json data/output/fullTestLabels.json data/featuresKept.json data/output/zero/test/ data/output/zero/test/summaryEvaluation.csv
# for prob in "${probThreshold[@]}";
# do
#     python src/main/probPredictor.py data/output/zero/test/ $prob data/output/zero/test/models.json data/output/zero/test/summaryEvaluation.csv
# done
