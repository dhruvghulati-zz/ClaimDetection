#!/bin/bash

declare -a APEthreshold=(0.0050);

 # 0.15

# Original params
# 0.0001 0.001 0.0050 0.01 0.02 0.04 0.05 0.06 0.075 0.1 0.15 0.30 0.40 0.50 0.75 1.0
# Shorter params
# 0.0001 0.001 0.01 0.05 0.1 0.15 0.30 0.50 0.75 1.0

declare -a costThreshold=(0.0001 0.001 0.01 0.05 0.1 0.15 0.30 0.50 0.75 1.0);

# declare -a biasThreshold=(0.00);

declare -a slopeThreshold=(0.0001 0.001 0.01 0.05 0.1 0.15 0.30 0.50 0.75 1.0);

FILES=data/output/zero/arow_test/

for ape in "${APEthreshold[@]}";
do
      python src/main/propertyPredictor.py data/freebaseTriples.json data/sentenceMatrixFilteredZero.json data/output/predictedPropertiesZero.json $ape data/featuresKept.json
      python src/main/testFeatures.py data/featuresKept.json $ape data/freebaseTriples.json data/output/devLabels.json data/output/testLabels.json data/output/fullTestLabels.json
      # Create different files with different thresholds and probability thresholds. We have models that do not care about this cost threshold though.
      for cost in "${costThreshold[@]}";
      do
        for slope in "${slopeThreshold[@]}";
        do
            python src/main/costSensitiveClassifier.py data/output/predictedPropertiesZero.json data/output/devLabels.json data/featuresKept.json data/output/zero/arow_test/ data/output/zero/arow_test/predict2/ data/output/zero/arow_test/probPredict2/ $cost 0.00 $slope
        done
      done
      # Now do the predictions with all these files outputting them somewhere
      # for f in ${FILES}*.dat
      # do
      #     file=$(basename "$f")
      #     file=${file%.*}
      #     if [ -s $f ];
      #     then
      #       if [[ $file == *"wordgrams"* ]]
      #       then
      #         python src/main/arow_claim_csc.py $f data/output/zero/arow_test/wordgrams_test.test data/output/zero/arow_test/predict/$file.predict data/output/zero/arow_test/probPredict/$file.probpredict
      #       fi
      #     fi
      # done
done

# Now make different probability predictions with the same input files and output these also somewhere in a tuning version for precision recall
python src/main/costPredictions.py data/output/zero/arow_test/open_label_mapping.txt data/output/zero/arow_test/closed_label_mapping.txt data/output/zero/arow_test/open_label_mapping_threshold.txt data/output/zero/arow_test/closed_label_mapping_threshold.txt data/output/devLabels.json data/output/zero/arow_test/predict2/ data/output/zero/arow_test/probPredict2/ data/output/zero/arow_test/results2/
