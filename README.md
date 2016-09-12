# Claim Detection Pipeline

The following repository contains the source code for a thesis completed for the MSc in Computer Science at UCL, supervised by the UCL Machine Reading group under Sebastian Riedel, Andreas Vlachos, and John Shawe-Taylor. Additional co-supervisors include George Spithourakis and Isabelle Augenstein.

Several large data files are needed to run the pipeline effectively, feel free to raise an issue for me to email them to you directly.

The annotated claims dataset is `claim_labels.csv` in the data folder.

##Data

_freebaseProperties.json_ is a dictionary of per statistical property values for each region from Freebase e.g.

    {"Canada": {"/location/statistical_region/size_of_armed_forces": 65700.0, "/location/statistical_region/gni_per_capita_in_ppp_dollars": 42530.0, "/location/statistical_region/gdp_nominal": 1736050505050.0, "/location/statistical_region/foreign_direct_investment_net_inflows": 8683048195.0, "/location/statistical_region/life_expectancy": 80.929, "/location/statistical_region/internet_users_percent_population": 86.765864, "/location/statistical_region/cpi_inflation_rate": 1.52,..
    
_sentenceRegionValue.json_ is a an array of dictionaries of parsed sentences with additional metadata e.g.

    [{
        "sentence": "The estimated population growth rate of Burundi was 3.104 % in 2012 which is more than double the world average of 1.1 % .", 
        "slotSentence": "The estimated population growth rate of LOCATION_SLOT was NUMBER_SLOT % in 2012 which is more than double the world average of NUMBER_SLOT % .", 
        "location-value-pair": {
            "Burundi": 1.1
        }, 
        "parsedSentence": "The estimated population growth rate of LOCATION_SLOT was 3.104 % in 2012 which is more than double the world average of NUMBER_SLOT % .", 
    }, ...]


##Data Extraction


2. **main/sentenceSlots.py**: This adapts the code from `buildMatrix.py`. Takes all the training JSONs from the previous step to obtain the full sentences and the regions/values in those sentences with location and number slots filled in, outputting `sentenceRegionValue.json`. Also generates sentences for labeling that have low word density and concentration. Create multiple sentences for every combination of region/value extracted, as long as at least one region and value in the sentence.
3. **main/sentenceMatrixFiltering.py**: This adapts `matrixFiltering.py` simply to account for alias regions extracted in the previous step, but does not account for sentences that contain the same value for every location, doesn't remove any values above a threshold standard deviation etc. Outputs `sentenceMatrixFiltered.json`. Also ensures no duplicate sentences and ensures no test sentences are in pool for training.
4. **main/propertyPredictor.py**: Takes the filtered sentence matrix and for each location, sentence, value, predicts the closest statistical property (e.g. population, gni per capita) in Freebase to that value based on mean absolute percentage error, performing distant supervision. Any prediction of a statistical value above the provided threshold is considered a 0 or 'no_prediction'. This generates `predictedProperties.json`.
5. **main/testFeatures.py**: Takes every labelled sentence in the labelled Excel files, and does some data munging to extract a MAPE, parsedSentence (with LOCATION_SLOT, NUMBER_SLOT) and statistical region accounting for empty values among other metadata. Does the same for any cleanly labeled annotated sentences. Also performs distant supervision on the test sentences themselves. This generates `testLabels.json` and `hyperTestLabels.json`.

###Distantly Supervised Classification Pipeline

1. **main/naiveBaselines.py**: Generates the distantly supervised classifier predictions without any APE thresholds.
2. **main/naiveProbPredictor.py**: Takes the output probabilistic predicts from the naive baselines and generates predictions with better recall
3. **main/logisticBagOfWords.py**: Extracts features from the HTML sentences in the training JSONs using a bag of words feature extractor as well as dependency bigrams. This includes the LOCATION_SLOT and NUMBER_SLOT features, all in lower case. Fits various multinomial logistic regression classifiers on the training sentences only for cases where there is an APE theshold. Obtains precision, recall, F1 and passes output to a probability threshold hyper-parameter tuner.
4. **main/probPredictor.py**:Takes the output probabilistic predicts from the APE threshold based distantly supervised models and generates predictions with better recall.

###Cost-Sensitive Classification Pipeline

1. **main/costSensitiveClassifier.py**: Imports the AROW algorithm with **main/arow_claim_csc.py** to generate cost sensitive predictions.
2. **main/costPredictions.py**: Takes the `label2score` predictions of the previous step to generate probabilistic thresholds and new predictions based on these.

###Textual Patterns Classification Pipeline

1. **main/buildMatrix.py**: Modifies the original code by Vlachos & Riedel 2015 to build a matrix of textual patterns excluding any test sentences evaluated by the other classifiers.
2. **main/factChecker.py**: Adapts previous code to detect claims in any random annotated test set which contains its surface patterns and dependencies.

##Tools

- **utils/testTrainSplit.py**: Takes HTML parsed JSONs obtained from Stanford CoreNLP parser and splits the JSONs into test and training (to ensure no JSONs that appeared in the labeled claims files that form the initial test set appear in the training set)
- **utils/thresholdCharting.py**: Produces the charts that evaluate the effect of APE, Cost and Sigmoid Hyper-parameters.
- **utils/charting.py**: Produces all other charts found in the thesis.
- **utils/sentenceCSVs.py**: Wrote out a CSV file of unannotated claims to label.
- **utils/precisionRecall.py**: Looks through all files produce by the textual patterns model across all properties and evaluates overall metrics.
- **utils/cleanLabels.py** Augments the labeled claims with other properties of sentences, for example surface patterns and dependency paths, for fair testing, converts negative labels, and other filtering to any annotated sentences.

### Bash Scripts

- **vowpal.sh** Runs the cost sensitive classifier pipeline with several thresholds, bias and sigmoid slopes for hyper-parameter tuning.
- **threshold.sh** Does the same for the distantly supervised model with several APE thresholds.
- **textualpatterns.sh** Outputs every prediction from the textual patterns model in one go instead of requiring to run one by one, evaluating recall in one go.
