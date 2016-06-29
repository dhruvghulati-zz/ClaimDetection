'''
This file goes through the labeled claim files and extracts the file names of the JSON's which contain labelled claims.

It then finds those files in the HTMLParsedPages folder and places them in a separatefolder called test_jsons
'''

import os
import xlrd
import shutil

for subdir, dirs, files in os.walk('../../data/labeled_claims'):
    http_values = []
    https_values = []
    values = []
    for file in files:
        # print os.path.join(subdir, file)
        filepath = subdir + os.sep + file

        if filepath.endswith(".xlsx"):
            wb = xlrd.open_workbook(filepath)

            for s in wb.sheets():
                for row in range(0, s.nrows):
                    for col in range(s.ncols):
                        value  = (s.cell(row,col).value)
                        # print value
                        # Only care about the rows with ascii unicode, convert to string
                        try : value = str((value))
                        except : pass
                        if value.find(".json")>-1:
                            # param, value = value.split("/",1)
                            # if (value[:5] == 'http:'):
                            #     value = value[:4] + '/' + value[5:]
                            #     http_values.append(value)
                            #     # print value
                            # if (value[:5] == 'https'):
                            #     value = value[:5] + '/' + value[6:]
                            #     https_values.append(value)
                                # print value
                            # print value
                            # values = http_values + https_values
                            values.append(value)
    print 'The number of matching files is ' + str(len(values))

jsondir = '../../data/htmlPages2textPARSEDALL/'
if os.path.exists(jsondir):
    path, dirs, jsonfiles = os.walk(jsondir).next()
print 'The total number of JSONs is ' + str(len(jsonfiles))

if not os.path.exists('../../data/test_jsons/htmlPages2textPARSEDALL/'):
    os.makedirs('../../data/test_jsons/htmlPages2textPARSEDALL/')

if not os.path.exists('../../data/train_jsons/'):
    os.makedirs('../../data/train_jsons/')

count = 0
for file in values:
    if os.path.exists(os.path.join("../../data/",str(file))):
        count=count+1
print str(count) + ' files exist in the directory.'

for file in values:
    if os.path.exists(os.path.join("../../data/",str(file))):
        os.rename(os.path.join("../../data/",str(file)), os.path.join("../../data/test_jsons/",str(file)))
        # shutil.move("../../data/" + str(file), "../../data/test_jsons/" + str(file))

# Counts files in test json directory
testdir = '../../data/test_jsons/htmlPages2textPARSEDALL/'
path, dirs, files = os.walk(testdir).next()
print 'The total number of test JSONs is ' + str(len(files))

# Counts files remaining
jsondir = '../../data/htmlPages2textPARSEDALL/'
path, dirs, files = os.walk(jsondir).next()
print 'Files remaining is ' + str(len(files))

# Moves the files up one tree in the test folder
fileList = os.listdir('../../data/test_jsons/htmlPages2textPARSEDALL/')
fileList = ['../../data/test_jsons/htmlPages2textPARSEDALL/'+filename for filename in fileList]

for f in fileList:
    shutil.move(f, '../../data/test_jsons/')

# Delete the blank folder
shutil.rmtree('../../data/test_jsons/htmlPages2textPARSEDALL/')

# Move everything in HTML pages left to train folder
os.rename("../../data/htmlPages2textPARSEDALL/", "../../data/train_jsons/")

# Counts files in training
traindir = '../../data/train_jsons/'
path, dirs, files = os.walk(traindir).next()
print 'Files in training set is ' + str(len(files))