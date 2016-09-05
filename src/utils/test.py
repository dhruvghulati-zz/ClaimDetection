#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import os
import sys
from sklearn.feature_extraction.text import CountVectorizer


print sys.getdefaultencoding()


train_wordbigram_list = ['location_slot great recessionalgeria gdp per capita lrb ppp rrb positive growth great recessionalgeria since end great recessionalgeria gdp per capita lrb ppp rrb positive growth since end great recessionalgeria algerian civil waralgeria gdp per capita lrb ppp rrb positive growth number_slot algerian civil waralgeria since end algerian civil waralgeria gdp per capita lrb ppp rrb negative growth lrb decline rrb since end algerian civil war LOCATION_SLOT~-nsubj+had~-dep had~-dep+had~dobj had~dobj+growth~prep_of growth~prep_of+PERCENT~num PERCENT~num+NUMBER_SLOT location_slot fertility rate cents average number births woman lifetime cents stands number_slot among lowest world LOCATION_SLOT~-poss+rate~-dep rate~-dep+MONEY~-dep MONEY~-dep+in~-csubj in~-csubj+â~dobj â~dobj+MONEY~rcmod MONEY~rcmod+stands~prep_at stands~prep_at+NUMBER_SLOT']

vectorizer = CountVectorizer(analyzer="word",token_pattern="[\S]+",stop_words=None,tokenizer=None,preprocessor=None,max_features=5000)

vectorizer.fit(train_wordbigram_list)

print vectorizer.vocabulary_
