# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 09:31:43 2024

@author: 20052
"""

## Setup chunk
import time, re
import pandas as pd
import numpy as np

from nltk import sent_tokenize, word_tokenize
from lexical_diversity import lex_div as ld
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 




wl_ops_url = "https://raw.githubusercontent.com/sudhir-voleti/OpsMgtSc/main/wl_ops_042023.csv"
wl_ops = pd.read.csv(wl_ops_url)

run_sub_uniq_300 = pd.read.csv("https://raw.githubusercontent.com/sudhir-voleti/OpsMgtSc/main/ECTs_300.csv")

"""
First obtain doc_wl level variables. toks and sents, counts and extr_sents
"""

# first edit wordlist to append '\\w*' both sides
wl_ops1 = []
for word0 in wl_ops:
    word1 = r'\\w*' + word0.lower() + r'\\w*'; word1
    word2 = re.sub(r'\\\\', r'\\', word1); word2
    wl_ops1.append(word2) # 'extend' appends chars 

# edited 2 also capture extracted toks
def extr_wordlist_tokens(doc0, wordlist0, num_wl_toks, extr_toks):
	num_wl_toks0 = 0
	extr_toks0 = []
	for word0 in wordlist0:
		#word1 = r'\\w*' + word0.lower() + r'\\w*'; word1
		#word2 = re.sub(r'\\\\', r'\\', word1); word2        
		a0 = re.findall(word0, doc0); a0
		num_wl_toks0 = num_wl_toks0 + len(a0)
		if len(a0) > 0:
			extr_toks0.extend(a0)

	num_wl_toks.append(num_wl_toks0)
	extr_toks.append(extr_toks0)
	return(num_wl_toks, extr_toks)

# test drive above
doc_wl_toks_pr = []; doc_wl_toks_qna=[]; extr_toks_pr=[]; extr_toks_qna=[]

for i0 in range(run_sub_uniq_300.shape[0]): # 9s for 1k docs
	doc0 = run_sub_uniq.prepRmks.iloc[i0]
	if type(doc0) == float:
		doc0 = str(doc0)
	doc_wl_toks_pr, extr_toks_pr = extr_wordlist_tokens(doc0, wl_mktg, doc_wl_toks_pr, extr_toks_pr)
	if i0%2000==0:
		print(i0) # 12682 s

for i0 in range(run_sub_uniq_300.shape[0]):
	doc0 = run_sub_uniq.QnA.iloc[i0]
	if type(doc0)==float:
		doc0 = str(doc0)
	doc_wl_toks_qna, extr_toks_qna = extr_wordlist_tokens(doc0, wl_mktg, doc_wl_toks_qna, extr_toks_qna)
	if i0%2000==0:
		print(i0)  # 16530 s


# functionize and run wordlist based sentence extractor
def extr_wordlist_sents(doc0, wordlist0, wl_hyp_sents_num, extr_sents):
    
    if type(doc0)==float:
        doc0 = str(doc0)
        
    sent_list0 = sent_tokenize(doc0)
    keep_words0 = [] # for any doc, use only relev words
    for word0 in wordlist0:
        word1 = re.sub(r'\\\\', r'\\', word0.lower())
        if(len(re.findall(word1, doc0))>0):
            keep_words0.append(word1)

    sent_df0 = pd.Series(sent_list0); sent_df0.head()
    sents_stor0 = []

    for word0 in keep_words0:
        a0 = sent_df0.apply(lambda x: len(re.findall(word0, x)))
        a1 = [sent_df0.iloc[x] for x in range(len(a0)) if a0[x]>0]
        sents_stor0.extend(a1)
	
    sents_stor1 = list(set(sents_stor0)); sents_stor1
    wl_hyp_sents_num.append(len(sents_stor1))
    if len(sents_stor1)==0: # potential anomaly handling
        sents_stor1 = ['empty']
    sents_stor2 = " ".join(sents_stor1); sents_stor2
    extr_sents.append(sents_stor2)

    return(wl_hyp_sents_num, extr_sents)

## run below in separate instance to parallelize
sents_num_pr = []; extr_sents_pr = []; sents_num_qna = []; extr_sents_qna = []

for i0 in range(run_sub_uniq.shape[0]):
	doc0 = run_sub_uniq.QnA.iloc[i0]
	if type(doc0)==float:
		doc0 = str(doc0)
	sents_num_qna, extr_sents_qna = extr_wordlist_sents(doc0, wl_ops, sents_num_qna, extr_sents_qna)
	if i0%1000==0:
		print(i0)  # 4517 s

doc_wl_vars_df100 = pd.read_csv("https://raw.githubusercontent.com/sudhir-voleti/OpsMgtSc/main/doc_wl_vars_df100.csv").drop(['Unnamed: 0'], axis=1)
doc_wl_vars_df100.columns

"""
Build aux_metrics too - simplified.
"""

## Find lexical features for each doc
analyzer = SentimentIntensityAnalyzer()

def build_aux_metrics(filename_series, doc_series):
	lex_vol = []; ttr = []; mtld = []; #vocd = []  # lexical div measures
	compound_mean = []; compound_std = []    
	filename = []  # sentiment measures

	for i0 in range(len(doc_series)):

		filename0 = filename_series.iloc[i0]; filename0
		doc0 = doc_series.iloc[i0]; doc0
		doc0_list = sent_tokenize(doc0); doc0_list
		doc0_string = " ".join(doc0_list); doc0_string
		n1 = len(doc0_list); n1

		if n1 > 1:
			vs_list = []	
			for i1 in range(n1):
				sent0 = doc0_list[i1]
				vs0 = analyzer.polarity_scores(sent0); vs0
				vs_list.append(vs0)
	
			doc0_df = pd.DataFrame(vs_list); doc0_df	
			mean_list0 = [x for x in doc0_df.mean()]; mean_list0
			std_list0 = [x for x in doc0_df.std()]; std_list0

		else:
			mean_list0 = [float(0) for x in range(4)]; mean_list0
			std_list0 = [float(0) for x in range(4)]; std_list0

		compound_mean.append(mean_list0[3]); compound_std.append(std_list0[3])                        
		filename.append(filename0)

		flt = ld.flemmatize(doc0_string); flt
		lex_vol0 = len(flt)  # lexical volume measure
		ttr0 = ld.ttr(flt)  # basic Text-Type Ratio or TTR
		mtld0 = ld.mtld(flt) # Measure of Textual Lexical Diversity (MTLD) for lexical variability
		# vocd0 = ld.hdd(flt) # vocd or Hypergeometric distribution D (HDD), as per McCarthy and Jarvis (2007, 2010)

		lex_vol.append(lex_vol0)
		ttr.append(ttr0)
		mtld.append(mtld0)
		# vocd.append(vocd0)

		if i0%5000 == 0:
			print(i0)

	# save as df
	df1 = pd.DataFrame({'prim_key':filename, 'senti_compound': compound_mean,'senti_compound_std': compound_std,
                      'lex_vol':lex_vol, 'ttr':ttr, 'mtld':mtld})
	return(df1)

## test-drive
%time doc_wl_aux_pr = build_aux_metrics(doc_wl_vars_df100.prim_key, doc_wl_vars_df100.doc_wl_extr_sents_pr) # 51m
%time doc_wl_aux_qna = build_aux_metrics(doc_wl_vars_df100.prim_key, doc_wl_vars_df100.doc_wl_extr_sents_qna) # 56m

## rename colms via df.rename(columns={"old": "new", "old1": "new1"})
doc_wl_aux_pr = doc_wl_aux_pr.rename(columns={'senti_compound': 'doc_wl_senti_mean_pr', 'senti_compound_std':'doc_wl_senti_std_pr',
                              'lex_vol':'doc_wl_lex_vol_pr', 'ttr':'doc_wl_ttr_pr', 'mtld':'doc_wl_mtld_pr'})

doc_wl_aux_qna = doc_wl_aux_qna.rename(columns={'senti_compound': 'doc_wl_senti_mean_qna', 'senti_compound_std':'doc_wl_senti_std_qna',
                              'lex_vol':'doc_wl_lex_vol_qna', 'ttr':'doc_wl_ttr_qna', 'mtld':'doc_wl_mtld_qna'})

doc_wl_aux_df = doc_wl_aux_pr.merge(doc_wl_aux_qna, how="inner", on="prim_key"); doc_wl_aux_df.columns


"""
===============================================================================
Extract lexical features, sentiment, readability metrics
===============================================================================
"""

analyzer = SentimentIntensityAnalyzer()
def build_senti_metrics(filename_series, doc_series):
	filename = []; vs_list = []  # sentiment measures
	for i0 in range(len(doc_series)):
		filename0 = filename_series.iloc[i0]; filename0
		doc0 = doc_series.iloc[i0]; doc0
		filename.append(filename0)
		vs0 = analyzer.polarity_scores(doc0); vs0
		vs_list.append(vs0)
        
		if i0%1000 == 0:
			print(i0)
        
	doc0_df = pd.DataFrame(vs_list)
	doc0_df.insert(0, 'prim_key', filename)
	return doc0_df

## test-drive above
%time testdf_pr = build_senti_metrics(doc_wl_vars_df100.prim_key, doc_wl_vars_df100.doc_wl_extr_sents_pr) # 1h 34 min
%time testdf_qna = build_senti_metrics(doc_wl_vars_df100.prim_key, doc_wl_vars_df100.doc_wl_extr_sents_qna) # 1h 34 min

## rename colms via df.rename(columns={"old": "new", "old1": "new1"})
testdf_pr = testdf_pr.rename(columns={'neg':'wl_neg_pr', 'pos':'wl_pos_pr', 'compound':'wl_senti_pr'})
testdf_pr = testdf_pr.drop(columns = ['neu']); testdf_pr.columns

testdf_qna = testdf_qna.rename(columns={'neg':'wl_neg_pr', 'pos':'wl_pos_pr', 'compound':'wl_senti_pr'})
testdf_qna = testdf_qna.drop(columns = ['neu']); testdf_qna.columns

senti_df = testdf_pr.merge(testdf_qna,  how='inner', on='prim_key')

"""
Do readby metrics too for these sents
"""

doc_wl_vars_df100.columns

# --- find readability indices for df_sents ---
import textstat
def calc_fogindex(sents_series0):
	fogIndex=[]; # flesch_kincaid=[]; flesch_readby=[];
	for i0 in range(len(sents_series0)):
		sent0 = sents_series0[i0]
		#flesch_readby.append(textstat.flesch_reading_ease(sent0))
		#flesch_kincaid.append(textstat.flesch_kincaid_grade(sent0))
		fogIndex.append(textstat.gunning_fog(sent0))
		if i0%1000==0:
			print(i0)

	df_readby = pd.DataFrame({'fogIndex':fogIndex})
	return(df_readby)

doc_wl_pr_fog_df = calc_fogindex(doc_wl_vars_df100.doc_wl_extr_sents_pr) 
doc_wl_qna_fog_df = calc_fogindex(doc_wl_vars_df100.doc_wl_extr_sents_qna)

doc_wl_pr_fog_df.insert(0, 'prim_key', doc_wl_vars_df100.prim_key)
doc_wl_qna_fog_df.insert(0, 'prim_key', doc_wl_vars_df100.prim_key)

doc_wl_pr_fog_df = doc_wl_pr_fog_df.rename(columns={'fogIndex': 'fogIndex_pr'})
doc_wl_qna_fog_df = doc_wl_qna_fog_df.rename(columns={'fogIndex': 'fogIndex_qna'})

fog_df = doc_wl_pr_fog_df.merge(doc_wl_qna_fog_df,  how='inner', on='prim_key'); fog_df.columns


## read in the DF with financial ratios included
df_ols_sub1k = pd.read_csv("https://raw.githubusercontent.com/sudhir-voleti/OpsMgtSc/main/df_ols_sub.csv").drop(['Unnamed: 0'], axis=1)
df_ols_sub1k.columns

## Merge all the disparate metrics together into one master-data-file
intermed1 = doc_wl_vars_df100.merge(doc_wl_aux_df, how="inner", on="prim_key"); intermed1.columns
intermed2 = intermed1.merge(senti_df, how="inner", on="prim_key"); intermed2.columns
intermed3 = intermed2.merge(fog_df, how="inner", on="prim_key"); intermed3.columns
analysis_df = df_ols_sub1k.merge(intermed3, how="inner", on="prim_key"); analysis_df.columns

# analysis_df above can be used for the event study. That part of the analysis is done in R.


