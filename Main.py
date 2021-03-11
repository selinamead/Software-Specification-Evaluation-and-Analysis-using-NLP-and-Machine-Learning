from Utils import *
from NLP_Features import * 

import warnings

import sys
import re
import os
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.base import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from nltk.classify.naivebayes import NaiveBayesClassifier
from nltk.classify.svm import SvmClassifier
from nltk.classify.decisiontree import DecisionTreeClassifier
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier,\
	GradientBoostingRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.base import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import IsolationForest

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV 

# import tensorflow as tf
import tensorflow as tf

from tensorflow.python.keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline 
from tensorflow.keras.layers import Embedding, Reshape, TimeDistributed
from tensorflow.python.keras.layers import Input, InputLayer, Dense, Dropout, Flatten, Conv1D, GlobalMaxPooling1D, Embedding, Reshape, TimeDistributed, Activation, RepeatVector, SpatialDropout1D
from tensorflow.python.keras.layers import LSTM, Bidirectional, GRU
from tensorflow.python.keras.optimizers import Adam, RMSprop 

from imblearn.over_sampling import SMOTE
from collections import Counter
from matplotlib import pyplot


CV = 10
# classifier = 'RFC'
# classifier = 'SVM'
# classifier = 'LogReg'
classifier = 'MNB'


class Specification_Evaluation():
	
	def __init__(self, file_path):
		warnings.filterwarnings('ignore')
		self.data_location = file_path
		print('Specification_Evaluation')

	''' Convert requirement evaluation spreadsheet into dataframe'''
	def get_data(self, score_target):
		print('Converting requirement evaluation spreadsheet into dataframe')

		self.df = pd.read_excel(self.data_location, encoding = "utf-8")
		print(self.df.shape)
		# print(self.df.head())

		# Remove rows where type is not evaluated
		self.df_type = pd.DataFrame(self.df[self.df["score"].notnull()].values, columns=self.df.columns)
		self.df_type["score"] = self.df_type["score"].astype('category')
		# print(self.df_type.head())
		print(self.df_type.shape)
		
		self.df_type["score_cat"] = self.df_type["score"].cat.codes
		# print(self.df_type.head())
		# print(self.df_type.shape)
			  
		self.data = pd.DataFrame(self.df_type[['requirement',score_target]].values,columns=['requirement',score_target])
		# print(self.data.head())	
		# Convert target to string
		self.data['target'] = self.data[score_target].astype(str)
		# print(self.data.head())

		# Remove string na's
		self.data = self.data[~self.data.target.str.contains("na")]

		print('\nNo of Requirements in dataset:', len(self.data))

		print('\nProcessing Requirements and Extracting Features...\n')
		
		return self.data

	''' Create feature matrix using NLP features created in NLP_Features.py '''
	def get_features(self, score_target, metric_create=True, hotenc=False, export_dfm=True):
		print('\nCreating Feature Matrix...')

		if metric_create:       
			parser = NLP_Features()
			pos_tag_req = []
			# get the features for all the requirements selected with the features() method                                                                
			self.features, pos_tag_req = parser.extract_features(self.data['requirement'], score_target, export = True, corpal = False)
			# Shift to Metrics.py to analyze requirements
			self.dfm = self.features
			# print(self.dfm.head())

			
		else:          
			self.dfm = pd.read_excel(u'/Users/selina/Code/Python/Thesis_Code/Generated_Files/'+ str(score_target)+'/Features_Export.xlsx',encoding = "utf-8")

		# The following cols will be dropped as not useful
		cols_to_drop = ['req',
					   'req_nlp',                                                                                              
					   'tags',
					   'sentences_by_nlp',
					   'sentence_nb_by_nlp',
					   'sentences_by_nltk',
					   'sentence_nb_by_nltk',
					   'sentences',
					   'sentences_tagged',
					   'weakwords_nb2',
					   'weakwords_nb2_lemma',
					   'difference',
					   'passive_per_sentence',
					   'passive_percent',
					   'Aux_Start_per_sentence',
					   'Sub_Conj_pro_sentece',
					   'Comp_conj_pro_sentence',
					   'Nb_of_verbs_pro_sentence',
					   'Nb_of_auxiliary_pro_sentence',
					   'werden_pro_sentence',
					   'formal_percent',
					   'formal_per_sentence',
					   'entities',
					   ]
		
		self.dfm.drop(cols_to_drop,axis=1,inplace=True)
		print(self.dfm)
		                 
		# hotenc 
		if hotenc:
			self.dfm["max_min_presence"] = self.dfm["max_min_presence"].astype('category')
			self.dfm["max_min_presence"] = self.dfm["max_min_presence"].cat.codes
			
			self.dfm["measurement_values"] = self.dfm["measurement_values"].astype('category')
			self.dfm["measurement_values"] = self.dfm["measurement_values"].cat.codes
			
			self.dfm["passive_global"] = self.dfm["passive_global"].astype('category')
			self.dfm["passive_global"] = self.dfm["passive_global"].cat.codes
			
			self.dfm["Aux_Start"] = self.dfm["Aux_Start"].astype('category')
			self.dfm["Aux_Start"] = self.dfm["Aux_Start"].cat.codes
			
			self.dfm["formal_global"] = self.dfm["formal_global"].astype('category')
			self.dfm["formal_global"] = self.dfm["formal_global"].cat.codes


		# function to seperate numerical and categorical columns of a input dataframe
		def num_cat_separation(df):
			metrics_num = []
			metrics_cat = []
			for col in df.columns:
				if df[col].dtype == "object":
					metrics_cat.append(col)
				else:
					metrics_num.append(col)
			# print("Categorical columns : {}".format(metrics_cat))
			# print("Numerical columns : {}".format(metrics_num))
			return(metrics_cat,metrics_num)
		
		features_cat,features_num = num_cat_separation(self.dfm)
		# transform categorical columns into 1 and 0   
		self.dfm = pd.get_dummies(self.dfm,features_cat)
		
		# print(self.dfm.head())

		# Normalise Data (StandardScaler or MinMax?)
		# RFC doesn't need scaled data however others do
		print('Scaling Data...')
		normalise = 'MM'
		# normalise = 'SS'
		
		if normalise == 'MM':
			dfm_scaled = self.dfm.copy()
			scaler = MinMaxScaler()
			dfm_scale = scaler.fit_transform(dfm_scaled)
			dfm_scaled = pd.DataFrame(dfm_scale, index = dfm_scaled.index, columns = dfm_scaled.columns)
			self.dfm_scaled = pd.DataFrame(dfm_scaled)
			# self.dfm_scaled = scaler.fit_transform(self.dfm)
			print(self.dfm_scaled)

		if normalise == 'SS':
			dfm_scaled = self.dfm.copy()
			scaler = StandardScaler()
			dfm_scale = scaler.fit_transform(dfm_scaled)
			dfm_scaled = pd.DataFrame(dfm_scale, index = dfm_scaled.index, columns = dfm_scaled.columns)
			self.dfm_scaled = pd.DataFrame(dfm_scaled)
			# self.dfm_scaled = scaler.fit_transform(self.dfm)
			print(self.dfm_scaled)


		if export_dfm:
			datafile = "./Generated_Files/" + str(score_target) + "/DFM_Export/DFM_Export_" + str(score_target) + ".xlsx"
			self.dfm.to_excel(datafile, index=False)
			dfm_scale = self.dfm.copy()
			scaler = MinMaxScaler()
			dfm_scale2 = scaler.fit_transform(dfm_scale)
			dfm_scale = pd.DataFrame(dfm_scale2, index = dfm_scale.index, columns = dfm_scale.columns)
			dfm_scale = pd.DataFrame(dfm_scale)
			datafile_scale = "./Generated_Files/" + str(score_target) + "/DFM_Export/DFM_Export_Scale_" + str(score_target) + ".xlsx"
			dfm_scale.to_excel(datafile_scale, index=False)
			# print ("Create Excel export file: %s"%(datafile))
			# print ("Create Excel export file: %s"%(datafile_scale))

		return self.dfm, self.dfm_scaled

	''' Preprocessing data ready for use in model (stemming, tokenisation)'''
	def preprocessing(self, score_target):
		print('preprocessing uses dfm and converts to features')
		
		# Function for stopwords removal
		self.ger_stopws = set(stopwords.words('german'))
		german_sw  = set(get_stop_words('german'))

		# Function to take the stop words out of a tokenized list
		def stopword_removal(token_list):
			new_list = []
			for token in token_list:
				if token not in self.ger_stopws:
					new_list.append(token)
			return new_list

		# Function to carry out stemming
		stemmer = SnowballStemmer("german")
		def word_stemming(token_list):
			stemming_list = []
			for word in token_list:
				stemming_list.append(stemmer.stem(word))
			return stemming_list
		
		# Tokenize requirements
		tokenizer = nltk.RegexpTokenizer(pattern='\w+|\$[\d\.]+|\S+')
		text_tokenized = self.data['requirement'].apply(lambda x: tokenizer.tokenize(x))
		# print(text_tokenized)
		# input()
		                      
		# Remove stopwords
		text_stopwords = text_tokenized.apply(stopword_removal)

		# stemming requirements
		text_stemmed = text_stopwords.apply(word_stemming)                                           
		text_stemmed_string = text_stemmed.apply(lambda x: " ".join(x))

		# print('Problem with req_process')
		
		# print(text_stemmed)
		# input()
		# print(text_stemmed_string)
		# input()                   
		
		# Add a new column to dataframe which contains the pre-processed requirements 
		self.data['req_process'] = text_stemmed_string
		# print(self.data['req_process'])
		# input()
		
		export_token = True
				
		if export_token:
			datafile = "./Generated_Files/" + str(score_target) + "/DFM_Export/Toke_Stop_Stem_Export_" + str(score_target) + ".xlsx" 
			self.data.to_excel(datafile, index=False)
			# print ("\nCreate Excel export file: %s"%(datafile))

		# Combine preprocessed requirements (dfm) with features
		self.features = self.dfm.copy()
		self.features['req_process'] = self.data['req_process']
		self.features['target'] = self.data['target']
		# Combine scaled dfm with features
		self.features_scaled = self.dfm_scaled.copy()
		self.features_scaled['req_process'] = self.data['req_process']
		self.features_scaled['target'] = self.data['target']

		export_token_features = True
				
		if export_token_features:
			datafile = "./Generated_Files/" + str(score_target) + "/DFM_Export/DFM_ProcReqs_Export_" + str(score_target) + ".xlsx"  
			self.features.to_excel(datafile, index=False)
			# print ("Create Excel export file: %s"%(datafile)) 
		
		# Remove na's from target for scaled and non-scaled 
		self.features = self.features[~self.features.target.str.contains("na")]
		self.features_scaled = self.features_scaled[~self.features_scaled.target.str.contains("na")]
		
		print('\nPreprocessing Data Complete!!!')

		# return self.features, self.features_scaled

	def save_features_matrix(self, score_target):

		print('Saving features Matrix to file...')
		self.features.to_csv('features_matrix.csv')
		self.features.to_csv('features_scaled_matrix.csv')

	def load_features_matrix(self, score_target): ########### FIX STR INT #############

		print('Loading features matrix from file...')
		# Load feature matrix (and convert target to int)
		features_matrix = pd.read_csv("features_matrix.csv")
		# del features_matrix['req_process']
		self.features = features_matrix.copy()
		# Scaled data for Neural Networks
		features_scaled_matrix = pd.read_csv("features_scaled_matrix.csv")
		# del features_scaled_matrix['req_process']
		# features_scaled_matrix['target'].astype(int)
		self.features_scaled = features_scaled_matrix.copy()

		# check all values are ints
		# intlist = self.features.loc[~self.features['target'].str.isdigit(), 'target'].tolist()
		# print(intlist)

	''' Function to compare differing ML models to find most suitable model'''
	def classifier_comparison(self, score_target):
		print('\n======== Model Comparison ========\n')
		
		'''  
		-----------  different models possibilities   -----------------------------------------------------------------------------
		Benchmark the following models:
			RandomForestClassifier                       # A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and use averaging to improve the predictive accuracy and control over-fitting.
			MultinomialNB                                # The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification). The multinomial distribution normally requires integer feature counts. However, in practice, fractional counts such as tf-idf may also work.     
			Logistic Regression                          # Logistic Regression (aka logit, MaxEnt) classifier.In the multiclass case, the training algorithm uses the one-vs-rest (OvR) scheme if the �multi_class� option is set to �ovr�, and uses the cross- entropy loss if the �multi_class� option is set to �multinomial�.  
	   ----------------------------------------------------------------------------------------------------------------------------------     
	   '''

		# Assign training_data and labels
		features = self.features_scaled.copy()
		features['target'] = self.data['target']

		# Assigning X, y
		X = features.loc[:, features.columns != 'target']
		y = features.loc[:, features.columns == 'target'].values.ravel()
		# Convert to ints
		y = [int(i) for i in y]

		# Split train and test data
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

		# Models to compare
		models = [
				  RandomForestClassifier(n_estimators=200, max_depth=30, random_state=0),                                          
				  LogisticRegression(random_state=42, solver='lbfgs', multi_class='multinomial', max_iter=1000),
				  SVC(decision_function_shape="ovo", C=1, gamma=0.1, verbose=False, kernel="rbf")    
				 ]
		
		# list that will merge the results
		entries = []
		# For each model: get name and accuracy
		for model in models:
			model_name = model.__class__.__name__
			accuracies = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=CV)
			# for each fold get fold num and acc
			for fold_idx, accuracy in enumerate(accuracies):
				entries.append((model_name, fold_idx, accuracy))
		
		# Df to store results
		cv_df = pd.DataFrame(index=range(CV * len(models)))
		cv_df = pd.DataFrame(entries, columns=['classifier', 'fold_idx', 'accuracy']) 
		print(cv_df.groupby('classifier').accuracy.mean())

		sns.boxplot(x='classifier', y='accuracy', data=cv_df)
		sns.stripplot(x='classifier', y='accuracy', data=cv_df, size=5, jitter=True, edgecolor="gray", linewidth=1)
		plt.xticks(rotation=45 , ha="right")
		plt.savefig('./Generated_Files/' + str(score_target) + '/Benchmark_Export_' + str(score_target) + '.png', dpi=300, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures
		plt.show(block=False)
		plt.pause(2)
		plt.clf()
		plt.close() 
	
	''' Function to find the best params for RFC '''	
	def grid_search(self, score_target):
		print('======== Grid Search ========')

		# Function to find accuracy score of each classifier
		def evaluate(model, X_test, y_test):
		    predictions = model.predict(X_test)
		    errors = abs(predictions - y_test)
		    mape = 100 * np.mean(errors / y_test)
		    accuracy = 100 - mape
		    print('\n+++ Model Performance +++')
		    print('Model: ', model) # Check model
		    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
		    print('Accuracy = {:0.2f}%.'.format(accuracy))
		    return accuracy


		# Assigning training and test data #

		nlp_feature_matrix = pd.DataFrame(self.features_scaled)
		nlp_feature_matrix = nlp_feature_matrix.loc[:, nlp_feature_matrix.columns != 'req_process']
		# Convert to numpy
		y = np.array(nlp_feature_matrix['target'])
		# axis 1 refers to the columns
		X = nlp_feature_matrix.drop('target', axis=1)
		# Saving feature names
		feature_names = list(X.columns)
		# print(feature_names)
		# Convert to numpy array
		X = np.array(X)
		print(X.shape, y.shape)
		
		# Split train and test data
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

		# ### Choose Classifier ###
		# # classifier = "RFC"
		# # classifier = "SVM"
		# classifier = "LogReg"
		# # classifier = "NB"
		
		if classifier == "RFC":
			clf = RandomForestClassifier()
			print('\n+++ Parameters currently in use +++\n')
			print(clf.get_params())
			
			# Number of trees in random forest
			n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
			# Number of features to consider at every split
			max_features = ['auto', 'sqrt']
			# Maximum number of levels in tree
			max_depth = [int(x) for x in np.linspace(start=10, stop=100, num=10)]
			max_depth.append(None)
			# Minimum number of samples required to split a node
			min_samples_split = [2, 5, 10]
			# Minimum number of samples required at each leaf node
			min_samples_leaf = [1, 2, 4]
			# Method of selecting samples for training each tree
			bootstrap = [True, False]
			# Create the random grid
			random_grid = {'n_estimators': n_estimators,
			               'max_features': max_features,
			               'max_depth': max_depth,
			               'min_samples_split': min_samples_split,
			               'min_samples_leaf': min_samples_leaf,
			               'bootstrap': bootstrap}
			# print('\n+++ Random Grid +++\n', random_grid)
			'''
			{'bootstrap': [True, False],
			 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
			 'max_features': ['auto', 'sqrt'],
			 'min_samples_leaf': [1, 2, 4],
			 'min_samples_split': [2, 5, 10],
			 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
			'''

			# Use the random grid to search for best hyperparameters
		
			# Random search of parameters, using 3 fold cross validation
			rf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, n_iter=100, cv=CV, verbose=2, random_state=42, n_jobs=-1)
			# Fit the random search model
			rf_random.fit(X_train, y_train)

			# View the best parameters from fitting the Random search
			print('\n+++ Best Params +++\n', rf_random.best_params_)

			base_model = clf
			base_model.fit(X_train, y_train)
			base_accuracy = evaluate(base_model, X_test, y_test)
			
			best_random = rf_random.best_estimator_
			random_accuracy = evaluate(best_random, X_test, y_test)
			
			print('\nImprovement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

		if classifier == "SVM":
			clf = SVC()
			print('\n+++ Parameters for SVM +++\n')
			print(clf.get_params())

			# Dict of possible parameters
			params_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], # 10
						   'gamma': [0.0001, 0.001, 0.01, 0.1], # 0.1
          				   # 'kernel':['rbf', 'linear'] } # rbf
          				   'kernel':['rbf'] } # rbf

			grid_SVM = GridSearchCV(SVC(), params_grid, refit=True, verbose=3, cv=CV, n_jobs=-1) 
			# Fit the data 
			grid_SVM.fit(X_train, y_train)

			# View the best parameters from fitting the Random search
			print('\n+++ Best Params +++\n', grid_SVM.best_params_)

			base_model = clf
			base_model.fit(X_train, y_train)
			base_accuracy = evaluate(base_model, X_test, y_test)
			
			best_random = grid_SVM.best_estimator_
			random_accuracy = evaluate(best_random, X_test, y_test)
			
			print('\nImprovement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

		if classifier == "LogReg":
			# variations with respect to type of regularization, size of penalty, and type of solver used
			clf = LogisticRegression()
			print('\n+++ Parameters for Logistic Regression +++\n')
			print(clf.get_params())

			# Dict of possible parameters
			params_grid = {'penalty' : ['l2'],
							'C' : (0.1, 1.0), 
    						# 'C' : np.logspace(-4, 4, 20), 
    						'max_iter': [1000, 1500],
    						'solver' : ['lbfgs'], #'newton-cg', 
    						'multi_class': ['ovr']}, #'multinomial'

			grid_LogReg = GridSearchCV(LogisticRegression(), params_grid, verbose=3, cv=CV, n_jobs=-1) 
			# Fit the data 
			grid_LogReg.fit(X_train, y_train)

			# View the best parameters from fitting the Random search
			print('\n+++ Best Params +++\n', grid_LogReg.best_params_)

			base_model = clf
			base_model.fit(X_train, y_train)
			base_accuracy = evaluate(base_model, X_test, y_test)
			
			best_random = grid_LogReg.best_estimator_
			random_accuracy = evaluate(best_random, X_test, y_test)
			
			print('\nImprovement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

	''' Build a model to classify the requirements into 5 categories (1, 2, 3, 4, 5) '''
	def multi_class_classification(self, score_target, smote, save_model=False):
		
		# -------------------------------------------------------------------------------------------------------------------------------------------- #
		# Assigning training and test data
		nlp_feature_matrix = pd.DataFrame(self.features_scaled)
		nlp_feature_matrix = nlp_feature_matrix.loc[:, nlp_feature_matrix.columns != 'req_process']
		
		# check if int or string
		# nlp_feature_matrix['target_is_digit'] = list(map(lambda x: x.isdigit(), nlp_feature_matrix['target']))
		# print(nlp_feature_matrix['target_is_digit'])
		
		# X = nlp_feature_matrix.loc[:, nlp_feature_matrix.columns != 'target']
		# y = nlp_feature_matrix.loc[:, nlp_feature_matrix.columns == 'target']
		# y = y.values.ravel()
		# print('--- NLP FEATURE MATRIX --- \n', nlp_feature_matrix)
		
		# Assigning X, y (used before I saved and loaded feature matrix)
		# X = self.features.loc[:, self.features.columns != 'target']
		# y = self.features.loc[:, self.features.columns == 'target']
		# y = y.values.ravel()

		# Convert to numpy
		y = np.array(nlp_feature_matrix['target'])
		# axis 1 refers to the columns
		X = nlp_feature_matrix.drop('target', axis=1)
		# Saving feature names
		feature_names = list(X.columns)
		# print(feature_names)
		# Convert to numpy array
		X = np.array(X)
		print(X.shape, y.shape)
		
		# Split train and test data
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

		## Testing linearity ##
		# plt.scatter(X[:, 0], y)
		# plt.show()
		# plt.scatter(X[:, 1], y)
		# plt.show()
		# # Applying linear reg
		# from sklearn.linear_model import LinearRegression
		# regressor = LinearRegression().fit(X, y)
		# # Checking the accuracy
		# print(r2_score(regressor.predict(X), y))
		# input()


		# -------------------------------------------------------------------------------------------------------------------------------------------- #
		### Balance Classes using SMOTE ###
		if smote:
			# summarize distribution before oversample
			counter = Counter(y_train)
			for k,v in counter.items():
				per = v / len(y_train) * 100
				print(k, v, per)
				# print('Class=%d, n=%d (%.3f%%)' % (k, v, per)) # NOT WORKING NOW???
			# plot distribution
			pyplot.bar(counter.keys(), counter.values())
			pyplot.ylabel('Number of Samples')
			pyplot.xlabel('Classes')
			pyplot.title('Class Distrubution MultiClass beofre SMOTE')
			# pyplot.show(1)
			pyplot.savefig('./Generated_Files/' + str(score_target) + '/Original_Class_Distribution_Multi_' + str(score_target) + '.png', dpi=300, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures

			oversample = SMOTE()
			X_train, y_train = oversample.fit_resample(X_train, y_train)

			# summarize distribution after oversample
			counter = Counter(y_train)
			for k,v in counter.items():
				per = v / len(y_train) * 100
				print(k, v, per)
				# print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
			# plot the distribution
			pyplot.bar(counter.keys(), counter.values())
			pyplot.ylabel('Number of Samples')
			pyplot.xlabel('Classes')
			pyplot.title('Class Distrubution MultiClass after SMOTE')
			# pyplot.show(1)
			pyplot.savefig('./Generated_Files/' + str(score_target) + '/Oversampled_Class_Distribution_Multi_' + str(score_target) + '.png', dpi=300, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures

			# sm = SMOTE(random_state=27, ratio=1.0)
			# X_train, y_train = sm.fit_sample(X_train, y_train)
			# print(X_train.shape, y_train.shape)
			# y_train = pd.DataFrame(y_train, columns = ['diagnosis'])
			# print(y_train.diagnosis.value_counts())

		# -------------------------------------------------------------------------------------------------------------------------------------------- #	
		### Train the model ###

		if classifier == "RFC":
			clf = RandomForestClassifier(n_estimators=200, max_depth=30, random_state=0)
			print('\n+++ RANDOM FOREST +++\n')
		if classifier == "LogReg":
			clf = LogisticRegression(C=1.0, random_state=42, solver='lbfgs', multi_class='ovr', max_iter=1000)
			print('\n+++ LOGISTIC REGRESSION +++\n')
		if classifier == "SVM":
			clf = svm.SVC(decision_function_shape="ovo", C=10, gamma=0.1, verbose=False, kernel="rbf")
			print('\n+++ SUPPORT VECTORE MACHINE +++\n')
		if classifier == "MNB":
			clf = GaussianNB()
			print('\n+++ MULTINOMIAL NAIVE BAYES +++\n')

		# classifier.fit(X_train, y_train.values.ravel())
		clf.fit(X_train, y_train)
		# Predict model
		predicted = clf.predict(X_test)

		# Peform cross validation
		# cv_score = cross_val_score(classifier, X_train, y_train.values.ravel(), cv=CV)
		cv_score = cross_val_score(clf, X_train, y_train, cv=CV)

		# Store results
		acc = accuracy_score(y_test, predicted)
		prec_rec_f1 = precision_recall_fscore_support(y_test, predicted, average='weighted')
		cv_accuracy = cv_score.mean()
		
		# -------------------------------------------------------------------------------------------------------------------------------------------- #
		# Display results #
		print('\nModel Accuracy:', '{:.1%}'.format(accuracy_score(y_test, predicted).round(3)))             
		print('\nPrecision: ', '{:.1%}'.format(prec_rec_f1[0].round(3)),'\nRecall: ', '{:.1%}'.format(prec_rec_f1[1].round(3)), '\nFscore: ', '{:.1%}'.format(prec_rec_f1[2].round(3)))
		print("\n=== Confusion Matrix === \n", confusion_matrix(y_test, predicted))
		print("\n=== Classification Report ===\n", classification_report(y_test, predicted))
		print('\n=== 10 Fold Cross Validation Scores ===')
		num = 0
		for i in cv_score:
			num += 1
			print('CVFold', num, '=', '{:.1%}'.format(i))
		print("\nMean Cross Validation Score: ", '{:.1%}'.format(cv_accuracy.round(3)))
		print('\n=== Final Model Results ===')
		print('\nAccuracy:', '{:.1%}'.format(accuracy_score(y_test, predicted).round(3)))             
		print('\nPrecision: ', '{:.1%}'.format(prec_rec_f1[0].round(3)),'\nRecall: ', '{:.1%}'.format(prec_rec_f1[1].round(3)), '\nFscore: ', '{:.1%}'.format(prec_rec_f1[2].round(3)))
		# Confusion Matrix 
		conf_mat = confusion_matrix(y_test, predicted)
		fig, ax = plt.subplots(figsize=(8,8))
		sns.heatmap(conf_mat, annot=True, fmt='d')
		plt.ylabel('Actual')
		plt.xlabel('Predicted')
		all_sample_title = 'Accuracy Score: {0}'.format(acc)
		plt.title(all_sample_title, size = 15);
		plt.tight_layout()
		plt.savefig('./Generated_Files/' + str(score_target) + '/Conf_Matrix_Multi_' + str(score_target) + '.png', dpi=300, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures
		# plt.show(block=False) 
		# plt.pause(3)
		plt.clf()
		plt.close()


		def neural_networks():
			print('\nPreparing data for Neural Networks')

			num_classes = 5
			batch_size = 10 # The higher the batch size, the faster to train, but lower accuracy
			
			# remove req_proc

			# num_features = self.dfm_scaled.shape[1] 
			num_features = self.features_scaled.shape[1]
			# print('Num of Features = ', self.dfm_scaled.shape[1])
			print('Num of Features = ', self.features_scaled.shape[1])
			# feature_names = self.dfm_scaled.columns
			feature_names = self.features_scaled.columns
			print(feature_names)
			
			# for col in self.dfm_scaled.columns:
			# 	print(col)
			# for col in self.features_scaled.columns:
			# 	print(col)

    		# feature_columns = [col for col in dfm_scaled.columns]
    		# print(feature_columns) 

			# print('DFM SCALED: \n', self.dfm_scaled)
			# print('FEATURES_SCALED: \n', self.features_scaled)
			# input()
			
			# # get dataset
			scaled_features = self.features_scaled.copy() 
			# scaled_features['req_process'] = self.data['req_process']
			scaled_features['target'] = self.data['target']
			# scaled_data = self.features_scaled
			print('Scaled_Features df \n', scaled_features)
			
			# SPlit test and train data
			train_set, test_set = train_test_split(scaled_features, test_size=0.2, random_state=42) #0.1 test
			
			X_train = train_set.loc[:, train_set.columns != "target"]
			y_train = train_set.target

			X_test = test_set.loc[:, test_set.columns != "target"]
			y_test = test_set.target

			# Reshape training data 
			print('Shape of Training Data:', X_train.shape)
			X_train = np.array(X_train)

			# Reshape dimension of data to fit CNN model
			X_train_res = np.expand_dims(X_train, axis=2)
			X_test_res = np.expand_dims(X_test, axis=2)
			print('Reshaped Data for Neural Networks: ', X_train_res.shape)

			# encode train class values as integers
			encoder = LabelEncoder()
			encoder.fit(y_train)
			encoded_y = encoder.transform(y_train)
			# convert integers to dummy variables (i.e. one hot encoded)
			y_train = np_utils.to_categorical(encoded_y)

			# encode test class values as integers
			encoder = LabelEncoder()
			encoder.fit(y_test)
			encoded_y = encoder.transform(y_test)
			# convert integers to dummy variables (i.e. one hot encoded)
			y_test = np_utils.to_categorical(encoded_y)

			# # The maximum number of words to be used. (most frequent)
			# MAX_NB_WORDS = 50000
			# # Max number of words in each complaint.
			# MAX_SEQUENCE_LENGTH = 250
			# # This is fixed.
			# EMBEDDING_DIM = 100
			# tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
			# tokenizer.fit_on_texts(df['Consumer complaint narrative'].values)
			# word_index = tokenizer.word_index
			# print('Found %s unique tokens.' % len(word_index))
						
			
			def cnn():
				# Build CNN Model
				print('- Building CNN Model -')

				# Reshape dimension of data to fit CNN model
				X_train_res = np.expand_dims(X_train, axis=2)
				X_test_res = np.expand_dims(X_test, axis=2)
				print('Reshaped Data for Neural Networks: ', X_train_res.shape)

				# encode train class values as integers
				encoder = LabelEncoder()
				encoder.fit(y_train)
				encoded_y = encoder.transform(y_train)
				# convert integers to dummy variables (i.e. one hot encoded)
				y_train = np_utils.to_categorical(encoded_y)

				# encode test class values as integers
				encoder = LabelEncoder()
				encoder.fit(y_test)
				encoded_y = encoder.transform(y_test)
				# convert integers to dummy variables (i.e. one hot encoded)
				y_test = np_utils.to_categorical(encoded_y)


				model = Sequential()
				# input_shape = X_train_res[0].shape
				# print('Input shape: ', input_shape)
				# Input Layer
				model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=(26,1)))
				# Dropout to prevent overfitting
				model.add(Dropout(0.25))
				# Hidden Layer 1
				model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
				model.add(Dropout(0.25))
				# Hidden Layer 2
				model.add(Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'))
				model.add(Dropout(0.25))
				# Flatten dimensions of data
				model.add(Flatten())
				# Output Layer
				model.add(Dense(num_classes, activation='softmax'))
				
				model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
				print(model.summary())

				# Fit Model
				model_train = model.fit(X_train_res, y_train, epochs=200, batch_size=batch_size, validation_data=(X_test_res, y_test))

				# model.save("'./Generated_Files/' + str(score_target) + '/Neural_Networks/CNN/model_2HL_100E_50BS.h5py")

				# Evaluate Model on Test Set
				print('Model Evaluation on Test Set')
				test_eval = model.evaluate(X_test_res, y_test, verbose=1)
				print('Test loss:', test_eval[0])
				print('Test accuracy:', test_eval[1])
				predicted_classes = model.predict(X_test_res)
				predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
				print(predicted_classes.shape, y_test.shape)
				correct = np.where(predicted_classes==y_test)[0]
				print("Found %d correct labels" % len(correct))

				for i, correct in enumerate(correct[:9]):
					print(predicted_classes[correct], test_Y[correct])
					# plt.subplot(3,3,i+1)
					# plt.imshow(test_X[correct].reshape(28,28), cmap='gray', interpolation='none')
					# plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_Y[correct]))
					# plt.tight_layout()

				display_plot = True
				if display_plot:
					accuracy = model_train.history['acc']
					val_accuracy = model_train.history['val_acc']
					loss = model_train.history['loss']
					val_loss = model_train.history['val_loss']
					epochs = range(len(accuracy))
					plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
					plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
					plt.title('Training and validation accuracy')
					plt.xlabel('Epochs')
					plt.ylabel('Accuracy')
					plt.legend()
					plt.savefig('./Generated_Files/' + str(score_target) + '/Multi_Class/CNN_Train_Val_Acc_' + str(score_target) + '.png', dpi=300, format='png', bbox_inches='tight')
					plt.show(block=False)
					plt.pause(0.5)
					plt.clf()
					plt.close()

					plt.plot(epochs, loss, 'bo', label='Training loss')
					plt.plot(epochs, val_loss, 'b', label='Validation loss')
					plt.title('Training and validation loss')
					plt.xlabel('Epochs')
					plt.ylabel('Loss')
					plt.legend()
					plt.savefig('./Generated_Files/' + str(score_target) + '/Multi_Class/CNN_Train_Val_Loss_' + str(score_target) + '.png', dpi=300, format='png', bbox_inches='tight')
					plt.show(block=False)
					plt.pause(0.5)
					plt.clf()
					plt.close()
			
				 
			def lstm():
				print('Building LSTM model...')
				# from keras.utils import to_categorical
				# y_binary = to_categorical(y_int)
				
				# print(X_train)
				# print('Y TRAIN \n', y_train) # why are target values represented as 0 or 1?
				# print(y_train.reshape(y_train.shape[0], y_train.shape[1], 1))

				epoch, dropout = 20, 0.2
				print('EPOCH = ', epoch)
				print('DROPOUT = ', dropout)

				model = Sequential()
				model.add(Embedding(input_dim=1000, output_dim=100, input_length=num_features))
				model.add(SpatialDropout1D(0.2))
				model.add(LSTM(100))
				# model.add(RepeatVector(num_features))
				# model.add(LSTM(100, return_sequences=True))
				model.add(Dropout(dropout))
				model.add(Dense(num_classes+1, activation='softmax')) #had to add +1 as range was [0,5] so error when class no. 5 is found
				model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
				# model.compile(optimizer=RMSprop(lr=0.01), loss='categorical_crossentropy',metrics=['acc'])
				model.summary()

				# Train model
				history = model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, verbose=1, validation_split=0.2)
				# history = model.fit(X_train, y_train.values.reshape(y_train.shape[0], y_train.shape[1], 1), epochs=epoch, batch_size=batch_size, verbose=1, validation_split=0.2)
				# Evaluate the model
				loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
				# loss, accuracy = model.evaluate(X_test_res, y_test.reshape(y_test.shape[0], y_test.shape[1], 1), verbose=1)
				print('Accuracy: %f' % (accuracy * 100))

				def display():
					plt.plot(history.history['acc'])
					plt.plot(history.history['val_acc'])

					plt.title('model accuracy')
					plt.ylabel('accuracy')
					plt.xlabel('epoch')
					plt.legend(['train','test'], loc = 'upper left')
					plt.show()

					plt.plot(history.history['loss'])
					plt.plot(history.history['val_loss'])

					plt.title('model loss')
					plt.ylabel('loss')
					plt.xlabel('epoch')
					plt.legend(['train','test'], loc = 'upper left')
					plt.show()
				# display()
			
			cnn()
			lstm()
		# neural_networks()

	''' Build a model to classify the requirements into 2 categories (1, 0) '''
	def binary_classification(self, score_target, smote):
		print('\n==== Binary Classification of Requirements ====')
		
		# -------------------------------------------------------------------------------------------------------------------------------------------- #
		## Create binary dataframe ##
		binary_df = pd.DataFrame()
		
		# binary_df['target'] = self.data['target']
		# for load/save matrix option
		# For use of save and load feature matrix
		# Assigning training and test data
		nlp_feature_matrix = pd.DataFrame(self.features)
		nlp_feature_matrix = nlp_feature_matrix.loc[:, nlp_feature_matrix.columns != 'req_process']
		# print(nlp_feature_matrix)
		# convert target scores to string
		binary_df['target'] = nlp_feature_matrix['target'].apply(str)

		# nlp_feature_matrix['target']= nlp_feature_matrix['target'].apply(str)

		# Replaced good and bad classes
		binary_df['target'] = binary_df.target.replace(['1', '2'], '0')
		binary_df['target'] = binary_df.target.replace(['4', '5'], '1')
		# binary_df['target'] = binary_df.target.replace([1, 2], 0)
		# binary_df['target'] = binary_df.target.replace([4, 5], 1)
		
		# Remove any requirements with target equalling 3
		binary_df['target'] = binary_df['target'][~binary_df.target.str.contains('3')]
		# binary_df['target'] = binary_df['target'][~binary_df.target.contains(3)]

		# not needed when loading feature matrix file
		# features = self.dfm.copy()

		# -------------------------------------------------------------------------------------------------------------------------------------------- #
		## Assign training and test data ##
		X = nlp_feature_matrix.loc[:, nlp_feature_matrix.columns != 'target']
		y = nlp_feature_matrix.loc[:, nlp_feature_matrix.columns == 'target']

		# features = self.features.copy()
		# features = nlp_feature_matrix.copy()

		# *** features['req_process'] = self.data['req_process']                                     
		nlp_feature_matrix['target'] = binary_df['target']

		print(nlp_feature_matrix.shape[0])
		nlp_feature_matrix = nlp_feature_matrix.dropna()
		print(nlp_feature_matrix.shape[0])

		print('Size of Dataframe after removing class 3: ', len(nlp_feature_matrix))

		# Find number of IO and NIO requirements
		IO_counts = len(nlp_feature_matrix[nlp_feature_matrix['target'].str.contains('1')])
		NIO_counts = len(nlp_feature_matrix[nlp_feature_matrix['target'].str.contains('0')])

		print('Num of In Ordnung Requirements: ', IO_counts)
		print('Num of Nicht In Ordnung Requirements: ', NIO_counts)

		# Assigning X, y
		# X = nlp_feature_matrix.loc[:, nlp_feature_matrix.columns != 'target']
		# y = nlp_feature_matrix.loc[:, nlp_feature_matrix.columns == 'target']

		# Assing X and y
		# Convert to numpy
		y = np.array(nlp_feature_matrix['target']).astype(int)
		# axis 1 refers to the columns
		X = nlp_feature_matrix.drop('target', axis=1)
		# Saving feature names
		feature_names = list(X.columns)
		# print(feature_names)
		# Convert to numpy array
		X = np.array(X)
		print(X.shape, y.shape)
		
		# Split train and test data
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

		# -------------------------------------------------------------------------------------------------------------------------------------------- #
		### Balance Classes using SMOTE ###
		if smote:
			# summarize distribution before oversample
			counter = Counter(y_train)
			for k,v in counter.items():
				per = v / len(y_train) * 100
				print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
			# plot distribution
			pyplot.bar(counter.keys(), counter.values())
			pyplot.ylabel('Number of Samples')
			pyplot.xlabel('Classes')
			pyplot.title('Class Distrubution Binary Classification beofre SMOTE')
			# pyplot.show(1)
			pyplot.savefig('./Generated_Files/' + str(score_target) + '/Original_Class_Distribution_Multi_' + str(score_target) + '.png', dpi=300, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures

			oversample = SMOTE()
			X_train, y_train = oversample.fit_resample(X_train, y_train)

			# summarize distribution after oversample
			counter = Counter(y_train)
			for k,v in counter.items():
				per = v / len(y_train) * 100
				print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
			# plot the distribution
			pyplot.bar(counter.keys(), counter.values())
			pyplot.ylabel('Number of Samples')
			pyplot.xlabel('Classes')
			pyplot.title('Class Distrubution Binary Classification after SMOTE')
			# pyplot.show(1)
			pyplot.savefig('./Generated_Files/' + str(score_target) + '/Oversampled_Class_Binary_Classification_' + str(score_target) + '.png', dpi=300, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures

		# -------------------------------------------------------------------------------------------------------------------------------------------- #
		## Build and Train Classifier ##

		if classifier == "RFC":
			clf = RandomForestClassifier(n_estimators=200, max_depth=30, random_state=0)
			print('\n+++ RANDOM FOREST +++\n')
		if classifier == "LogReg":
			clf = LogisticRegression(C=1.0, random_state=42, solver='lbfgs', multi_class='ovr', max_iter=1000)
			print('\n+++ LOGISTIC REGRESSION +++\n')
		if classifier == "SVM":
			clf = svm.SVC(decision_function_shape="ovo", C=10, gamma=0.1, verbose=False, kernel="rbf")
			print('\n+++ SUPPORT VECTORE MACHINE +++\n')
		if classifier == "MNB":
			clf = GaussianNB()
			print('\n+++ MULTINOMIAL NAIVE BAYES +++\n')

		clf.fit(X_train, y_train)
		predicted = clf.predict(X_test)
		print ('Shape of training data: \n', X_train.shape)

		# Peform cross validation
		cv_score = cross_val_score(clf, X_train, y_train, cv=CV)

		# Store results
		acc = accuracy_score(y_test, predicted)
		prec_rec_f1 = precision_recall_fscore_support(y_test, predicted, average='weighted')
		cv_accuracy = cv_score.mean()
		
		# -------------------------------------------------------------------------------------------------------------------------------------------- #
		# Display results
		print('\nModel Accuracy:', '{:.1%}'.format(accuracy_score(y_test, predicted).round(3)))             
		print('\nPrecision: ', '{:.1%}'.format(prec_rec_f1[0].round(3)),'\nRecall: ', '{:.1%}'.format(prec_rec_f1[1].round(3)), '\nFscore: ', '{:.1%}'.format(prec_rec_f1[2].round(3)))
		print("\n=== Confusion Matrix === \n", confusion_matrix(y_test, predicted))
		print("\n=== Classification Report ===\n", classification_report(y_test, predicted))
		print('\n=== 10 Fold Cross Validation Scores ===')
		num = 0
		for i in cv_score:
			num += 1
			print('CVFold', num, '=', '{:.1%}'.format(i))
		print("\nMean Cross Validation Score: ", '{:.1%}'.format(cv_accuracy.round(3)))
		print('\n=== Final Model Results ===')
		print('\nAccuracy:', '{:.1%}'.format(accuracy_score(y_test, predicted).round(3)))             
		print('\nPrecision: ', '{:.1%}'.format(prec_rec_f1[0].round(3)),'\nRecall: ', '{:.1%}'.format(prec_rec_f1[1].round(3)), '\nFscore: ', '{:.1%}'.format(prec_rec_f1[2].round(3)))
		# COnfusion Matrix
		conf_mat = confusion_matrix(y_test, predicted)
		fig, ax = plt.subplots(figsize=(8,8))
		sns.heatmap(conf_mat, annot=True, fmt='d')
		plt.ylabel('Actual')
		plt.xlabel('Predicted')
		all_sample_title = 'Accuracy Score: {0}'.format(acc)
		plt.title(all_sample_title, size = 15);
		plt.tight_layout()
		plt.savefig('./Generated_Files/' + str(score_target) + '/Conf_Matrix_Binary_' + str(score_target) + '.png', dpi=300, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures
		# plt.show(block=False) 
		# plt.pause(5)
		plt.clf()
		plt.close()


		# CNN Model
		def neural_networks():

			num_classes = 2
			batch_size = 10 # The higher the batch size, the faster to train, but lower accuracy
			num_features = self.dfm.shape[1] 
			
			# get dataset
			features = self.dfm.copy() 
			features['target'] = binary_df['target']
			print(features.shape[0])
			features = features.dropna()
			print(features.shape[0])

			print('Size of Dataframe after removing class 3: ', len(features))
			
			# SPlit test and train data
			train_set, test_set = train_test_split(features, test_size=0.2, random_state=42) #0.1 test
			
			X_train = train_set.loc[:, train_set.columns != "target"]
			y_train = train_set.target

			X_test = test_set.loc[:, test_set.columns != "target"]
			y_test = test_set.target

			# Reshape training data to fit baseline model
			print('Shape of Training Data:', X_train.shape)
			X_train = np.array(X_train)

			# Reshape dimension of data to fit CNN model
			X_train_res = np.expand_dims(X_train, axis=2)
			X_test_res = np.expand_dims(X_test, axis=2)
			print('Reshaped Data for CNN: ', X_train_res.shape)

			# encode train class values as integers
			encoder = LabelEncoder()
			encoder.fit(y_train)
			encoded_y = encoder.transform(y_train)
			# convert integers to dummy variables (i.e. one hot encoded)
			y_train = np_utils.to_categorical(encoded_y)

			# encode test class values as integers
			encoder = LabelEncoder()
			encoder.fit(y_test)
			encoded_y = encoder.transform(y_test)
			# convert integers to dummy variables (i.e. one hot encoded)
			y_test = np_utils.to_categorical(encoded_y)

			print('- Building CNN Model -')
			model = Sequential()
			# input_shape = X_train_res[0].shape
			# print('Input shape: ', input_shape)
			# Input Layer
			model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=(26,1)))
			# Dropout to prevent overfitting
			model.add(Dropout(0.25))
			# Hidden Layer 1
			model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
			model.add(Dropout(0.25))
			# Hidden Layer 2
			model.add(Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'))
			model.add(Dropout(0.25))
			# Flatten dimensions of data
			model.add(Flatten())
			# Output Layer
			model.add(Dense(num_classes, activation='softmax'))
			model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
			
			print(model.summary())

			# Fit Model (save for later analysis)
			model_train = model.fit(X_train_res, y_train, epochs=200, batch_size=batch_size, verbose=0, validation_data=(X_test_res, y_test))

			# model.save("'./Generated_Files/' + str(score_target) + '/Neural_Networks/CNN/model_2HL_100E_50BS.h5py")

			# Evaluate Model on Test Set
			print('Model Evaluation on Test Set')
			test_eval = model.evaluate(X_test_res, y_test, verbose=1)
			print('Test loss:', test_eval[0])
			print('Test accuracy:', test_eval[1])

			predicted_classes = model.predict(X_test_res)
			predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
			# print(predicted_classes.shape, y_test.shape)

			correct = np.where(predicted_classes==y_test)[0]
			print("Found %d correct labels" % len(correct))

			for i, correct in enumerate(correct[:9]):
				print(predicted_classes[correct], test_Y[correct])
				# plt.subplot(3,3,i+1)
				# plt.imshow(test_X[correct].reshape(28,28), cmap='gray', interpolation='none')
				# plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_Y[correct]))
				# plt.tight_layout()

			display_plot = True
			if display_plot:
				accuracy = model_train.history['acc']
				val_accuracy = model_train.history['val_acc']
				loss = model_train.history['loss']
				val_loss = model_train.history['val_loss']
				epochs = range(len(accuracy))
				plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
				plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
				plt.title('Training and validation accuracy')
				plt.xlabel('Epochs')
				plt.ylabel('Accuracy')
				plt.legend()
				plt.savefig('./Generated_Files/' + str(score_target) + '/Binary_Class/CNN_Train_Val_Acc_' + str(score_target) + '.png', dpi=300, format='png', bbox_inches='tight')
				plt.show(block=False)
				plt.pause(0.5)
				plt.clf()
				plt.close()

				plt.plot(epochs, loss, 'bo', label='Training loss')
				plt.plot(epochs, val_loss, 'b', label='Validation loss')
				plt.title('Training and validation loss')
				plt.xlabel('Epochs')
				plt.ylabel('Loss')
				plt.legend()
				plt.savefig('./Generated_Files/' + str(score_target) + '/Binary_Class/CNN_Train_Val_Loss_' + str(score_target) + '.png', dpi=300, format='png', bbox_inches='tight')
				plt.show(block=False)
				plt.pause(0.2)
				plt.clf()
				plt.close()
	
		# neural_networks()

		def RNN_model():

			epoch, dropout = 10, 0.2
			print('EPOCH = ', epoch)
			print('DROPOUT = ', dropout)

			model = Sequential()
			model.add(Embedding(input_dim=ger_vocab_size, output_dim=128, input_length=11))
			model.add(LSTM(128))
			model.add(RepeatVector(11))
			model.add(LSTM(128, return_sequences=True))
			model.add(Dropout(dropout))
			model.add(Dense(eng_vocab_size, activation='softmax'))
			model.compile(optimizer=RMSprop(lr=0.01), 
						  loss='sparse_categorical_crossentropy', 
						  metrics=['acc'])
			model.summary()

			# Train model
			history = model.fit(X_train, y_train.reshape(y_train.shape[0], y_train.shape[1], 1), epochs=epoch, batch_size=128, verbose=1, validation_split=0.2)
			# Evaluate the model
			loss, accuracy = model.evaluate(X_test, y_test.reshape(y_test.shape[0], y_test.shape[1], 1), verbose=1)
			print('Accuracy: %f' % (accuracy * 100))

			def display():
				plt.plot(history.history['acc'])
				plt.plot(history.history['val_acc'])

				plt.title('model accuracy')
				plt.ylabel('accuracy')
				plt.xlabel('epoch')
				plt.legend(['train','test'], loc = 'upper left')
				plt.show()

				plt.plot(history.history['loss'])
				plt.plot(history.history['val_loss'])

				plt.title('model loss')
				plt.ylabel('loss')
				plt.xlabel('epoch')
				plt.legend(['train','test'], loc = 'upper left')
				plt.show()
			display()

	''' Build a model to classify the requirements into 3 categories (-1, 0, 1) '''
	def tri_classification(self, score_target, smote):
		print('\n=== Tri Class CLassification ===\n')
		# -------------------------------------------------------------------------------------------------------------------------------------------- #
		# Convert into 3 Classes
		tri_df = self.features_scaled.copy() # use for load/save matrix
		tri_df = tri_df.drop('req_process', axis=1)
		
		# tri_df['target'] = self.data['target']
		tri_df['target'] = self.features['target'].astype(str)
		tri_df['target'] = tri_df.target.replace(['1', '2'], '-1')
		tri_df['target'] = tri_df.target.replace(['3'], '0')
		tri_df['target'] = tri_df.target.replace(['4', '5'], '1')

		nlp_feature_matrix = pd.DataFrame(self.features_scaled)
		nlp_feature_matrix = nlp_feature_matrix.loc[:, nlp_feature_matrix.columns != 'req_process']

		nlp_feature_matrix = tri_df.copy()
		# features['req_process'] = self.data['req_process']
		print(nlp_feature_matrix.shape[0])
		nlp_feature_matrix = nlp_feature_matrix.dropna() 
		print(nlp_feature_matrix.shape[0])
		print(nlp_feature_matrix.head())
		
		# -------------------------------------------------------------------------------------------------------------------------------------------- #
		# Assinging training and test data 
		# Convert to numpy
		y = np.array(nlp_feature_matrix['target']).astype(int)
		# axis 1 refers to the columns
		X = nlp_feature_matrix.drop('target', axis=1)
		# Saving feature names
		feature_names = list(X.columns)
		print(feature_names)
		# Convert to numpy array
		X = np.array(X)
		print(X.shape, y.shape)
		
		# Split train and test data
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

		# -------------------------------------------------------------------------------------------------------------------------------------------- #
		### Balance Classes using SMOTE ###
		if smote:
			# summarize distribution before oversample
			counter = Counter(y_train)
			for k,v in counter.items():
				per = v / len(y_train) * 100
				print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
			# plot distribution
			pyplot.bar(counter.keys(), counter.values())
			pyplot.ylabel('Number of Samples')
			pyplot.xlabel('Classes')
			pyplot.title('Class Distrubution Binary Classification beofre SMOTE')
			# pyplot.show(1)
			pyplot.savefig('./Generated_Files/' + str(score_target) + '/Original_Class_Distribution_Tri_' + str(score_target) + '.png', dpi=300, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures

			oversample = SMOTE()
			X_train, y_train = oversample.fit_resample(X_train, y_train)

			# summarize distribution after oversample
			counter = Counter(y_train)
			for k,v in counter.items():
				per = v / len(y_train) * 100
				print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
			# plot the distribution
			pyplot.bar(counter.keys(), counter.values())
			pyplot.ylabel('Number of Samples')
			pyplot.xlabel('Classes')
			pyplot.title('Class Distrubution Binary Classification after SMOTE')
			# pyplot.show(1)
			pyplot.savefig('./Generated_Files/' + str(score_target) + '/Oversampled_Class_Distribution_Tri_' + str(score_target) + '.png', dpi=300, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures
		
		# -------------------------------------------------------------------------------------------------------------------------------------------- #
		# Build  Model
		if classifier == "RFC":
			print('\n+++ RANDOM FOREST +++\n')
			clf = RandomForestClassifier(n_estimators=200, max_depth=30, random_state=0)
		if classifier == "LogReg":
			print('\n+++ LOGISTIC REGRESSION +++\n')
			clf = LogisticRegression(C=1.0, random_state=42, solver='lbfgs', multi_class='ovr', max_iter=1000)
		if classifier == "SVM":
			print('\n+++ SUPPORT VECTORE MACHINE +++\n')
			clf = svm.SVC(decision_function_shape="ovo", C=10, gamma=0.1, verbose=False, kernel="rbf")
		if classifier == "MNB":
			clf = GaussianNB()
			print('\n+++ MULTINOMIAL NAIVE BAYES +++\n')

		clf.fit(X_train, y_train)
		predicted = clf.predict(X_test)
		# print ('Shape of training data: \n', X_train.shape)

		# Peform cross validation
		cv_score = cross_val_score(clf, X_train, y_train, cv=CV)

		# Store results
		acc = accuracy_score(y_test, predicted)
		prec_rec_f1 = precision_recall_fscore_support(y_test, predicted, average='weighted')
		cv_accuracy = cv_score.mean()
		
		# -------------------------------------------------------------------------------------------------------------------------------------------- #
		# Display results
		print('\nModel Accuracy:', '{:.1%}'.format(accuracy_score(y_test, predicted).round(3)))             
		print('\nPrecision: ', '{:.1%}'.format(prec_rec_f1[0].round(3)),'\nRecall: ', '{:.1%}'.format(prec_rec_f1[1].round(3)), '\nFscore: ', '{:.1%}'.format(prec_rec_f1[2].round(3)))
		print("\n=== Confusion Matrix === \n", confusion_matrix(y_test, predicted))
		print("\n=== Classification Report ===\n", classification_report(y_test, predicted))
		print('\n=== 10 Fold Cross Validation Scores ===')
		num = 0
		for i in cv_score:
			num += 1
			print('CVFold', num, '=', '{:.1%}'.format(i))
		print("\nMean Cross Validation Score: ", '{:.1%}'.format(cv_accuracy.round(3)))
		print('\n=== Final Model Results ===')
		print('\nAccuracy:', '{:.1%}'.format(accuracy_score(y_test, predicted).round(3)))             
		print('\nPrecision: ', '{:.1%}'.format(prec_rec_f1[0].round(3)),'\nRecall: ', '{:.1%}'.format(prec_rec_f1[1].round(3)), '\nFscore: ', '{:.1%}'.format(prec_rec_f1[2].round(3)))
		# Confusion Matrix
		conf_mat = confusion_matrix(y_test, predicted)
		fig, ax = plt.subplots(figsize=(8,8))
		sns.heatmap(conf_mat, annot=True, fmt='d')
		plt.ylabel('Actual')
		plt.xlabel('Predicted')
		all_sample_title = 'Accuracy Score: {0}'.format(acc)
		plt.title(all_sample_title, size = 15);
		plt.tight_layout()
		plt.savefig('./Generated_Files/' + str(score_target) + '/Conf_Matrix_Tri_' + str(score_target) + '.png', dpi=300, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures
		# plt.show(block=False) 
		# plt.pause(5)
		plt.clf()
		plt.close()

		# CNN Model
		def neural_networks():

			num_classes = 3
			batch_size = 10 # The higher the batch size, the faster to train, but lower accuracy
			num_features = self.dfm.shape[1] 
			
			# get dataset
			features = self.dfm.copy() 
			# features['req_process'] = self.data['req_process']
			# features['target'] = self.data['target']
			features['target'] = tri_df['target']
			
			# SPlit test and train data
			train_set, test_set = train_test_split(features, test_size=0.2, random_state=42) #0.1 test
			
			X_train = train_set.loc[:, train_set.columns != "target"]
			y_train = train_set.target

			X_test = test_set.loc[:, test_set.columns != "target"]
			y_test = test_set.target

			# Reshape training data to fit baseline model
			print('Shape of Training Data:', X_train.shape)
			X_train = np.array(X_train)

			# Reshape dimension of data to fit CNN model
			X_train_res = np.expand_dims(X_train, axis=2)
			X_test_res = np.expand_dims(X_test, axis=2)
			print('Reshaped Data for CNN: ', X_train_res.shape)

			# encode train class values as integers
			encoder = LabelEncoder()
			encoder.fit(y_train)
			encoded_y = encoder.transform(y_train)
			# convert integers to dummy variables (i.e. one hot encoded)
			y_train = np_utils.to_categorical(encoded_y)

			# encode test class values as integers
			encoder = LabelEncoder()
			encoder.fit(y_test)
			encoded_y = encoder.transform(y_test)
			# convert integers to dummy variables (i.e. one hot encoded)
			y_test = np_utils.to_categorical(encoded_y)

			print('- Building CNN Model -')
			model = Sequential()
			# input_shape = X_train_res[0].shape
			# print('Input shape: ', input_shape)
			# Input Layer
			model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=(26,1)))
			# Dropout to prevent overfitting
			model.add(Dropout(0.25))
			# Hidden Layer 1
			model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
			model.add(Dropout(0.25))
			# Hidden Layer 2
			model.add(Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'))
			model.add(Dropout(0.25))
			# Flatten dimensions of data
			model.add(Flatten())
			# Output Layer
			model.add(Dense(num_classes, activation='softmax'))
			model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
			
			print(model.summary())

			# Fit Model (save for later analysis)
			model_train = model.fit(X_train_res, y_train, epochs=200, batch_size=batch_size, validation_data=(X_test_res, y_test))

			# model.save("'./Generated_Files/' + str(score_target) + '/Neural_Networks/CNN/model_2HL_100E_50BS.h5py")

			# Evaluate Model on Test Set
			print('Model Evaluation on Test Set')
			test_eval = model.evaluate(X_test_res, y_test, verbose=1)
			print('Test loss:', test_eval[0])
			print('Test accuracy:', test_eval[1])

			predicted_classes = model.predict(X_test_res)
			predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
			# print(predicted_classes.shape, y_test.shape)

			correct = np.where(predicted_classes==y_test)[0]
			print("Found %d correct labels" % len(correct))

			for i, correct in enumerate(correct[:9]):
				print(predicted_classes[correct], test_Y[correct])
				# plt.subplot(3,3,i+1)
				# plt.imshow(test_X[correct].reshape(28,28), cmap='gray', interpolation='none')
				# plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_Y[correct]))
				# plt.tight_layout()

			display_plot = True
			if display_plot:
				accuracy = model_train.history['acc']
				val_accuracy = model_train.history['val_acc']
				loss = model_train.history['loss']
				val_loss = model_train.history['val_loss']
				epochs = range(len(accuracy))
				plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
				plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
				plt.title('Training and validation accuracy')
				plt.xlabel('Epochs')
				plt.ylabel('Accuracy')
				plt.legend()
				plt.savefig('./Generated_Files/' + str(score_target) + '/Tri_Class/CNN_Train_Val_Acc_' + str(score_target) + '.png', dpi=300, format='png', bbox_inches='tight')
				plt.show(block=False)
				plt.pause(0.2)
				plt.clf()
				plt.close()

				plt.plot(epochs, loss, 'bo', label='Training loss')
				plt.plot(epochs, val_loss, 'b', label='Validation loss')
				plt.title('Training and validation loss')
				plt.xlabel('Epochs')
				plt.ylabel('Loss')
				plt.legend()
				plt.savefig('./Generated_Files/' + str(score_target) + '/Tri_Class/CNN_Train_Val_Loss_' + str(score_target) + '.png', dpi=300, format='png', bbox_inches='tight')
				plt.show(block=False)
				plt.pause(0.2)
				plt.clf()
				plt.close()
	

		# neural_networks()

	''' A function to test model performance using only tfidf features '''
	def tfidf_features(self, score_target):
		# Only works when creating new features matrix (i.e not to be used when loading matrix)
		# https://datascience.stackexchange.com/questions/22813/using-tf-idf-with-other-features-in-sklearn
		# https://stackoverflow.com/questions/48573174/how-to-combine-tfidf-features-with-other-features
		
		''' The idea is to test whether the model performs better when using only the natural features found in the data - the tfidf featureds
			compared to using the NLP features.

			RESULTS: There seems to be no improvement in model performance when using only TFIDF features. 

			* create a feature matrix using tfidf?
		'''

		print('\n===== Tfidf Features =====\n')
		
		# Create a dataframe with only single requirements
		df = pd.DataFrame()
		df['Requirement'] = self.features['req_process'] # Should be req_process?
		df['Rating'] = self.features['target']
		df['category_id'] = self.features['target'].values
		print(df.head)

		category_id_df = df[['Requirement', 'category_id']].sort_values('category_id')
		category_to_id = dict(category_id_df.values)
		id_to_category = dict(category_id_df[['category_id', 'Requirement']].values)

		tfidf = TfidfVectorizer(sublinear_tf=True, min_df=50, norm='l2', encoding='utf-8', ngram_range=(1, 2), stop_words=self.ger_stopws)

		features = tfidf.fit_transform(df.Requirement).toarray()
		labels = df.category_id 
		print('\nShape of tfidf features matrix: ', features.shape)
		print('\ntfidf features matrix: \n', features)
	   
		# Split test, train and perform tfidf vectorizer
		X_train, X_test, y_train, y_test = train_test_split(df['Requirement'], df['Rating'], random_state = 0)

		count_vect = CountVectorizer()
		X_train_counts = count_vect.fit_transform(X_train)
		tfidf_transformer = TfidfTransformer()
		X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

		# Test different classifiers
		models = [
				RandomForestClassifier(n_estimators=200, max_depth=30, random_state=0, class_weight='balanced'),
				svm.SVC(decision_function_shape="ovo", C=10, gamma=0.1, verbose=False, kernel="rbf"),
				LinearSVC(),
				MultinomialNB(),
				LogisticRegression(C=1.0, random_state=42, solver='lbfgs', multi_class='ovr', max_iter=1000),
				]
		# print('Model = RFC \n', models[0] )
		index = 0
		# Cross Validation
		cv_df = pd.DataFrame(index=range(CV * len(models)))
		entries = []
		# Find best model
		for model in models:
		  model_name = model.__class__.__name__
		  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
		  for fold_idx, accuracy in enumerate(accuracies):
		  	entries.append((model_name, fold_idx, accuracy))
		cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

		print('\nModel Performance: \n', cv_df.groupby('model_name').accuracy.mean().round(3), '\n')

		X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.20, random_state=0)
		models[index].fit(X_train, y_train)
		y_pred = models[index].predict(X_test)

		# acc = np.mean(y_test == y_pred)
		acc = accuracy_score(y_pred, y_test)
		prec_rec_f1 = precision_recall_fscore_support(y_test, y_pred, average='weighted')
		print('\n--- TFIDF Results ---')
		print('Accuracy: ', '{:.1%}'.format(accuracy_score(y_test.values.ravel(), y_pred).round(3)))
		# print("\nPrec_Recall_FScore: \n", precision_recall_fscore_support(y_test, y_pred, average='weighted'))
		print('\nPrecision: ', '{:.1%}'.format(prec_rec_f1[0].round(3)),'\nRecall: ', '{:.1%}'.format(prec_rec_f1[1].round(3)), '\nFscore: ', '{:.1%}'.format(prec_rec_f1[2].round(3)))
		
		# Feature Importance when using Random Forest
		if index == 0:
			feat_importance = models[0].feature_importances_
			# print(feat_importance)
			feat_names = tfidf.get_feature_names()
			# print(feat_names)
			df_feat_importance = pd.DataFrame()
			df_feat_importance['feat'] = feat_names
			df_feat_importance['importance'] = feat_importance
			print('\n--- Top TFIDF features ---\n', df_feat_importance.sort_values(by='importance', ascending=False).head(), '\n')


		# Display
		conf_mat = confusion_matrix(y_test, y_pred)
		fig, ax = plt.subplots(figsize=(8,8))
		sns.heatmap(conf_mat, annot=True, fmt='d')
		plt.ylabel('Actual')
		plt.xlabel('Predicted')
		all_sample_title = 'Accuracy Score: {0}'.format(acc)
		plt.title(all_sample_title, size = 15);
		plt.tight_layout()
		plt.savefig('./Generated_Files/' + str(score_target) + '/Conf_Matrix_TFIDF_Multi_' + str(score_target) + '.png', dpi=300, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures
		plt.show(block=False) 
		plt.pause(5)
		plt.clf()
		plt.close()

	def test_function(self, score_target):
		print(self.features)
				



if __name__ == '__main__':

	# file = u'/Users/selina/Code/Python/Thesis_Code/software_requirements_small.xlsx'
	file = u'/Users/selina/Code/Python/Thesis_Code/software_requirements_full.xlsx'
	score_target = 'score'
	extractor = Specification_Evaluation(file)
	# Load saved features_matrix once model has been run
	extractor.load_features_matrix(score_target)
	'''
	extractor.get_data(score_target)
	extractor.get_features(score_target, metric_create=True, export_dfm=True, hotenc=True)
	extractor.preprocessing(score_target)
	extractor.save_features_matrix(score_target)
	'''
	extractor.multi_class_classification(score_target, smote=False)
	extractor.binary_classification(score_target, smote=False)
	extractor.tri_classification(score_target, smote=False)
	# extractor.tfidf_features(score_target)
	# extractor.classifier_comparison(score_target)
	# extractor.grid_search(score_target)
	# extractor.test_function(score_target)
