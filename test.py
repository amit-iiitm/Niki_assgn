import csv
import sklearn
import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

#load the vectorizer
vectorizer = joblib.load('vectorizer.pkl')

# Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()
print vocab



#load the random forest classifier
forest = joblib.load('rf.pkl')


#testing the classifier
X_sents_test=[]
Y_class_test=[]
Features_test=[]
with open('testdata.csv','r') as f:
	for line in f:
		temp=""
		print line
		
		class_sent=line.split(':')
		ques=class_sent[1].strip()
		class_ques=class_sent[0].strip()
		tokens=word_tokenize(ques)
		tagged_tokens=pos_tag(tokens)
		#print tagged_tokens
		for tagged_token in tagged_tokens:
			if tagged_token[1] in ['WRB','WP','WDT','WP$']:
				#temp.append(tagged_token[0])
				temp=temp+" "+tagged_token[0]
		for tagged_token in tagged_tokens:
			if tagged_token[1] in ['NN','NNS','NNP','NNPS']:
				#temp.append(tagged_token[0])
				temp=temp+" "+tagged_token[0]
				break;
		#print temp
		Features_test.append(temp)
		X_sents_test.append(ques)
		Y_class_test.append(class_ques)
		


#print X_sents_test
print Y_class_test
print Features_test
print "Creating the bag of words for the test data...\n"




# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(Features_test)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

print result
correct=0
total=0
for i in range(len(result)):
	if Y_class_test[i]==result[i]:
		correct=correct+1
	total=total+1

print correct
print total


#user interactiont
print "You wanna test! Please go ahead"
ch='Y'
while ch=='Y':
	print "hello user"
	print "Enter Y to continue, N to go back:"
	ch=str(raw_input())
	test_sent=str(raw_input())
	temp=""
	tokens=word_tokenize(test_sent)
	tagged_tokens=pos_tag(tokens)
	#print tagged_tokens
	#checking for affirmative questions 
	#the rule is these type of questions start with a verb phrase(is,was,being,has,does) or a noun phrase(am,are,have,had,do,did,will,shall) or a model MD(can,could,should,would) or adverb(were)
	#print tagged_tokens[0]
	if tagged_tokens[0][1] in ['VB','VBD','VBG','VBN','VBP','VBZ','NNP','RB','MD']:
		print "class: Affirmation"
		print
	else:
		for tagged_token in tagged_tokens:
			if tagged_token[1] in ['WRB','WP','WDT','WP$']:
				#temp.append(tagged_token[0])
				temp=temp+" "+tagged_token[0]
		for tagged_token in tagged_tokens:
			if tagged_token[1] in ['NN','NNS','NNP','NNPS']:
				temp=temp+" "+tagged_token[0]
				break;
		#print temp
		temp_list=[]
		temp_list.append(temp)
		#print temp_list
		user_data_features = vectorizer.transform(temp_list)
		user_data_features = user_data_features.toarray()
		user_result=forest.predict(user_data_features)
		print "Class: ", user_result[0]
		print
