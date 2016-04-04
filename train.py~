import csv
import sklearn
import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


X_sents=[]
Y_class=[]
Features=[]
with open('traindata.csv','r') as f:
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
		Features.append(temp)
		X_sents.append(ques)
		Y_class.append(class_ques)
		

#print X_sents
print Y_class
print Features
print "Creating the bag of words...\n"
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(Features)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()
print train_data_features.shape



# Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()
print vocab

# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
for tag, count in zip(vocab, dist):
    print count, tag

print train_data_features[0]
with open('features.txt','w') as f:
	f.write(train_data_features)



print "Training the random forest..."


# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, Y_class )

#pickle the vectorizer
joblib.dump(vectorizer,'vectorizer.pkl')
#pickle the rf classifier
joblib.dump(forest,'rf.pkl') 


















print "Now the testing phase starts"
print "Now the testing phase starts"
print "Now the testing phase starts"
print "Now the testing phase starts"



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




#user interactive classification
print "You wanna test! Please go ahead"
print "Enter Y to continue, N to go back:"
ch=str(raw_input())
while ch=='Y':
	print "hello dumbo"
	print "Enter Y to continue, N to go back:"
	ch=str(raw_input())
	test_sent=str(raw_input())
	temp=""
	tokens=word_tokenize(test_sent)
	tagged_tokens=pos_tag(tokens)
	print tagged_tokens
	for tagged_token in tagged_tokens:
		if tagged_token[1] in ['WRB','WP','WDT','WP$']:
			#temp.append(tagged_token[0])
			temp=temp+" "+tagged_token[0]
	for tagged_token in tagged_tokens:
		if tagged_token[1] in ['NN','NNS','NNP','NNPS']:
			temp=temp+" "+tagged_token[0]
			break;
	print temp
	temp_list=[]
	temp_list.append(temp)
	print temp_list
	user_data_features = vectorizer.transform(temp_list)
	user_data_features = user_data_features.toarray()
	user_result=forest.predict(user_data_features)
	print user_result
