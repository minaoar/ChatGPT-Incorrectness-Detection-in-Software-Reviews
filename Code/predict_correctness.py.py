# suppress warnings
import warnings
warnings.filterwarnings("ignore")

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score


# draw a scatter plot
import matplotlib.pyplot as plt

# MR_ALL 42, 42 --> .81
# MR1+MR3 (without context) 42 5 --> 75
# MR1 42 5 --> 74
# 42 5, MR_ALL .80, 
# 42 5, MR1 .74,
# 42 42, MR1+MR3 .75,


RANDOM_STATE = 42
def k_fold_cross_validation(dataset, k):
#    dataset = dataset.sample(frac=1).reset_index(drop=True)
    dataset = dataset.sample(frac=1,random_state=42).reset_index(drop=True)

    num_samples_per_fold = len(dataset) // k

    k_folds = []

    for i in range(k):
        start_idx = i * num_samples_per_fold
        end_idx = start_idx + num_samples_per_fold

        if i == k - 1:
            end_idx = len(dataset)

        test_data = dataset.iloc[start_idx:end_idx]

        train_data = pd.concat([dataset.iloc[:start_idx], dataset.iloc[end_idx:]])

        k_folds.append((train_data, test_data))

    return dataset, k_folds


data_file = '' # input data file

df = pd.read_csv(data_file)


def train_model(column_names, clf_all, write_file = False, file_suffix = ''):

	df_logistic_all = df

	# run a for loop for 10 times
	total_acc = 0


	fold = 10
	df_logistic_all, k_folds_data = k_fold_cross_validation(df_logistic_all, k=fold)

	y_pred_all = []
	y_pred_proba_all = []

	# create a new datafram x_test_all
	test_data_all = pd.DataFrame()

	for i, (train_data, test_data) in enumerate(k_folds_data):
		# print(f"Fold {i+1}:")
		# print(len(train_data))
		# print(len(test_data))
		X_train = train_data[column_names]
		y_train = train_data['label']
		X_test = test_data[column_names]
		y_test = test_data['label']

		from imblearn.over_sampling import RandomOverSampler, SMOTE,SVMSMOTE,ADASYN,BorderlineSMOTE
		from imblearn.under_sampling import RandomUnderSampler
		# ros = SVMSMOTE(random_state=RANDOM_STATE,sampling_strategy=1)
		#ros = SMOTE(random_state=RANDOM_STATE,sampling_strategy=1)
		ros = BorderlineSMOTE(random_state=RANDOM_STATE,sampling_strategy=1, kind='borderline-1')
		X_train, y_train = ros.fit_resample(X_train, y_train)

		#clf_all = svm.SVC(probability=True,kernel='rbf')
		clf_all.fit(X_train, y_train)

		y_pred = clf_all.predict(X_test)
		y_pred_all = [*y_pred_all, *y_pred]

		y_pred_proba = clf_all.predict_proba(X_test)
		y_pred_proba_all = [*y_pred_proba_all , *y_pred_proba]

		# append the test_data to test_data_all
		test_data_all = test_data_all.append(test_data, ignore_index=True)


	from sklearn.metrics import classification_report
        
	report = classification_report(test_data_all['label'], y_pred_all, output_dict=True)	
	print(classification_report(test_data_all['label'], y_pred_all))

	# print the confusion matrix
	from sklearn.metrics import confusion_matrix
	print(confusion_matrix(test_data_all['label'], y_pred_all))

	y_pred_proba_all = np.array(y_pred_proba_all)


	test_data_all['predict_probo_0'] = y_pred_proba_all[:,0]
	test_data_all['predict_probo_1'] = y_pred_proba_all[:,1]
	test_data_all['predicted_label'] = y_pred_all

	# get the accuracy score
	acc_score = accuracy_score(test_data_all['label'], y_pred_all)

	# get the f1 score

	f1_score_weighted = f1_score(test_data_all['label'], y_pred_all, average='weighted')
	
        
	# get the f1 score for class 0, not class 1
	f1_score_0 = report['0']['f1-score']
	print('F1 score: ', report['0']['f1-score'])
        

	if write_file:
		# check file_suffix
		if len(file_suffix) > 0:
			file_suffix = '_' + file_suffix

		new_file_suffix = file_suffix + '_predicted.csv'
		# write the test_data_all to a file
		test_data_all.to_csv(data_file.replace('.csv', new_file_suffix), index=False)
        
		
	return acc_score, f1_score_weighted # f1_score_0
        

	# test_data_all.to_excel('prediction.xlsx', index=False)




model_logistic = LogisticRegression(random_state=RANDOM_STATE)
model_random_forest = RandomForestClassifier(max_depth=2, random_state=RANDOM_STATE)
model_svm = svm.SVC(probability=True,kernel='rbf')
models = {'logistic': model_logistic, 'random_forest': model_random_forest, 'svm': model_svm}


# get column names that start with sim_
column_names = df.columns[df.columns.str.startswith('sim_')]

# put the columns_names in a dictionary with label 'all_attributes'
column_names_dict = {'All': column_names}

# get column names that start with sim_. Also add column 'gpt'
column_names = df.columns[df.columns.str.startswith('sim_')]
column_names = [*column_names, 'gpt']

# put the columns_names in a dictionary with label 'all_gpt'
column_names_dict['All_GPT'] = column_names






# get column names that start with sim_ and end with _qaqa
column_names = df.columns[df.columns.str.startswith('sim_') & df.columns.str.endswith('_qaqa')]

# put the columns_names in a dictionary with label 'qaqa_attributes'
column_names_dict['Only_Mutated_Challenges'] = column_names

# column_names = df.columns[df.columns.str.endswith('_qaqa')]

# get column names that starts with sim_ but does not end with _qaqa
column_names = df.columns[df.columns.str.startswith('sim_') & ~df.columns.str.endswith('_qaqa')]

# put the columns_names in a dictionary with label 'regular_attributes'
column_names_dict['Only_Basic_Challenges'] = column_names

# get column names that starts with sim_ and contains _ans_
column_names = df.columns[df.columns.str.startswith('sim_') & df.columns.str.contains('_ans_')]

# put the columns_names in a dictionary with label 'answer_attributes'
column_names_dict['Only_Answers'] = column_names

# get column names that starts with sim_ and contains _ques_ and does not contain _ans_
column_names = df.columns[df.columns.str.startswith('sim_') & df.columns.str.contains('_ques_') & ~df.columns.str.contains('_ans_')]

# put the columns_names in a dictionary with label 'question_attributes'
column_names_dict['Only_Questions'] = column_names

# only the basic challenges and answers
column_names = df.columns[df.columns.str.startswith('sim_') & ~df.columns.str.endswith('_qaqa') & df.columns.str.contains('_ans_')]

# put the columns_names in a dictionary with label 'basic_answers_attributes']
column_names_dict['Basic_Challenges_Answers'] = column_names

# only the qaqa challenges and answers
column_names = df.columns[df.columns.str.startswith('sim_') & df.columns.str.endswith('_qaqa') & df.columns.str.contains('_ans_')]
# put the columns_names in a dictionary with label 'qaqa_answers_attributes'
column_names_dict['Mutated_Challenges_Answers'] = column_names

# columns without how
column_names = df.columns[df.columns.str.startswith('sim_') & ~df.columns.str.contains('_how_')]
# put the columns_names in a dictionary with label 'without_how'
column_names_dict['Without_How'] = column_names

# columns without why
column_names = df.columns[df.columns.str.startswith('sim_') & ~df.columns.str.contains('_why_')]
# put the columns_names in a dictionary with label 'without_why'
column_names_dict['Without_Why'] = column_names

# columns without really
column_names = df.columns[df.columns.str.startswith('sim_') & ~df.columns.str.contains('_really_')]
# put the columns_names in a dictionary with label 'without_really'
column_names_dict['Without_Really'] = column_names

# only answers without really
column_names = df.columns[df.columns.str.startswith('sim_') & df.columns.str.contains('_ans_') & ~df.columns.str.contains('_really_')]
# put the columns_names in a dictionary with label 'only_answers_without_really'
column_names_dict['Only_Answers_Without_Really'] = column_names

# either column name contains ans or (ques and qaqa). All columns must contain sim_
column_names = df.columns[df.columns.str.startswith('sim_') & (df.columns.str.contains('_ans_') | (df.columns.str.contains('_ques_') & df.columns.str.contains('_qaqa')))]
# put the columns_names in a dictionary with label 'only_answers_without_really'
column_names_dict['All_Answers_Mutated_Questions'] = column_names


# either column name contains ans or (ques and qaqa). All columns must contain sim_. No column should contain really
column_names = df.columns[df.columns.str.startswith('sim_') & (df.columns.str.contains('_ans_') | (df.columns.str.contains('_ques_') & df.columns.str.contains('_qaqa'))) & ~df.columns.str.contains('_really_')]
# put the columns_names in a dictionary with label 'only_answers_without_really'
column_names_dict['All_Answers_Mutated_Questions_Without_Really'] = column_names



def experiemnt_all(specific_columns='', specific_model=''):
	df_accuracy_f1 = pd.DataFrame(columns=['model', 'columns', 'accuracy', 'f1'])
	# df_accuracy = pd.DataFrame(columns=['model', 'columns', 'accuracy'])


	for model_name, model in models.items():
		if len(specific_model) > 0 and model_name != specific_model:
			continue

		for columns, value in column_names_dict.items():
			if len(specific_columns) > 0 and columns != specific_columns:
				continue

			print('-------------------')
			print(model_name)
			# print(columns)
			# print(value)
			acc_score, f1 = train_model(value, model)

			# add the accuracy and f1 weighted score to the dataframe
			df_accuracy_f1 = df_accuracy_f1.append({'model': model_name, 'columns': columns, 'accuracy': acc_score, "f1": f1}, ignore_index=True)

	# print the dataframe in a readable format after dropping the index

	print(df_accuracy_f1.to_string(index=False))

	# print only the accuracy values. models should be in columns, columns should be index
	df_acc = df_accuracy_f1.pivot(index='columns', columns='model', values='accuracy')

	print("\n-----------Accuracy-------")
	print(df_acc.to_string())


	# print only the f1 values. models should be in columns, columns should be index
	df_f1 = df_accuracy_f1.pivot(index='columns', columns='model', values='f1')

	print("\n-----------F1 score -------")
	print(df_f1.to_string())


# experiemnt_all('All')
# experiemnt_all(specific_model='svm')
# experiemnt_all()
# train_model(column_names_dict['All_GPT'], models['svm'], write_file=True)
# train_model(column_names_dict['All'], models['svm'], write_file=True)
# train_model(column_names_dict['All'], models['svm'], write_file=True)
# train_model(column_names_dict['All'], models['svm'], write_file=False)

setting = 'Basic_Challenges_Answers'
train_model(column_names_dict[setting], models['svm'], write_file=True, file_suffix=setting)
# setting = 'Mutated_Challenges_Answers'
# train_model(setting, models['svm'], write_file=True, file_suffix=setting)
