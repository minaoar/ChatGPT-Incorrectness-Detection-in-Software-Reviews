
# suppress warnings
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np


file_path = '' # path to the file

# read the data file
df = pd.read_csv(file_path)
# keep the rows where question_approach is Ask deeper question
df = df[df['question_approach'] == 'Ask deeper question']

# trim the answer column before [ and after ]
df['answer'] = df['answer'].apply(lambda x: x[x.find('['):x.find(']')+1])

# parse the answer column as json
import json
df['answer'] = df['answer'].apply(lambda x: json.loads(x))

# explode the answer column into multiple rows for each json key
df = df.explode('answer')

# create a new column for each json key
df['title'] = df['answer'].apply(lambda x: x['title'])
df['explanation'] = df['answer'].apply(lambda x: x['explanation'])

print(df.head(10))

# save the file
df.to_csv(file_path.replace('.csv', '-exploded.csv'), index=False)

target_file ='' # path to the file

# read the data file
df_target = pd.read_csv(target_file)

# lower case the question_keyword column
df_target['question_keyword'] = df_target['question_keyword'].str.lower()
df['title'] = df['title'].str.lower()


# merge the two data frames based on the question id and df.title and df_target.question_keyword. Add only the explanation column from df to df_target
df_target = pd.merge(df_target, df[['id', 'title', 'explanation']], left_on=['id', 'question_keyword'], right_on=['id', 'title'], how='left')

print(df_target.head(10))

# save the file
df_target.to_csv(target_file.replace('.csv', '-exploded.csv'), index=False)