
import pandas as pd

import os
import openai
from datetime import datetime

import json
import time

openai.api_key_path = '.env'
MAX_RETRY = 20
MAX_API_CALLS = 500
MAX_TOKEN_SIZE = 512

log_dir = '' # set the log directory
log_file = log_dir + 'log_july04.csv'

data_dir = '' # set the data directory

input_file = data_dir + 'so_nltk_spacy_data_similar_answers_qaqa_applied_EQ.csv'


# suppress warnings
import warnings
warnings.filterwarnings("ignore")

def write_log(prompt, model_output, role, model, version):
  # log the prompt, model output, and the role, and the time, model name, and the model version
  if os.path.exists(log_file):
    log_df = pd.read_csv(log_file)
  else:
    log_df = pd.DataFrame(columns=['Prompt', 'Model Output', 'Role', 'Time', 'Model', 'Version', 'Callid'])
  
  # generate a call id
  callid = "C"+datetime.now().strftime("%Y%m%d%H%M%S%f")
  log_df = log_df.append({'Prompt': prompt, 'Model Output': model_output, 'Role': role, 'Time': datetime.now(), 'Model': model, 'Version': version, 'Callid': callid}, ignore_index=True) 
  log_df.to_csv(log_file, index=False)

  return callid

api_calls = 0
retry_count = 0

def get_model_output_single(prompt, role = 'user'):
  global api_calls
  global retry_count

  api_calls += 1

  messages=[
      # # adding the system message deterioates the model performance. SO-BL f1score becomes 0.72 from 0.77 with the system message
      # {"role": 'system', "content": 'You will respond as sentiment detection model where your detected sentiments will strictly be either positive, negative or neutral.'},
      {"role": role, "content": prompt},
  ]

  model = "gpt-3.5-turbo-0301"



  completion = openai.ChatCompletion.create(
    model= model,
    messages=messages
  )
  # print(completion)

  # reset the retry count after a successful API call
  retry_count = 0

  model_output = completion.choices[0].message

  callid = write_log(prompt, model_output["content"], role, 'openai-chat', model)
 
  # sleep for 10 ms to avoid the API call limit
  time.sleep(0.01)

  
  return model_output["content"], callid

def get_model_output_chat(messages):
  global api_calls
  global retry_count

  api_calls += 1

  # messages=[
  #     # # adding the system message deterioates the model performance. SO-BL f1score becomes 0.72 from 0.77 with the system message
  #     # {"role": 'system', "content": 'You will respond as sentiment detection model where your detected sentiments will strictly be either positive, negative or neutral.'},
  #     {"role": role, "content": prompt},
  # ]

  model = "gpt-3.5-turbo-0301"



  completion = openai.ChatCompletion.create(
    model= model,
    messages=messages
  )
  # print(completion)

  # reset the retry count after a successful API call
  retry_count = 0

  model_output = completion.choices[0].message

  # find out content from all the messages and concatenate them
  prompt = ''
  for message in messages:
    prompt += message['role']+": "+ message['content'] + '\n'


  callid = write_log(prompt, model_output["content"], 'conversation', 'openai-chat', model)
 
  # sleep for 10 ms to avoid the API call limit
  time.sleep(0.01)

  
  return model_output["content"], callid


df = pd.read_csv(input_file)
# df_output = pd.read_csv(output_file)

def prepare_context(question_title, question_body, accepted_answer):
    # prepare the context
    context = 'Question Title: '+question_title+'\n'
    context = 'Question: '+question_title+'\n'+question_body+'\n'
    context += 'Answer:\n'+accepted_answer+'\n'
    
    return context


# loop over the rows
for index, row in df.iterrows():
    if api_calls > MAX_API_CALLS:
        print('API calls exceeded '+str(MAX_API_CALLS)+'. Breaking the loop')
        break
    
    # get ID
    # id = row['id']
    # approach = row['question_approach']

    # skip it if the row is already processed
    if not pd.isnull(row['new_answer']):
        continue

    context = row['new_context']
    if pd.isnull(context):
      question_title = row['so_question_title']
      question_body = row['so_question_body']
      accepted_answer = row['so_answer']
      context = prepare_context(question_title, question_body, accepted_answer)

    limit = ' Answer in 50 words strictly based on the following conversation:\n'

    prompt = row['new_question'] + limit


    # print approach and prompt
    print(row['uid'])

    # get the model output
    try:
     model_answer, callid = get_model_output_single(prompt+context)
    except: # TODO: this should go inside the get_model_output_* function to retry and exit.
      print('API call failed. Retrying...')
    
      # sleep for 1s to avoid the API call limit
      time.sleep(1)

      retry_count += 1
      if retry_count > MAX_RETRY:
        print('API calls failed '+str(MAX_RETRY)+' times. Breaking the loop')
        exit()

      continue
    
    # copy the row to a new row 
    row['new_answer'] = model_answer
    row['callid'] = callid

    # update the dataframe df with row
    df.loc[index] = row

    # save the dataframe
    df.to_csv(input_file, index=False)




