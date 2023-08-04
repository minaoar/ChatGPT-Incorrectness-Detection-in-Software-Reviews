
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

# get current date string in the format of YYYYMMDD
date_string = datetime.now().strftime("%Y%m%d")
log_file = log_dir + 'log_'+date_string+'.csv'


data_dir = '' # set the data directory

input_file  = data_dir + 'so_nltk_spacy_data.v5.csv'
output_file = data_dir + 'so_nltk_spacy_data_output.csv'


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


# def get_model_output_api(prompt):
#   response = openai.Completion.create(
#     model="text-davinci-003",
#     prompt=prompt,
#     temperature=0.7,
#     max_tokens=MAX_TOKEN_SIZE*4,
#     top_p=1,
#     frequency_penalty=0,
#     presence_penalty=0
#   )



#   model_output = response["choices"][0]["text"]
#   write_log(prompt, model_output, "", 'openai-api', 'text-davinci-003')

#   # sleep for 10 ms to avoid the API call limit
#   time.sleep(0.01)

#   return model_output

def prepare_context(question_title, question_body, accepted_answer):
    # prepare the context
    context = 'Question Title: '+question_title+'\n'
    context = 'Question: '+question_title+'\n'+question_body+'\n'
    context += 'Answer:\n'+accepted_answer+'\n'
    
    return context

def prepare_prompt_aspect_detection(library, aspect, aspect_question):
   
    prompt = 'Answer strictly with "yes" or "no" to the query\n'
    prompt += '"Does this following conversation (question, answer) discuss about '+aspect_question+' of the library '+library+'?"\n'
    # prompt += 'Must not add any extra remarks.\n'
    # prompt += 'Provide answer strictly using "yes", "no", "no answer given". Must not add any extra remarks.\n'

    return prompt

def prepare_prompt_aspect_list(library, aspect, aspect_question):

    aspect_list = " aspects (Active Maintenance, Documentation, Ease of use, Feature, Performance, Security, Stability)"
    feature_list = "features (only from dependency parsing, Entity linking, Lemmatization, NER, POS Tagging, Rule-based matching, Sentence segmentation, Text classification, tokenization)"

    if aspect == 'Feature':
        aspect_list = feature_list
   
    prompt = 'Provide comma separated list of the '+aspect_list+' which are discussed in this following conversation (question, answer).\n'

    return prompt

def prepare_prompt_aspect_sentiment(library, aspect, aspect_question):
     
      prompt = 'Provide in JSON the sentiment ({"positive":x, "negative":y, "neutral":z) probability for the '+aspect_question+' of the library '+library+' which is discussed in this following conversation (question, answer).\n'
      return prompt
    
def prepare_prompt_ask_query_on_aspect(library, aspect, aspect_question):
      aspect_query_map = {'Active Maintenance':'How actively the library '+library+' is maintained',
        'Documentation':'How is the documentation of the library '+ library,
        'Ease of use':'How easy it is to use the library '+ library,
        'Feature':'How well does this library '+library+' support '+aspect_question,
        'Performance':'How is the performance of the library '+ library,
        'Security':'How is the security of the library '+ library,
        'Stability':'How stable or well tested is the library '+ library
      }
      
      prompt = "Respond in less than 200 words "+aspect_query_map[aspect] + " strictly based on the following conversation (question, answer).\n"
      return prompt

def prepare_prompt_ask_aspect_score(library, aspect, aspect_question):
    aspect_query_map_intro = {'Active Maintenance':'How actively the library '+library+' is maintained',
        'Documentation':'How is the documentation of the library '+ library,
        'Ease of use':'How easy it is to use the library '+ library,
        'Feature':'How well does this library '+library+' support '+aspect_question,
        'Performance':'How is the performance of the library '+ library,
        'Security':'How is the security of the library '+ library,
        'Stability':'How stable or well tested is the library '+ library
      }
    
    aspect_query_map = {'Active Maintenance':'(higher score means actively maintained)', 
      'Documentation':'(higher score means well documented)', 
      'Ease of use':'(higher score means easier to use)', 
      'Feature':'(higher score means better this feature)', 
      'Performance':'(higher score means better performance)', 
      'Security':'(higher score means more secure)', 
      'Stability':'(higher score means more stable/bug-free)', 
    }

    intro = aspect_query_map_intro[aspect]
    prompt = "Comment within 20 words " + intro + '? Provide score out of 5 '+aspect_query_map[aspect] + '. Respond in JSON {"score":x, "comment":y} strictly based on the following conversation (question, answer).\n'
    return prompt

def prepare_prompt_ask_deeper_question(library, aspect, aspect_question):
   
   #return 'Provide explanation for the answer. Each point separately with keyword and the explanation in 50 words. Respond in JSON [{"keyword":x, "explanation":y}] strictly based on the previous conversation (question, answer).\n'
   return 'Provide explanation for the answer. Each reason separately with title in 4 words and the explanation in 50 words. Respond strictly in JSON [{"title":x, "explanation":y}] without any additional remarks strictly based on the previous conversation (question, answer).\n'

def prepare_prompt_challenge_internal(challenge_keyword):
    prompt = 'generate a question that starts with "'+challenge_keyword+'" to challenge each reason explanation. Respond strictly in JSON [{"title":x, "challenge_question":y}] without ant additional remarks.\n'

    return prompt

def prepare_prompt_challenge_with_how(library, aspect, aspect_question):
    
    #initial_prompt = 'generate a question that starts with "how" to challenge each explanation answered in the previous response. Respond in JSON [{"keyword":x, "challenge_question":y}].\n'
    # initial_prompt = 'generate a question that starts with "how" to challenge each explanation answered. Respond in JSON [{"keyword":x, "challenge_question":y}].\n'
    # initial_prompt = 'generate a question that starts with "how" to challenge each reason explanation. Respond strictly in JSON [{"title":x, "challenge_question":y}] without ant additional remarks.\n'

    # model_output, callid_ignored = get_model_output_single(initial_prompt)

    initial_prompt = prepare_prompt_challenge_internal('how')

    return initial_prompt


def prepare_prompt_challenge_with_really(library, aspect, aspect_question):
    
    initial_prompt = prepare_prompt_challenge_internal('really')

    return initial_prompt


def prepare_prompt_challenge_with_why(library, aspect, aspect_question):
    
    initial_prompt = prepare_prompt_challenge_internal('why')

    return initial_prompt

def prepare_prompt_ask_binary_question(library, aspect, aspect_question):
   return 'Is the following statement correct based on the given conversation (question, answer)? Respond strictly with "yes" or "no".\n STATEMENT: '
   

def prepare_prompt(approach, library, aspect, aspect_question):
   # create a approach vs function mapping
    approach_function_map = {
        'Detect Aspect/Feature': prepare_prompt_aspect_detection,
        'List Aspect': prepare_prompt_aspect_list,
        'Detect Aspect Sentiment': prepare_prompt_aspect_sentiment,
        'Ask Query on Aspect': prepare_prompt_ask_query_on_aspect,
        'Ask Query on Aspect-how': prepare_prompt_ask_query_on_aspect,
        'Ask Query on Aspect-really': prepare_prompt_ask_query_on_aspect,
        'Ask Query on Aspect-why': prepare_prompt_ask_query_on_aspect,
        'Ask Aspect Score': prepare_prompt_ask_aspect_score,
        'Ask deeper question': prepare_prompt_ask_deeper_question, 
        'Ask deeper question-really': prepare_prompt_ask_deeper_question, 
        'Ask deeper question-why': prepare_prompt_ask_deeper_question, 
        'Challenge with how': prepare_prompt_challenge_with_how, 
        'Challenge with really': prepare_prompt_challenge_with_really,
        'Challenge with why': prepare_prompt_challenge_with_why,
        'Binary': prepare_prompt_ask_binary_question,
    }

    # get the function
    function = approach_function_map[approach]
    # print(function)

    # call the function
    prompt = function(library, aspect, aspect_question)
    # print(prompt)
    return prompt



df = pd.read_csv(input_file)
df_output = pd.read_csv(output_file)

approach = 'Detect Aspect/Feature'
approach = 'List Aspect'
approach = 'Detect Aspect Sentiment'
approach = 'Ask Query on Aspect' 
approach = 'Ask Aspect Score'
approach = 'Binary'


# loop over the rows
for index, row in df.iterrows():
    if api_calls > MAX_API_CALLS:
        print('API calls exceeded '+str(MAX_API_CALLS)+'. Breaking the loop')
        break
    
    chat_messages = []

    target_approaches = [approach]
    # target_approaches = ['Ask Query on Aspect-how', 'Ask deeper question', 'Challenge with how']
    # target_approaches = ['Challenge with really']
    # target_approaches = ['Challenge with why']

    query_on_aspect_prompt = ''
    query_on_aspect_answer = ''
    deeper_question_prompt = ''
    deeper_question_answer = ''
 
    for approach in target_approaches:
      # get ID
      id = row['id']

      # get the list of approaches for this id
      approaches = df_output[df_output['id'] == id]['question_approach'].tolist()
     
      # if the approach is in the list, then skip
      if approach in approaches:
        continue

      question_title = row['so_question_title']
      question_body = row['so_question_body']
      accepted_answer = row['so_answer']
      library = row['library']
      aspect = row['aspect']
      aspect_question = row['aspect_question']

      context = prepare_context(question_title, question_body, accepted_answer)

      prompt = prepare_prompt(approach, library, aspect, aspect_question)
      

      # print approach and prompt
      print('Approach: ', approach)
      print('Prompt: ', prompt)

      keywords = [approach]
      prompts = [prompt]

      ask_query_on_aspect_approach = (approach == 'Ask Query on Aspect-how' or approach == 'Ask Query on Aspect-really' or approach == 'Ask Query on Aspect-why')
      ask_deeper_question_approach = (approach == 'Ask deeper question' or approach == 'Ask deeper question-really' or approach == 'Ask deeper question-why')
      challenge_approach = (approach == 'Challenge with how' or approach == 'Challenge with really' or approach == 'Challenge with why')
      
      if (approach == 'Challenge with really' or approach == 'Challenge with why'):
        # retrieve the row by matching  question_approach = 'Ask Query on Aspect-how' and id = id
        query_on_aspect_row = df_output[(df_output['id'] == id) & (df_output['question_approach'] == 'Ask Query on Aspect-how')]

        query_on_aspect_prompt = prompt+context
        query_on_aspect_answer = query_on_aspect_row['answer'].tolist()[0]

        # update the row with the current approach
        query_on_aspect_row['question_approach'] = approach

        # add the row to the dataframe
        df_output = df_output.append(query_on_aspect_row, ignore_index=True)


        deeper_question_row = df_output[(df_output['id'] == id) & (df_output['question_approach'] == 'Ask deeper question')]

        deeper_question_prompt = deeper_question_row['question'].tolist()[0]
        deeper_question_answer = deeper_question_row['answer'].tolist()[0]

        # update the row with the current approach
        deeper_question_row['question_approach'] = approach

        # add the row to the dataframe
        df_output = df_output.append(deeper_question_row, ignore_index=True)


        # save the dataframe
        df_output.to_csv(output_file, index=False)


      if challenge_approach == True:
          messages = [{"role": "system", "content": "You are a helpful assistant."},
                        {"role": 'user', "content": query_on_aspect_prompt},
                        {"role": 'assistant', "content": query_on_aspect_answer},
                        {"role": 'user', "content": deeper_question_prompt},
                        {"role": 'assistant', "content": deeper_question_answer},
                        {"role": 'user', "content": prompt}]
          model_answer_original, callid = get_model_output_chat(messages)
          
          # prompt = [{"keyword":"Cats are natural hunters because they have sharp claws and keen senses.","challenge_question":"How do sharp claws and keen senses make cats natural hunters?"},{"keyword":"The human brain weighs about 3 pounds.","challenge_question":"How heavy is the human brain?"},{"keyword":"The capital of France is Paris.","challenge_question":"How do you spell the name of the capital of France?"}]

          # parse the prompt in JSON
          # clean the model_answer. Remove anything before '[' and after ']'
          model_answer = model_answer_original[model_answer_original.find('['):model_answer_original.find(']')+1]
          prompt_json = json.loads(model_answer)

          # list of keywords
          keywords = [item['title'] for item in prompt_json]

          # list of prompts
          limit = ' Answer in 50 words strictly based on the conversation (question, answer).'
          prompts = [item['challenge_question']+limit for item in prompt_json]

          # keep only the first 5 keywords and prompts
          keywords = keywords[:5]
          prompts = prompts[:5]

      if approach == 'Binary':
          # get the answer from the row with same id and question_approach = 'Ask deeper question' 
          # from the output file
          deeper_question_row = df_output[(df_output['id'] == id) & (df_output['question_approach'] == 'Ask deeper question')]
          deeper_question_answer = deeper_question_row['answer'].tolist()[0]

          answer_json = deeper_question_answer[deeper_question_answer.find('['):deeper_question_answer.find(']')+1]
          prompt_json = json.loads(answer_json)

             # list of keywords
          keywords = [item['title'] for item in prompt_json]

          # list of prompts
          limit = ' Answer in 50 words strictly based on the conversation (question, answer).'
          prompts = [prompt+item['explanation']+'\n' for item in prompt_json]

          # keep only the first 5 keywords and prompts
          keywords = keywords[:5]
          prompts = prompts[:5]



      # loop over the keywords and prompts
      for keyword, prompt in zip(keywords, prompts):
        print('Keyword: ', keyword)
        print('Prompt: ', prompt)


        # get the model output
        while retry_count < MAX_RETRY:
          try:
            
              
            if ask_deeper_question_approach == True:
              messages = [{"role": "system", "content": "You are a helpful assistant."},
                          {"role": 'user', "content": query_on_aspect_prompt},
                          {"role": 'assistant', "content": query_on_aspect_answer},
                          {"role": 'user', "content": prompt}
                          ]
              model_answer, callid = get_model_output_chat(messages)

            elif challenge_approach == True:
              messages = [{"role": "system", "content": "You are a helpful assistant."},
                          {"role": 'user', "content": query_on_aspect_prompt},
                          {"role": 'assistant', "content": query_on_aspect_answer},
                          # {"role": 'user', "content": deeper_question_prompt},
                          # {"role": 'assistant', "content": deeper_question_answer},
                          {"role": 'user', "content": prompt}]
              model_answer, callid = get_model_output_chat(messages)

            else:
              model_answer, callid = get_model_output_single(prompt+context)

            break

          except: # TODO: this should go inside the get_model_output_* function to retry and exit.
            print('API call failed. Retrying...')
          
            # sleep for 1s to avoid the API call limit
            time.sleep(1)

            retry_count += 1
            if retry_count >= MAX_RETRY:
              print('API calls failed '+str(MAX_RETRY)+' times. Breaking the loop')
              exit()
        
        # copy the row to a new row 
        row['answer'] = model_answer
        row['question_approach'] = approach
        row['question_keyword'] = keyword
        row['question'] = prompt
        row['callid'] = callid

        # add the row to the dataframe
        df_output = df_output.append(row, ignore_index=True)
        
        # print the results
        # print('Question Title: ', question_title)
        #print('Context: ', context)
        print('Answer: ', row['answer'])
    
        # save the dataframe
        df_output.to_csv(output_file, index=False)
        # print('Saved the dataframe to ', output_file)

        
        #target_approaches = ['Ask Query on Aspect-how', 'Ask deeper question', 'Challenge with how']
        if ask_query_on_aspect_approach == True:
           query_on_aspect_prompt = prompt+context
           query_on_aspect_answer = model_answer

        if ask_deeper_question_approach == True:
            deeper_question_prompt = prompt
            deeper_question_answer = model_answer

    



