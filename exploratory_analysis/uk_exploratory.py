#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[7]:


# get_ipython().system(' pip install --upgrade pip setuptools wheel')


# # In[8]:


# get_ipython().system(' pip install gensim')


# # In[11]:


# get_ipython().system(' pip install language_tool_python')


# # In[12]:


# get_ipython().system(' pip install pycontractions --no-deps')


# # In[13]:


# get_ipython().system(' pip show pycontractions')


# # In[15]:


# get_ipython().system(' pip install wordcloud')


# # In[17]:


# get_ipython().system(' pip install dill')


# # In[6]:


# get_ipython().system(' java -version')


# # In[35]:


# get_ipython().system(' pip install POT')


# In[1]:


get_ipython().run_line_magic('load_ext', 'jupyternotify')


# ### Imports

# In[1]:


import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import gensim.corpora as corpora
import gensim
from gensim.models import CoherenceModel
from pycontractions import Contractions
from nltk.probability import FreqDist
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from nltk import trigrams
from nltk import bigrams
from nltk import ngrams
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import copy
import dill
import os
import pickle
import gensim.downloader as api
from gensim.models import KeyedVectors
import json


# In[4]:


nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('vader_lexicon')


#   LANGUAGE_TOOL_PYTHON REQUIRES JAVA 17

# In[2]:


java_folder = "jdk-17.0.10+7"
java_path = os.path.expanduser(f"~/{java_folder}")
os.environ["JAVA_HOME"] = java_path
os.environ["PATH"] = os.path.join(java_path, "bin") + ":" + os.environ["PATH"]


# In[3]:


plt.style.use('ggplot')


# ### Functions

# In[16]:


# cont = Contractions(api_key="glove-twitter-100")
# cont.load_models()


# In[21]:


# cont_model = api.load("glove-twitter-100")


# In[23]:


# cont_model.save("glove_twitter_100.kv")


# In[4]:


cont_model = KeyedVectors.load("glove_twitter_100.kv", mmap='r')
cont = Contractions(kv_model=cont_model)
cont.load_models()


# In[7]:


list(cont.expand_texts(["I'd like to know how I'd done that!",
                        "We're going to the zoo and I don't think I'll be home for dinner.",
                        "Theyre going to the zoo and she'll be home for dinner."], precise=True))


# In[8]:


def remove_contractions(text_list):
    return list(cont.expand_texts(text_list, precise=True))


# In[9]:


def clean(text):
    '''Clean text to make it suitable for NLP techniques'''
    text = remove_contractions([text])[0]
    text = re.sub(r'[^a-zA-Z0-9\s£%?!\.]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


# In[10]:


lemmatizer = WordNetLemmatizer()
def to_wordnet_pos(nltk_tag):
    '''Convert NLTK pos tags to those compatible with the WordNet lemmatizer'''
    if nltk_tag.startswith('J'):
        return wn.ADJ
    elif nltk_tag.startswith('V'):
        return wn.VERB
    elif nltk_tag.startswith('R'):
        return wn.ADV
    else:        
        return wn.NOUN  


# In[11]:


def lemmatize(tokens):
    '''Lemmatise tokens'''
    tagged_tokens = pos_tag(tokens)
    lemma_list = []
    for token, tag in tagged_tokens:
        if token != 'us':
            lemma_list.append(lemmatizer.lemmatize(token, to_wordnet_pos(tag)))
        else:
            lemma_list.append(lemmatizer.lemmatize(token, wn.ADJ))            
    return lemma_list


# In[12]:


stop_words = set(stopwords.words('english'))
stop_words.remove('i')
def remove_stop_words(lemmas, stopword_list=stop_words):
    '''Remove stopwords from lemmas'''     
    return [lemma for lemma in lemmas if lemma not in stopword_list and len(lemma) >= 2] 


# In[4]:


print(stop_words)


# In[2]:


path_to_debates = '../transcription/uk_general/edited_final_transcripts/' 


# In[3]:


# get lists of participants
with open('../transcription/data/debate_participants.json', 'r') as f:
    participants_data = f.read()
participants = json.loads(participants_data)
participants


# What I'll be doing for the EDA:
# 1. separate debates into sections of speech for each participating politician 
# 2. Word frequency plot of each speaker before and after pre-processing (lemmatisation, etc.)
# 3. Word count per speaker (before and after too)
# 4. Word cloud (same)
# 5. N-gram analysis (after pre-processing)
# 6. Type-token ratio
# 7. Sentiment Analysis
# 8. Topic Modelling for all speakers, then for each speaker

# format: e.g. ('deb_2010_04_29', 'separate'/'combined', 'proc'/'raw', 'grouped'/'turns', 'CLEGG')
#                 

# In[4]:


full_list = os.listdir(path_to_debates)
deb_options = []
for i in full_list:
    if '.txt' in i:
        i = i[:-4]
        deb_options.append(i)
    
deb_options = sorted(deb_options)
deb_options


# In[16]:


# separating debates into sections
raw_debates = {}
#raw_debates[deb_2024]=full debate
sep_raw_debates = {}
#seprawdeb[deb2024][person]=[sec1,sec2,sec3,...]
grouped_sep_raw = {}
comb_raw_debates = {}
#combrawdeb[deb2024]=full text without names
sep_proc_debates = {}
grouped_sep_proc = {}
comb_proc_debates = {}


# In[17]:


for deb in deb_options:
    print(deb)
    with open(path_to_debates+deb+'.txt', 'r') as file:
        deb_text = file.read()
    raw_debates[deb] = deb_text


# In[18]:


for deb in deb_options:
    print(deb)
    with open(path_to_debates+deb+'.txt', 'r') as file:
        deb_text = file.read()
    # separate deb_text according to speakers
    # get list of speakers
    speakers = participants[deb]
    # get no of questioners
    count = 0
    temp_text = deb_text
    print("accounting for all questioners")
    while True:
        if temp_text.find(f"QUESTIONER_{count}") == -1:
            break
        speakers.append(f"QUESTIONER_{count}")
        count+=1
    
    # create lists
    sep_raw_debates[deb]={speaker: [] for speaker in speakers} 
    grouped_sep_raw[deb]={speaker: "" for speaker in speakers}
    # set up combination of all speech
    combined_raw_text = []

    current_speaker = None
    current_turn = []
    line_pattern = re.compile(r'^([A-Z0-9_]+):\s*(.*)')
    all_lines = deb_text.strip().split('\n')
    print("separating transcript into turns")
    print(f"no of lines: {len(all_lines)}")
    line_count=1
    for line in all_lines:   
        print(f"line {line_count}/{len(all_lines)}")
        match = line_pattern.match(line)            
        if match:                
            speaker_name, content = match.groups()

            if current_speaker is None:
                current_speaker = speaker_name
                current_turn = [content]
                combined_raw_text.append(content)

            elif speaker_name == current_speaker:
                current_turn.append(content)
                combined_raw_text.append(content)
            elif speaker_name != current_speaker:
                # reached the end of a turn. combine all lines of speech                    
                full_turn = " ".join(current_turn).strip()
                sep_raw_debates[deb][current_speaker].append(full_turn) 
                #reset
                current_speaker = speaker_name
                current_turn = [content]
                combined_raw_text.append(content)
        else:                
            if current_speaker:
                current_turn.append(line.strip())
                combined_raw_text.append(line.strip())
        line_count+=1
        
    #reached the end of all turns. save for final speaker
    if current_speaker:
        full_turn = " ".join(current_turn).strip()
        sep_raw_debates[deb][current_speaker].append(full_turn) 
    print("Finished separating raw transcript according to speakers") 
    print("Grouping turns together")
    for speaker in speakers:
        grouped_sep_raw[deb][speaker] = " ".join(sep_raw_debates[deb][speaker]).strip()

    #-----------------------------------------------------------------------
    #save combination of all speech        
    print("Combining all raw debate speech together")
    comb_raw_debates[deb] = combined_raw_text


# In[19]:


get_ipython().run_cell_magic('notify', '', 'for deb in deb_options:\n    print(deb)\n    with open(path_to_debates+deb+\'.txt\', \'r\') as file:\n        deb_text = file.read()\n    #apply pre-processing to debates\n    speakers = participants[deb]\n    sep_proc_debates[deb]={speaker: [] for speaker in speakers}  \n    grouped_sep_proc[deb]={speaker: "" for speaker in speakers}\n    \n    speaker_count = 1\n    for speaker in speakers:\n        print(f"Speaker {speaker_count}/{len(speakers)}")\n        #clean\n        print("Cleaning text")\n        sep_proc_debates[deb][speaker] = [clean(text) for text in sep_raw_debates[deb][speaker]] \n        # tokenise and lemmatise\n        # for each chunk of text in the list\n        print("Tokenising and Lemmatising text")\n        for i in range(len(sep_proc_debates[deb][speaker])):\n            print(f"---Turn {i+1}/{len(sep_proc_debates[deb][speaker])}")\n            text = sep_proc_debates[deb][speaker][i]\n            tok_list = word_tokenize(text)\n            lem_list = lemmatize(tok_list)\n            sep_proc_debates[deb][speaker][i] = " ".join(lem_list)\n        speaker_count+=1\n    \n    print("Grouping turns together")\n    speaker_count = 1\n    for speaker in speakers:\n        print(f"Speaker {speaker_count}/{len(speakers)}")\n        grouped_sep_proc[deb][speaker] = " ".join(sep_proc_debates[deb][speaker]).strip()\n        speaker_count+=1\n        \n    #---------------------------------------------------------------------\n    #save processed combo of all speech\n    print("Combining all processed debate speech together")\n    combined_proc_text = []\n    line_count = 1\n    for line in comb_raw_debates[deb]:\n        print(f"-Line {line_count}/{len(comb_raw_debates[deb])}")\n        print("---Cleaning")\n        clean_line = clean(line)\n        print("---Tokenising")\n        tok_line = word_tokenize(clean_line)\n        print("---Lemmatising")\n        lem_line = lemmatize(tok_line)\n        combined_proc_text.append(" ".join(lem_line))\n        line_count+=1\n        \n    comb_proc_debates[deb] = combined_proc_text')


# In[ ]:


# separating debates into sections
raw_debates = {}
#raw_debates[deb_2024]=full debate
sep_raw_debates = {}
#seprawdeb[deb2024][person]=[sec1,sec2,sec3,...]
grouped_sep_raw = {}
comb_raw_debates = {}
#combrawdeb[deb2024]=full text without names
sep_proc_debates = {}
grouped_sep_proc = {}
comb_proc_debates = {}


# In[34]:


comb_proc_debates


# In[ ]:


with open('raw_debates.json', 'w', encoding='utf-8') as f:
    json.dump(raw_debates, f, indent=4)


# In[36]:


with open('sep_raw_debates.json', 'w', encoding='utf-8') as f:
    json.dump(sep_raw_debates, f, indent=4)


# In[37]:


with open('grouped_sep_raw.json', 'w', encoding='utf-8') as f:
    json.dump(grouped_sep_raw, f, indent=4)


# In[38]:


with open('sep_proc_debates.json', 'w', encoding='utf-8') as f:
    json.dump(sep_proc_debates, f, indent=4)


# In[39]:


with open('grouped_sep_proc.json', 'w', encoding='utf-8') as f:
    json.dump(grouped_sep_proc, f, indent=4)


# In[40]:


with open('comb_raw_debates.json', 'w', encoding='utf-8') as f:
    json.dump(comb_raw_debates, f, indent=4)


# In[41]:


with open('comb_proc_debates.json', 'w', encoding='utf-8') as f:
    json.dump(comb_proc_debates, f, indent=4)


# In[5]:


# with open('grouped_sep_raw.json', 'r') as f:
#     grouped_sep_raw_data = f.read()
# grouped_sep_raw = json.loads(grouped_sep_raw_data)
# with open('raw_debates.json', 'r') as f:
#     raw_debates_data = f.read()
# raw_debates=json.loads(raw_debates_data)    
# with open('sep_raw_debates.json', 'r') as f:
#     sep_raw_debates_data = f.read()
# sep_raw_debates=json.loads(sep_raw_debates_data)
# with open('sep_proc_debates.json', 'r') as f:
#     sep_proc_debates_data = f.read()
# sep_proc_debates = json.loads(sep_proc_debates_data)    
# with open('grouped_sep_proc.json', 'r') as f:
#     grouped_sep_proc_data = f.read()
# grouped_sep_proc = json.loads(grouped_sep_proc_data)
# with open('comb_raw_debates.json', 'r') as f:
#     comb_raw_debates_data = f.read()
# comb_raw_debates = json.loads(comb_raw_debates_data)
# with open('comb_proc_debates.json', 'r') as f:
#     comb_proc_debates_data = f.read()    
# comb_proc_debates = json.loads(grouped_sep_raw_data)    


# In[6]:


grouped_sep_raw


# In[11]:


def find_debate(deb, structure, processing, grouping, person=None):  
    '''
    Easily find pre-processed debate transcript text in desired format
    deb - name of debate from options
    structure - 'separate'(each speaker) / 'combined'(all speakers)
    processing - 'proc'(pre-processing applied) / 'raw'(raw content)
    grouping - 'grouped'(all speaker turns combined) / 'turns'(separated into speaker turns)
    person - name of speaker in debate
    '''
    if structure == 'separate':
        #use sep and grouped dicts
        if grouping == 'turns':
            #use sep dicts
            if processing == 'proc':
                if person != None:                    
                    if person in participants[deb]:
                        return sep_proc_debates[deb][person]
                    else:                        
                        print("Invalid 'person' param. Non-participant in debate")
                        return
                else:                    
                    return sep_proc_debates[deb]
            elif processing == 'raw':
                if person != None:
                    if person in participants[deb]:
                        return sep_raw_debates[deb][person]
                    else:
                        print("Invalid 'person' param. Non-participant in debate")
                        return
                else:                    
                    return sep_raw_debates[deb]
            else:
                print("Invalid 'processing' param. Use options given in docstring")
                return
        elif grouping == 'grouped':
            #use grouped dicts
            if processing == 'proc':
                if person != None:
                    if person in participants[deb]:
                        return grouped_sep_proc[deb][person]
                    else:
                        print("Invalid 'person' param. Non-participant in debate")
                        return
                else:                    
                    return grouped_sep_proc[deb]
            elif processing == 'raw':
                if person != None:
                    if person in participants[deb]:
                        return grouped_sep_raw[deb][person]
                    else:
                        print("Invalid 'person' param. Non-participant in debate")
                        return
                else:                    
                    return grouped_sep_raw[deb]
            else:
                print("Invalid 'processing' param. Use options given in docstring")
                return
        else:
            print("Invalid 'grouping' param. Use options given in docstring")
            return
    elif structure == 'combined':
        #use comb dicts
        if processing == 'proc':
            return comb_proc_debates[deb]
        elif processing == 'raw':
            return comb_raw_debates[deb]
        else:
            print("Invalid 'processing' param. Use options given in docstring")
            return
    else:
        print("Invalid 'structure' param. Use options given in docstring")
        return
    


# format: e.g. ('deb_2010_04_29', 'separate'/'combined', 'proc'/'raw', 'grouped'/'turns', 'CLEGG')

# In[58]:


find_debate('deb_2010_04_29', 'separate', 'raw', 'grouped', 'CLEGG')


# In[80]:


def plot_word_freq(deb, processing, rem_stopwords=True, top_n=10):    
    for speaker in participants[deb]:
        #remove stopwords        
        content = find_debate(deb, 'separate', processing, 'grouped', speaker).split()
        if processing == "proc" and rem_stopwords:
            content = remove_stop_words(content)        
        fdist = FreqDist(content)  
        fdist.plot(top_n)
        plt.title(f"{speaker} ({deb})")
        plt.show()


# In[81]:


plot_word_freq('deb_2024_06_04', 'proc')


# In[78]:


def plot_word_count(deb, processing):
    plt.style.use('ggplot')
    plt.bar([speaker for speaker in participants[deb]], [len(find_debate(deb, 'separate', processing, 'grouped', speaker).split()) for speaker in participants[deb]])  
    plt.title(f"Word Count per speaker ({deb})")
    plt.ylabel("Word Count")
    plt.xlabel("Speaker")
    plt.xticks(rotation=90)    
    plt.show()


# In[85]:


plot_word_count('deb_2024_06_04', 'raw')


# In[84]:


def plot_word_cloud(deb, processing):
    for speaker in participants[deb]:
        word_cloud = WordCloud(background_color="white").generate(find_debate(deb, 'separate', processing, 'grouped', speaker))
        plt.figure(figsize=(10,10))
        plt.imshow(word_cloud)
        plt.title(f"{speaker} ({deb})")
        plt.axis("off")
        plt.show()


# In[86]:


plot_word_cloud('deb_2024_06_04', 'raw')


# In[91]:


def plot_ngrams(deb, processing, n_gram, top_n, rem_stopwords=True):
    for speaker in participants[deb]:
        #remove stopwords        
        content = find_debate(deb, 'separate', processing, 'grouped', speaker).split()
        if processing == "proc" and rem_stopwords:
            content = remove_stop_words(content)  
        n_grams = list(ngrams(content,n_gram))
        fdist = FreqDist(n_grams)
        fdist.plot(top_n)
        plt.title(f"{speaker} ({deb})")
        plt.show()


# In[92]:


plot_ngrams('deb_2024_06_04', 'proc', 3, 10)


# In[95]:


def plot_ttr(deb, processing):
    ttr_list = []
    for speaker in participants[deb]:
        if 'QUESTIONER' not in speaker:
            distinct_words = set(find_debate(deb, 'separate', processing, 'grouped', speaker).split())
            ttr = len(distinct_words) / len(find_debate(deb, 'separate', processing, 'grouped', speaker).split())
            ttr_list.append(ttr)
            print(f"{speaker}: TTR={ttr}")
    plt.plot(participants[deb], ttr_list, marker='o')
    plt.title(f"Type-Token Ratio (TTR) per speaker ({deb})")
    plt.xlabel("Speaker")
    plt.ylabel("TTR")
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.show()


# In[96]:


plot_ttr('deb_2024_06_04', 'raw')


# In[14]:


def plot_ttr2(deb, processing):
    ttr_list = []
    speakerss = [x for x in participants[deb] if 'QUESTIONER' not in x]
    for speaker in speakerss:        
        distinct_words = set(find_debate(deb, 'separate', processing, 'grouped', speaker).split())
        ttr = len(distinct_words) / len(find_debate(deb, 'separate', processing, 'grouped', speaker).split())
        ttr_list.append(ttr)
        print(f"{speaker}: TTR={ttr}")
    plt.plot(speakerss, ttr_list, marker='o')
    plt.title(f"Type-Token Ratio (TTR) per speaker ({deb})")
    plt.xlabel("Speaker")
    plt.ylabel("TTR")
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.show()


# In[15]:


plot_ttr2('deb_2024_06_04', 'raw')


# In[101]:


sentia = SentimentIntensityAnalyzer()
def plot_sentiment_scores(deb):
    for speaker in participants[deb]:
        #create sentiment score list
        sent_score_list = [sentia.polarity_scores(turn)['compound'] for turn in find_debate(deb, 'separate', 'proc', 'turns', speaker)]
        plt.plot(range(len(sent_score_list)), sent_score_list)
        plt.ylim(-1,1)
        plt.title(f"{speaker}'s sentiment scores over time ({deb})", pad=20)
        plt.xlabel("Turn")
        plt.ylabel("Compound Sentiment Score")
        plt.grid(True)
        plt.show()
    


# In[102]:


plot_sentiment_scores('deb_2024_06_04')


# In[103]:


def individual_lda_topics(deb):
    all_turns = []
    print("Individual Topics")
    for speaker in participants[deb]:
        # remove stopwords
        turn_list = find_debate(deb, 'separate', 'proc', 'turns', speaker)
        new_turn_list = []
        for turn in turn_list:
            tok_list = turn.split()
            tok_list = remove_stop_words(tok_list)
            new_turn_list.append(tok_list)
        all_turns += new_turn_list
        
        # model topics
        print(f"\n{speaker} topics:")
        docs = new_turn_list
        debate_token_dictionary = corpora.Dictionary(docs)
        debate_corpus = [debate_token_dictionary.doc2bow(tokens) for tokens in docs]
        lda_model = gensim.models.ldamodel.LdaModel(corpus=debate_corpus,
                                                    id2word=debate_token_dictionary,
                                                    num_topics=5,
                                                    random_state=42,                                                                                                
                                                    passes=10,                                                
                                                    per_word_topics=True)                
        
        # word cloud of topics
        for topic_id, topic in enumerate(lda_model.print_topics(num_words=7)):
            print(f"Topic {topic_id}: {topic[1]} ")
            topic_words = ""
            for weighted_word in topic[1].split(" + "):
                weight, word = weighted_word.split("*")
                word = word[1:-1]
                topic_words += word+' '                
            wordcloud = WordCloud(background_color="white", random_state=42).generate(topic_words)
            plt.figure(figsize=(10,5))
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.title(f"Topic {topic_id}")
            plt.show()
            print("")
        # coherence score of topics
        coherence_model_lda = CoherenceModel(model=lda_model,
                                        texts=docs,
                                        dictionary=debate_token_dictionary,
                                        coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print(f'\nCoherence Score: ', coherence_lda)
    
    print("All Topics Collectively")
        
        


# In[104]:


individual_lda_topics('deb_2024_06_04')


# Maybe include the QUESTIONER's content into the debates so that LDA topic modelling can be carried out on the whole of the debate

# In[ ]:




