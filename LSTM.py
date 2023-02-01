#!/usr/bin/env python
# coding: utf-8

# - you can download dataset from kaggle : https://www.kaggle.com/datasets/arunasivapragasam/donors-choose

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# ## Model-1 

# In[ ]:


# import all the libraries
#make sure that you import your libraries from tf.keras and not just keras
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[ ]:


#read the csv file
import pandas as pd
df = pd.read_csv(r'/content/drive/MyDrive/preprocessed_data.csv')


# In[ ]:


df.head()


# In[ ]:


df['project_is_approved'].value_counts()


# In[ ]:


print(df.shape)


# In[ ]:


y = df['project_is_approved']
x = df.drop('project_is_approved',axis = 1)


# In[ ]:


# perform stratified train test split on the dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,stratify = y,random_state = 0)


# In[ ]:


print(x_train.shape)
print(x_test.shape)


# In[ ]:


y_train.shape


# ## 1.1 Text Vectorization

# In[ ]:


#essay column
tokenizer = Tokenizer(oov_token = "<oov>")
tokenizer.fit_on_texts(x_train['essay'])
vocab_size = len(tokenizer.word_index) + 1
sequences_train = tokenizer.texts_to_sequences(x_train['essay'])
sequences_test = tokenizer.texts_to_sequences(x_test['essay'])
max_length = max([len(x) for x in sequences_train])
train_padded_docs = pad_sequences(sequences_train,maxlen = max_length,padding = 'post')
test_padded_docs = pad_sequences(sequences_test,maxlen = max_length,padding = 'post')


# In[ ]:


print("max document length",max_length)
print("vocab size",vocab_size)
print(train_padded_docs.shape,test_padded_docs.shape)


# In[ ]:



import numpy as np
embeddings_index = dict()
f = open('/content/drive/MyDrive/glove.6B.300d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))


# In[ ]:


embedding_matrix = np.zeros((vocab_size, 300))
for word, i in tokenizer.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector


# ## 1.2 Categorical feature Vectorization

# In[ ]:


#school_state
tokenizer_school = Tokenizer(oov_token = "<oov>")
tokenizer_school.fit_on_texts(x_train['school_state'])
vocab_size_school = len(tokenizer_school.word_index) + 1
sequences_train_school = tokenizer_school.texts_to_sequences(x_train['school_state'])
sequences_test_school = tokenizer_school.texts_to_sequences(x_test['school_state'])
max_length_school = max([len(x) for x in sequences_train_school])
train_padded_docs_school = pad_sequences(sequences_train_school,maxlen = max_length_school,padding = 'post')
test_padded_docs_school = pad_sequences(sequences_test_school,maxlen = max_length_school,padding = 'post')


# In[ ]:


print("max document length",max_length_school)
print("vocab size",vocab_size_school)
print(train_padded_docs_school.shape,test_padded_docs_school.shape)


# In[ ]:


#project_grade_category
tokenizer_grade = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n',oov_token = "<oov>")
tokenizer_grade.fit_on_texts(x_train['project_grade_category'])
vocab_size_grade = len(tokenizer_grade.word_index) + 1
sequences_train_grade = tokenizer_grade.texts_to_sequences(x_train['project_grade_category'])
sequences_test_grade = tokenizer_grade.texts_to_sequences(x_test['project_grade_category'])
max_length_grade = max([len(x) for x in sequences_train_grade])
train_padded_docs_grade = pad_sequences(sequences_train_grade,maxlen = max_length_grade,padding = 'post')
test_padded_docs_grade = pad_sequences(sequences_test_grade,maxlen = max_length_grade,padding = 'post')
print("max document length",max_length_grade)
print("vocab size",vocab_size_grade)
print(train_padded_docs_grade.shape,test_padded_docs_grade.shape)


# In[ ]:


#clean categories
tokenizer_clean = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n',oov_token = "<oov>")
tokenizer_clean.fit_on_texts(x_train['clean_categories'])
vocab_size_clean = len(tokenizer_clean.word_index) + 1
sequences_train_clean = tokenizer_clean.texts_to_sequences(x_train['clean_categories'])
sequences_test_clean = tokenizer_clean.texts_to_sequences(x_test['clean_categories'])
max_length_clean = max([len(x) for x in sequences_train_clean])
train_padded_docs_clean = pad_sequences(sequences_train_clean,maxlen = max_length_clean,padding = 'post')
test_padded_docs_clean = pad_sequences(sequences_test_clean,maxlen = max_length_clean,padding = 'post')
print("max document length",max_length_clean)
print("vocab size",vocab_size_clean)
print(train_padded_docs_clean.shape,test_padded_docs_clean.shape)


# In[ ]:


#clean_subcategories
tokenizer_clean_sub = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n',oov_token = "<oov>")
tokenizer_clean_sub.fit_on_texts(x_train['clean_subcategories'])
vocab_size_clean_sub = len(tokenizer_clean_sub.word_index) + 1
sequences_train_clean_sub = tokenizer_clean_sub.texts_to_sequences(x_train['clean_subcategories'])
sequences_test_clean_sub = tokenizer_clean_sub.texts_to_sequences(x_test['clean_subcategories'])
max_length_clean_sub = max([len(x) for x in sequences_train_clean_sub])
train_padded_docs_clean_sub = pad_sequences(sequences_train_clean_sub,maxlen = max_length_clean_sub,padding = 'post')
test_padded_docs_clean_sub = pad_sequences(sequences_test_clean_sub,maxlen = max_length_clean_sub,padding = 'post')
print("max document length",max_length_clean_sub)
print("vocab size",vocab_size_clean_sub)
print(train_padded_docs_clean_sub.shape,test_padded_docs_clean_sub.shape)


# In[ ]:


#teacher_prefix
tokenizer_teacher = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n',oov_token = "<oov>")
tokenizer_teacher.fit_on_texts(x_train['teacher_prefix'])
vocab_size_teacher = len(tokenizer_teacher.word_index) + 1
sequences_train_teacher = tokenizer_teacher.texts_to_sequences(x_train['teacher_prefix'])
sequences_test_teacher = tokenizer_teacher.texts_to_sequences(x_test['teacher_prefix'])
max_length_teacher = max([len(x) for x in sequences_train_teacher])
train_padded_docs_teacher = pad_sequences(sequences_train_teacher,maxlen = max_length_teacher,padding = 'post')
test_padded_docs_teacher = pad_sequences(sequences_test_teacher,maxlen = max_length_teacher,padding = 'post')
print("max document length",max_length_teacher)
print("vocab size",vocab_size_teacher)
print(train_padded_docs_teacher.shape,test_padded_docs_teacher.shape)


# ## 1.3 Numerical feature Vectorization

# In[ ]:


# you have to standardise the numerical columns
# stack both the numerical features
#after numerical feature vectorization you will have numerical_data_train and numerical_data_test


# In[ ]:


from sklearn.preprocessing import StandardScaler
#price
s = StandardScaler()
s.fit(x_train['price'].values.reshape(-1,1))
std_price_train = s.transform(x_train['price'].values.reshape(-1,1))
std_price_test = s.transform(x_test['price'].values.reshape(-1,1))


# In[ ]:


std_price_train.shape


# In[ ]:


#teacher_number_of_previously_posted_projects
t = StandardScaler()
t.fit(x_train['teacher_number_of_previously_posted_projects'].values.reshape(-1,1))
std_teacher_train = t.transform(x_train['teacher_number_of_previously_posted_projects'].values.reshape(-1,1))
std_teacher_test = t.transform(x_test['teacher_number_of_previously_posted_projects'].values.reshape(-1,1))


# In[ ]:


std_teacher_train.shape


# In[ ]:


#concatenating both numerical features
x_train_numerical = np.hstack((std_price_train,std_teacher_train))
x_test_numerical = np.hstack((std_price_test,std_teacher_test))


# In[ ]:


x_train_numerical.shape


# ## 1.4 Defining the model

# In[ ]:


#essay column functional API
from tensorflow.keras.layers import Embedding,Flatten,Dense
text = Input(shape = (max_length,))
essay_embed = Embedding(vocab_size,300,weights = [embedding_matrix],input_length = max_length,trainable = False)(text)
essay_lstm = LSTM(units = 128,return_sequences = True)(essay_embed)
essay_out = Flatten()(essay_lstm)


# In[ ]:


#school state
school = Input(shape=(max_length_school,))
school_embed = Embedding(vocab_size_school,300,input_length = max_length_school)(school)
school_out = Flatten()(school_embed)


# In[ ]:


#project_grade_category
pg = Input(shape = (max_length_grade,))
grade_embed = Embedding(vocab_size_grade,300,input_length = max_length_grade)(pg)
grade_out = Flatten()(grade_embed)


# In[ ]:


#clean categories
clean = Input(shape = (max_length_clean,))
clean_embed = Embedding(vocab_size_clean,300,input_length = max_length_clean)(clean)
clean_out = Flatten()(clean_embed)


# In[ ]:


#clean subcategories
clean_sub = Input(shape = (max_length_clean_sub),)
clean_sub_embed = Embedding(vocab_size_clean_sub,300,input_length = max_length_clean_sub)(clean_sub)
clean_sub_out = Flatten()(clean_sub_embed)


# In[ ]:


#teacher_prefix
teacher = Input(shape=(max_length_teacher),)
teacher_embed = Embedding(vocab_size_teacher,300,input_length = max_length_teacher)(teacher)
teacher_out = Flatten()(teacher_embed)


# In[ ]:


#numerical features
import tensorflow as td
num = Input(shape = (2,))
dense_1 = Dense(32,activation = "relu",kernel_initializer=tf.keras.initializers.he_normal())(num)


# ## 1.5 Compiling and fititng your model

# In[ ]:


get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().system(' rm -rf ./logs/')


# In[ ]:


import os
import datetime
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Concatenate,Dropout
log_dir = os.path.join("logs",'fits', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1,write_graph=True)
filepath="model_save/weights-{epoch:02d}-{val_auc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss',  verbose=1, save_best_only=True, mode='auto')


# In[ ]:


#concatenate
concat = Concatenate()([essay_out,school_out,grade_out,clean_out,clean_sub_out,teacher_out,dense_1])
dense_layer_1 = Dense(128,activation = "relu",kernel_initializer=tf.keras.initializers.he_normal())(concat)
drop = Dropout(0.5)(dense_layer_1)
dense_layer_2 = Dense(64,activation = "relu",kernel_initializer=tf.keras.initializers.he_normal())(drop)
drop_1 = Dropout(0.5)(dense_layer_2)
dense_layer_3 = Dense(32,activation = "relu",kernel_initializer=tf.keras.initializers.he_normal())(drop_1)
drop_2 = Dropout(0.5)(dense_layer_3)
dense_layer_4 = Dense(16,activation = "relu",kernel_initializer=tf.keras.initializers.he_normal())(drop_2)
out = Dense(2,activation = "softmax")(dense_layer_4)
model_1 = tf.keras.Model(inputs=[text,school,pg,clean,clean_sub,teacher,num],outputs = out)


# In[ ]:


model_1.summary()


# In[ ]:


import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.keras.utils import to_categorical

#if y_true is of only one class
#https://stackoverflow.com/questions/45139163/roc-auc-score-only-one-class-present-in-y-true
def roc_auc(y_true, y_pred):
    if len(np.unique(y_true[:,1])) == 1:
        return 0.5
    else:
        return roc_auc_score(y_true, y_pred, average='weighted')

def auc(y_true, y_pred):
    return tf.py_function(roc_auc, (y_true, y_pred), tf.double)


# In[ ]:


model_1.compile(loss = "categorical_crossentropy",optimizer = "adam",metrics =[auc])
model_1.fit([train_padded_docs,train_padded_docs_school,train_padded_docs_grade,train_padded_docs_clean,train_padded_docs_clean_sub,train_padded_docs_teacher,x_train_numerical],to_categorical(y_train),epochs = 5,validation_split = 0.2,batch_size = 32,callbacks = [tensorboard_callback,checkpoint])


# In[ ]:


print("test AUC for model_1 is {}".format(roc_auc_score(to_categorical(y_test),model_1.predict([test_padded_docs,test_padded_docs_school,test_padded_docs_grade,test_padded_docs_clean,test_padded_docs_clean_sub,test_padded_docs_teacher,x_test_numerical]))))


# In[ ]:


tf.keras.utils.plot_model(model_1, to_file='model_1.png', show_shapes=True, show_layer_names=True)


# In[ ]:


get_ipython().run_line_magic('tensorboard', '--logdir logs')


# # <font color='black'> Model-2 </font>

# Use the same model as above but for 'input_seq_total_text_data' give only some words in the sentance not all the words. Filter the words as below. 

# In[ ]:


#fitting tf_idf vectorizer on essay
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df = 3)
x_train_vect = vectorizer.fit_transform(x_train['essay'])
idf = vectorizer.idf_
feature_idf = dict(zip(vectorizer.get_feature_names(), idf))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
sns.distplot(idf,kde = False)


# As we can see from the plot lowest value is 2 and highest value is 11

# In[ ]:


#excluding the words which are less than 2 and greater than 11
def remove_words(text):
  return ' '.join([word for word in text.split() if feature_idf.get(word) is None or (feature_idf.get(word) and (feature_idf.get(word)>=2 and feature_idf.get(word)<=11))])


# In[ ]:


#creating new essay column
x_train['essay_mod'] = x_train['essay'].apply(remove_words)


# In[ ]:


x_train['essay_mod']


# In[ ]:


#test
x_test['essay_mod'] = x_test['essay'].apply(remove_words)


# In[ ]:


#essay_mod
tokenizer = Tokenizer(oov_token = "<oov>")
tokenizer.fit_on_texts(x_train['essay_mod'])
vocab_size = len(tokenizer.word_index) + 1
sequences_train = tokenizer.texts_to_sequences(x_train['essay_mod'])
sequences_test = tokenizer.texts_to_sequences(x_test['essay_mod'])
max_length = max([len(x) for x in sequences_train])
train_padded_docs = pad_sequences(sequences_train,maxlen = max_length,padding = 'post')
test_padded_docs = pad_sequences(sequences_test,maxlen = max_length,padding = 'post')


# In[ ]:


print("max document length",max_length)
print("vocab size",vocab_size)
print(train_padded_docs.shape,test_padded_docs.shape)


# In[ ]:


import numpy as np
embeddings_index = dict()
f = open('/content/drive/MyDrive/glove.6B.300d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))


# In[ ]:


embedding_matrix = np.zeros((vocab_size, 300))
for word, i in tokenizer.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector


# In[ ]:


#essay_mod 
from tensorflow.keras.layers import Embedding,Flatten,Dense
text = Input(shape = (max_length,))
essay_embed = Embedding(vocab_size,300,weights = [embedding_matrix],input_length = max_length,trainable = False)(text)
essay_lstm = LSTM(units = 128,return_sequences = True)(essay_embed)
essay_out = Flatten()(essay_lstm)


# In[ ]:


get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().system(' rm -rf ./logs/')


# In[ ]:


import os
import datetime
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Concatenate,Dropout
log_dir = os.path.join("logs",'fits', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1,write_graph=True)
filepath="model_save/weights-{epoch:02d}-{val_auc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss',  verbose=1, save_best_only=True, mode='auto')


# In[ ]:


concat = Concatenate()([essay_out,school_out,grade_out,clean_out,clean_sub_out,teacher_out,dense_1])
dense_layer_1 = Dense(128,activation = "relu",kernel_initializer=tf.keras.initializers.he_normal())(concat)
drop = Dropout(0.5)(dense_layer_1)
dense_layer_2 = Dense(64,activation = "relu",kernel_initializer=tf.keras.initializers.he_normal())(drop)
drop_1 = Dropout(0.5)(dense_layer_2)
dense_layer_3 = Dense(32,activation = "relu",kernel_initializer=tf.keras.initializers.he_normal())(drop_1)
drop_2 = Dropout(0.5)(dense_layer_3)
dense_layer_4 = Dense(16,activation = "relu",kernel_initializer=tf.keras.initializers.he_normal())(drop_2)
out = Dense(2,activation = "softmax")(dense_layer_4)
model_2 = tf.keras.Model(inputs=[text,school,pg,clean,clean_sub,teacher,num],outputs = out)


# In[ ]:


model_2.summary()


# In[ ]:


model_2.compile(loss = "categorical_crossentropy",optimizer = "adam",metrics =[auc])
model_2.fit([train_padded_docs,train_padded_docs_school,train_padded_docs_grade,train_padded_docs_clean,train_padded_docs_clean_sub,train_padded_docs_teacher,x_train_numerical],to_categorical(y_train),epochs = 5,validation_split = 0.2,batch_size = 32,callbacks = [tensorboard_callback,checkpoint])


# In[ ]:


print("test AUC for model_2 is {}".format(roc_auc_score(to_categorical(y_test),model_2.predict([test_padded_docs,test_padded_docs_school,test_padded_docs_grade,test_padded_docs_clean,test_padded_docs_clean_sub,test_padded_docs_teacher,x_test_numerical]))))


# In[ ]:


tf.keras.utils.plot_model(model_2, to_file='model_2.png', show_shapes=True, show_layer_names=True)


# In[ ]:


get_ipython().run_line_magic('tensorboard', '--logdir logs')


# # <font color='black'> Model-3 </font>

# In[ ]:


#one hot encoding of categorical features
from sklearn.preprocessing import OneHotEncoder


# In[ ]:


#school_state
o = OneHotEncoder()
o.fit(x_train['school_state'].values.reshape(-1,1))
ohe_schhol_train = o.transform(x_train['school_state'].values.reshape(-1,1))
ohe_school_test = o.transform(x_test['school_state'].values.reshape(-1,1))


# In[ ]:


#project_grade_categories
o = OneHotEncoder()
o.fit(x_train['project_grade_category'].values.reshape(-1,1))
ohe_project_train = o.transform(x_train['project_grade_category'].values.reshape(-1,1))
ohe_project_test = o.transform(x_test['project_grade_category'].values.reshape(-1,1))


# In[ ]:


#clean_categories
o = OneHotEncoder()
o.fit(x_train['clean_categories'].values.reshape(-1,1))
ohe_clean_train = o.transform(x_train['clean_categories'].values.reshape(-1,1))
ohe_clean_test = o.transform(x_test['clean_categories'].values.reshape(-1,1))


# In[ ]:


#clean subcategories
o = OneHotEncoder(handle_unknown='ignore')
o.fit(x_train['clean_subcategories'].values.reshape(-1,1))
ohe_clean_sub_train = o.transform(x_train['clean_subcategories'].values.reshape(-1,1))
ohe_clean_sub_test = o.transform(x_test['clean_subcategories'].values.reshape(-1,1))


# In[ ]:


#teacher_prefix
o = OneHotEncoder()
o.fit(x_train['teacher_prefix'].values.reshape(-1,1))
ohe_teacher_pre_train = o.transform(x_train['teacher_prefix'].values.reshape(-1,1))
ohe_teacher_pre_test = o.transform(x_test['teacher_prefix'].values.reshape(-1,1))


# In[ ]:


#combining all features other than text 
from scipy.sparse import hstack
x_train_not_text = np.hstack((ohe_schhol_train.todense(),ohe_project_train.todense(),ohe_clean_train.todense(),ohe_clean_sub_train.todense(),ohe_teacher_pre_train.todense(),x_train_numerical))
x_test_not_text = np.hstack((ohe_school_test.todense(),ohe_project_test.todense(),ohe_clean_test.todense(),ohe_clean_sub_test.todense(),ohe_teacher_pre_test.todense(),x_test_numerical))


# In[ ]:


x_train_not_text.shape


# In[ ]:


get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().system(' rm -rf ./logs/')


# In[ ]:


#conv1d layers
inp = Input(shape = (x_train_not_text.shape[1],1))
conv_1 = tf.keras.layers.Conv1D(128,3,activation = "relu")(inp)
conv_2 = tf.keras.layers.Conv1D(64,3,activation = "relu")(conv_1)

flatter = Flatten()(conv_2)


# In[ ]:


#model_3
from tensorflow.keras.layers import Dense,Concatenate,Dropout,BatchNormalization
concat = Concatenate()([essay_out,flatter])
layer_1 = Dense(128,activation = "relu",kernel_initializer=tf.keras.initializers.he_normal())(concat)
drop_1 = Dropout(0.5)(layer_1)
layer_2 = Dense(64,activation = "relu",kernel_initializer=tf.keras.initializers.he_normal())(drop_1)
drop_2 = Dropout(0.5)(layer_2)
layer_3 = Dense(32,activation = "relu",kernel_initializer=tf.keras.initializers.he_normal())(drop_2)
drop_3 = Dropout(0.5)(layer_3)
out = Dense(2,activation = "softmax")(drop_3)
model_3 = tf.keras.Model(inputs=[text,inp],outputs = out)


# In[ ]:


model_3.summary()


# In[ ]:


import os
import datetime
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Concatenate,Dropout
log_dir = os.path.join("logs",'fits', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1,write_graph=True)
filepath="model_save/weights-{epoch:02d}-{val_auc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss',  verbose=1, save_best_only=True, mode='auto')


# In[ ]:


model_3.compile(loss = "categorical_crossentropy",optimizer = "adam",metrics =[auc])
model_3.fit([train_padded_docs,x_train_not_text],to_categorical(y_train),epochs = 3,validation_split = 0.3,batch_size = 32,callbacks = [tensorboard_callback,checkpoint])


# In[ ]:


tf.keras.utils.plot_model(model_3, to_file='model_3.png', show_shapes=True, show_layer_names=True)


# In[ ]:


get_ipython().run_line_magic('tensorboard', '--logdir logs')


# In[1]:


#summarising the model
from prettytable import PrettyTable
  

myTable = PrettyTable(["Models", "Loss","AUC"])
  

myTable.add_row(["Model_1", "0.39","0.729"])
myTable.add_row(["Model_2", "0.37","0.742"])
myTable.add_row(["Model_3", "0.21","0.84"])


# In[2]:


print(myTable)


# # Observations
# 
# 

# ## Model_1
# 
# 1.Since we are using Pretrained glove vectors for embedding of text data we have few parameters for training
# 
# 2.Total we have 20966642.out of this only 6167342 are only for trainable.
# 
# 3.After training for 5 epochs we get a loss of 0.39 and AUC is 0.72. if we train for more epochs will get good AUC.

# ##Model_2
# 
# 1.Compared to model_1 we are using same architecture but from given text data we are removing the less frequent words and high frequent words in text corpus which are not useful. based on idf value of the words.
# 
# 2.We train tfidf vectorizer on the text data and take idf values of the words for elimination.
# 
# 3.So we get good AUC value in 5th epoch only i.e 0.75

# ##Model_3
# 
# 1.Stacking all the data other than text data into one and passing text data and stacked data as input to the model.
# 
# 2.It gives more AUC compared to model_1 and model_2. after 3 epochs we get the AUC value of 0.8
