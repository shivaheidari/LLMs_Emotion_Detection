from transformers import AutoTokenizer
#training parameter
learning_rate = 0.001
num_classes = 13
num_epochs = 10
input_size = 224
batch_size = 32
model_name = 'roberta-base'
#dataset parameter
num_workers = 4
dir_data= 'Data/tweet_emotions.csv'
train_split = 0.7
val_split = 0.15
test_split = 0.15

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, clean_up_tokenization_spaces=True) 
max_len_token = 128
#compute parameter
accelerator = 'cpu'
devices = 1
percision = 16
