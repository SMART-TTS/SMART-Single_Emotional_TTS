# Audio
num_mels = 80
n_fft = 2048
sr = 22050 
preemphasis = 0.97
hop_length = 256
win_length = 1024
n_mels = 80 # Number of Mel banks to generate
power = 1.2 # Exponent for amplifying the predicted magnitude
min_level_db = -100
ref_level_db = 20

n_layers = 3
n_heads = 4
text_hidden_size = 256
embedding_size = 512
history_hidden_size = 256
hidden_size = 256
max_db = 100
ref_db = 20
num_emo = 5
emo_emb = 64

ref_filter_size1 = 128
ref_filter_size2 = 256
ref_filter_size3 = 128
ref_kernel_size = 5 
ref_gru_width = 64
style_size = 4 

n_iter = 60
# power = 1.5
outputs_per_step = 1

epochs = 10000
lr = 0.001
save_step = 2000
val_step = 2000
image_step = 500
batch_size = 4 
n_gpus = 1 
bce_pos_weight = 100 

cleaners='english_cleaners'
accum=5
restore_step = 0 
restore_step_post = 0
data_path = '/home/bjchoi/DB/KOR/'
train_data_csv = 'metadata_jka_train.csv'
val_data_csv = 'metadata_jka_val.csv'
test_data_csv = 'metadata_jka_test.csv'
checkpoint_path = './checkpoints'
log_directory = 'transformer-tts-phone'
sample_path = './samples'
preprocess_path = './preprocessed'

stop_epoch = 2000
