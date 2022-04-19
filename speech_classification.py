# -*- coding: cp949 -*-
from glob import glob
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import torchaudio


from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.metrics import f1_score

from transformers import Wav2Vec2Processor, Wav2Vec2Model
import argparse
from torch.nn.utils.rnn import pad_sequence

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default=19)
parser.add_argument('--dataset_dir', type=str, default='./KEMDy19/')
parser.add_argument('--ckpt', type=str, default='0', help='checkpoint')
parser.add_argument('--num_fold', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-5, help='learning_rate')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
parser.add_argument('--gpus', type=str, default='0', help='gpu numbers')
parser.add_argument('--epochs', type=int, default=5, help='epochs')
parser.add_argument('--max_seq_len', type=int, default=5, help='max sequence length of speech')
parser.add_argument('--num_labels', type=int, default=7, help='num_labels')
parser.add_argument('--regress', type=int, default=1, help='regression (0-1)')
parser.add_argument('--seed', type=int, default=1234, help='seed')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#seed
if args.seed is not None:
    import random
    import numpy as np
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#loader
class SpectrogramDataset(Dataset):
    def __init__(self, path_list, max_seq_len, tokenizer):
        super(SpectrogramDataset, self).__init__()
        # [0] = wav, [1] = txt, [2] = emo_label, [3] = valence, [4] = arousal
        self.wav_list = path_list[0]
        self.txt_list = path_list[1]
        self.label_list = path_list[2]
        self.valence_list = path_list[3]
        self.arousal_list = path_list[4]
        self.size = len(self.wav_list)
        self.max_seq_len = max_seq_len

    def __getitem__(self, index):
        
        try:        
            wav_data, based_sr = torchaudio.load(self.wav_list[index])            
        except:
            print('data {} has problem, can not reading '.format(self.wav_list[index]))
            return None
        
        if wav_data.size(-1) > self.max_seq_len:
            wav_data = wav_data[:, :self.max_seq_len]
        
        #input_dict = tokenizer(self.txt_list[index], padding = 'max_length', max_length = args.max_text_len, return_tensors = 'pt', return_attention_mask = False)
        #output_text = torch.cat([input_dict['input_ids'], input_dict['token_type_ids'], ~(input_dict['input_ids']==0)], dim=0)
        
        return wav_data, self.label_list[index]
        
    def __len__(self):
        return self.size


def _collate_fn(batch):    
    batches = list(filter(lambda x: x is not None, batch))
    batch = sorted(batches, key=lambda sample: sample[0].size(1), reverse=True)
    
    seq_lengths = [s[0].size(1) for s in batch]
    max_seq_size = max(seq_lengths)
    
    seqs = torch.zeros(len(batch), max_seq_size)
    targets = torch.zeros(len(batch), 7).to(torch.long)
    
    for x in range(len(batch)):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        seqs[x].narrow(0, 0, seq_length).copy_(tensor.squeeze())
        targets[x].narrow(0, 0, len(target)).copy_(target)
    
    return seqs, targets

class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


#preprocessing
wav_dir = os.path.join(args.dataset_dir, 'wav')
label_df = sorted(glob(os.path.join(args.dataset_dir, 'annotation', '*')))

all_wav = []
all_txt = []
all_emotion = []
all_valence = []
all_arousal = []
bad_data_for_20 = 0
bad_data_for_19 = 0

emotion_dict = {"angry": 0, "disqust": 1, "fear": 2, "happy": 3, "neutral": 4, "sad": 5, "surprise": 6}

for d in range(len(label_df)):    
    _, file_tmp = os.path.split(label_df[d])
    file_num = file_tmp.split('_')[0]
    
    usecols_element = [3, 4, 5, 6] if args.dataset is 20 else [9, 10, 11, 12]
    df = pd.read_csv(label_df[d], usecols=usecols_element, skiprows=[0])
    
    val_list = df.values.tolist()
    
    for i in range(len(val_list)):
    
        if args.dataset == 20:
            first_folder = val_list[i][0].split('_')[0]
            if file_num != first_folder and file_num == 'Sess17':
                first_folder = 'Sess17'
                first_folder_tmp = first_folder[:-2]+'ion'+first_folder[-2:]
                direc = os.path.join(wav_dir, first_folder_tmp)
                    
                wav_file = os.path.join(direc, first_folder+val_list[i][0][6:]+'.wav')
                txt_file = os.path.join(direc, first_folder+val_list[i][0][6:]+'.txt')
                   
            else:
                first_folder_tmp = first_folder[:-2]+'ion'+first_folder[-2:]
                direc = os.path.join(wav_dir, first_folder_tmp)
                wav_file = os.path.join(direc, val_list[i][0]+'.wav')
                txt_file = os.path.join(direc, val_list[i][0]+'.txt')
        
            if os.path.isfile(wav_file) and os.path.isfile(txt_file):
            
                emotion = val_list[i][1]
                val = val_list[i][2]
                aro = val_list[i][-1]
                
                all_wav.append(wav_file)
                with open(txt_file, 'r', encoding='cp949') as f:
                    infor = f.readline()
                    infor = infor.split('\n')[0]
                all_txt.append(infor)
                all_emotion.append(emotion)
                all_valence.append(val)
                all_arousal.append(aro)
            else:
                bad_data_for_20 += 1
    
    
        elif args.dataset == 19:
            first_folder = val_list[i][0].split('_')[0]
            first_folder_tmp = first_folder[:-2]+'ion'+first_folder[-2:]
            direc = os.path.join(wav_dir, first_folder_tmp)
            speaker_num = val_list[i][0][:-5]

            wav_file = os.path.join(direc, speaker_num, val_list[i][0]+'.wav')
            txt_file = os.path.join(direc, speaker_num, val_list[i][0]+'.txt')
            
            if os.path.isfile(wav_file) and os.path.isfile(txt_file):
                
                emotion = val_list[i][1]
                val = val_list[i][2]
                aro = val_list[i][-1]
                all_wav.append(wav_file)
                
                with open(txt_file, 'r') as f:
                    infor = f.readline()
                    infor = infor.split('\n')[0]
                all_txt.append(infor)
                all_emotion.append(emotion)
                all_valence.append(val)
                all_arousal.append(aro)
            else:
                bad_data_for_19 +=1

print('bad for 20', bad_data_for_20)
print('bad for 19', bad_data_for_19)


#making label as multi-label classification
make_label = torch.FloatTensor(len(all_emotion), 7).random_(1)
for k in range(len(all_emotion)):
    data_tmp = all_emotion[k]
    if len(data_tmp.split(';')) == 1:
        make_label[k][emotion_dict.get(data_tmp)] = 1.
        
    else:
        for j in range(len(data_tmp.split(';'))):
            make_label[k][emotion_dict.get(data_tmp.split(';')[j])] = 1.
            
#print(len(all_wav), len(all_txt), len(make_label), len(all_valence), len(all_arousal))


## cut as k-fold
if args.num_fold == 1:    
    train_wav = all_wav[int(len(all_wav)*0.2):]
    train_txt = all_txt[int(len(all_wav)*0.2):]
    train_emo_label = make_label[int(len(all_wav)*0.2):]
    train_valence_label = all_valence[int(len(all_wav)*0.2):]
    train_arousal_label = all_arousal[int(len(all_wav)*0.2):]
    
    test_wav = all_wav[:int(len(all_wav)*0.2)]
    test_txt = all_txt[:int(len(all_wav)*0.2)]
    test_emo_label = make_label[:int(len(all_wav)*0.2)]
    test_valence_label = all_valence[:int(len(all_wav)*0.2)]
    test_arousal_label = all_arousal[:int(len(all_wav)*0.2)]
    
    
elif args.num_fold == 2:
    train_wav = all_wav[:int(len(all_wav)*0.2)] + all_wav[int(len(all_wav)*0.4):]
    train_txt = all_txt[:int(len(all_wav)*0.2)] + all_txt[int(len(all_wav)*0.4):]
    train_emo_label = torch.cat((make_label[:int(len(all_wav)*0.2)],make_label[int(len(all_wav)*0.4):]))
    train_valence_label = all_valence[:int(len(all_wav)*0.2)] + all_valence[int(len(all_wav)*0.4):]
    train_arousal_label = all_arousal[:int(len(all_wav)*0.2)] + all_arousal[int(len(all_wav)*0.4):]
    
    test_wav = all_wav[int(len(all_wav)*0.2):int(len(all_wav)*0.4)]
    test_txt = all_txt[int(len(all_wav)*0.2):int(len(all_wav)*0.4)]
    test_emo_label = make_label[int(len(all_wav)*0.2):int(len(all_wav)*0.4)]
    test_valence_label = all_valence[int(len(all_wav)*0.2):int(len(all_wav)*0.4)]
    test_arousal_label = all_arousal[int(len(all_wav)*0.2):int(len(all_wav)*0.4)]
    
elif args.num_fold == 3:
    train_wav = all_wav[:int(len(all_wav)*0.4)] + all_wav[int(len(all_wav)*0.6):]
    train_txt = all_txt[:int(len(all_wav)*0.4)] + all_txt[int(len(all_wav)*0.6):]
    train_emo_label = torch.cat((make_label[:int(len(all_wav)*0.4)],make_label[int(len(all_wav)*0.6):]))
    train_valence_label = all_valence[:int(len(all_wav)*0.4)] + all_valence[int(len(all_wav)*0.6):]
    train_arousal_label = all_arousal[:int(len(all_wav)*0.4)] + all_arousal[int(len(all_wav)*0.6):]
    
    test_wav = all_wav[int(len(all_wav)*0.4):int(len(all_wav)*0.6)]
    test_txt = all_txt[int(len(all_wav)*0.4):int(len(all_wav)*0.6)]
    test_emo_label = make_label[int(len(all_wav)*0.4):int(len(all_wav)*0.6)]
    test_valence_label = all_valence[int(len(all_wav)*0.4):int(len(all_wav)*0.6)]
    test_arousal_label = all_arousal[int(len(all_wav)*0.4):int(len(all_wav)*0.6)]

elif args.num_fold == 4:
    train_wav = all_wav[:int(len(all_wav)*0.6)] + all_wav[int(len(all_wav)*0.8):]
    train_txt = all_txt[:int(len(all_wav)*0.6)] + all_txt[int(len(all_wav)*0.8):]
    train_emo_label = torch.cat((make_label[:int(len(all_wav)*0.6)],make_label[int(len(all_wav)*0.8):]))
    train_valence_label = all_valence[:int(len(all_wav)*0.6)] + all_valence[int(len(all_wav)*0.8):]
    train_arousal_label = all_arousal[:int(len(all_wav)*0.6)] + all_arousal[int(len(all_wav)*0.8):]
    
    test_wav = all_wav[int(len(all_wav)*0.6):int(len(all_wav)*0.8)]
    test_txt = all_txt[int(len(all_wav)*0.6):int(len(all_wav)*0.8)]
    test_emo_label = make_label[int(len(all_wav)*0.6):int(len(all_wav)*0.8)]
    test_valence_label = all_valence[int(len(all_wav)*0.6):int(len(all_wav)*0.8)]
    test_arousal_label = all_arousal[int(len(all_wav)*0.6):int(len(all_wav)*0.8)]

elif args.num_fold == 5:
    train_wav = all_wav[:int(len(all_wav)*0.8)]
    train_txt = all_txt[:int(len(all_wav)*0.8)]
    train_emo_label = make_label[:int(len(all_wav)*0.8)]
    train_valence_label = all_valence[:int(len(all_wav)*0.8)]
    train_arousal_label = all_arousal[:int(len(all_wav)*0.8)]
    
    test_wav = all_wav[int(len(all_wav)*0.8):]
    test_txt = all_txt[int(len(all_wav)*0.8):]
    test_emo_label = make_label[int(len(all_wav)*0.8):]
    test_valence_label = all_valence[int(len(all_wav)*0.8):]
    test_arousal_label = all_arousal[int(len(all_wav)*0.8):]
train_fold = (train_wav, train_txt, train_emo_label, train_valence_label, train_arousal_label)
test_fold = (test_wav, test_txt, test_emo_label, test_valence_label, test_arousal_label)
print('len of train_fold {} wav {} emo {}'.format(len(train_fold), len(train_wav), len(train_emo_label)))

print('fold = {}, training wav {} txt {} emo {} val {} aro {}'.format(args.num_fold, len(train_wav), len(train_txt), len(train_emo_label), len(train_valence_label), len(train_arousal_label)))
print('test wav {} txt {} emo {} val {} aro {}'.format(args.num_fold, len(test_wav), len(test_txt), len(test_emo_label), len(test_valence_label), len(test_arousal_label)))


class wav2vec_classifier(nn.Module):
    def __init__(self, extractor, num_labels, dropout_prob=0.1):
        super(wav2vec_classifier, self).__init__()

        self.extractor = extractor
        self.dropout = nn.Dropout(dropout_prob)
        self.nu_labels = num_labels
        self.classifier = nn.Linear(512, num_labels)
        #self.softmax = F.softmax()

    def forward(self, wav):
        extracted_wav = self.extractor(wav)
        
        #last_hidden_states = extracted_wav.last_hidden_state
        last_hidden_states = extracted_wav.extract_features
        last_hidden_states = self.dropout(last_hidden_states)
        output = self.classifier(last_hidden_states)
                
        return F.softmax(output[:, -1], dim=-1)


extractor = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
ser_model = wav2vec_classifier(extractor, args.num_labels)
ser_model.cuda()
######################################################

#data_loader
train_dataset = SpectrogramDataset(path_list=train_fold, max_seq_len=int(args.max_seq_len*16000))
test_dataset = SpectrogramDataset(path_list=test_fold, max_seq_len=int(args.max_seq_len*16000))
train_sampler = RandomSampler(train_dataset)

train_dataloader = AudioDataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=8, pin_memory=True, sampler=train_sampler)
test_dataloader = AudioDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
print('len of train {} test {}'.format(len(train_dataloader), len(test_dataloader)))
optimizer = AdamW(ser_model.parameters(), lr = args.lr,  eps = 1e-8)

criterion = nn.MultiLabelSoftMarginLoss()
for epoch_i in range(args.epochs):

    #train
    ser_model.train()
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.epochs))
    print('Training...')

    total_loss = 0
    train_acc_sum = 0
    train_loss = []
    for step, (data, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        if step % 100 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
        
        speech = data.cuda()
        labels = labels.cuda()        

        outputs = ser_model(speech)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        train_loss.append(total_loss/(step+1))

        targets = labels.detach().cpu().numpy()
        preds = outputs.detach().cpu().numpy() > 0.5
        train_acc = np.equal(targets, preds).sum()/7
        train_acc_sum += train_acc

        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)

    print(f'  Average training loss: {avg_train_loss:.2f}')
    print(f'  Train Accuracy: {100 * train_acc_sum / len(train_dataloader.dataset):.4f}')

    #validation
    with torch.no_grad():
        ser_model.eval()
        print('Running evaluation...')

        val_acc_sum = 0
        targets_list = []
        preds_list = []
        for data, labels in test_dataloader:

            speech = data.cuda()
            labels = labels.cuda()
            
            outputs = ser_model(speech)
            loss = criterion(outputs, labels)
            
            targets = labels.detach().cpu().numpy()
            preds = outputs.detach().cpu().numpy() > 0.5
            val_acc = np.equal(targets, preds).sum()/7
            val_acc_sum += val_acc
            targets_list.append(targets)
            preds_list.append(preds)


        targets_list = np.concatenate(targets_list, axis = 0)
        preds_list = np.concatenate(preds_list, axis = 0)
        f1_scores = f1_score(targets_list, preds_list, average="micro") * 100.0
        total_acc = (preds_list == targets_list).mean() * 100.0

    print(f'  Evaluation Accuracy: {100 * val_acc_sum / len(test_dataloader.dataset):.4f}')
    print(f'  f1_score: {f1_scores:.4f}')
    print(f'  total_acc: {total_acc:.4f}')
