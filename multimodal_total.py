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
from transformers import AutoTokenizer, AutoModel, AdamW
from sklearn.metrics import f1_score

from transformers import Wav2Vec2Processor, Wav2Vec2Model
import argparse
from audtorch.metrics.functional import concordance_cc
from torch.nn.utils.rnn import pad_sequence
from scipy.stats import pearsonr


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default=19)
parser.add_argument('--dataset_dir', type=str, default='./KEMDy19/')
parser.add_argument('--ckpt', type=str, default='0', help='checkpoint')
parser.add_argument('--num_fold', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-5, help='learning_rate')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
parser.add_argument('--gpus', type=str, default='0', help='gpu numbers')
parser.add_argument('--epochs', type=int, default=5, help='epochs')
parser.add_argument('--max_text_len', type=int, default=256, help='max sequence length of speech')
parser.add_argument('--max_seq_len', type=int, default=5, help='max sequence length of speech')
parser.add_argument('--num_labels', type=int, default=7, help='num_labels')
parser.add_argument('--regress', type=int, default=1, help='0-1')
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
        #print('wav size {} max_seq_len {}'.format(wav_data.size(-1), self.max_seq_len))
        
        if wav_data.size(-1) > self.max_seq_len:
            wav_data = wav_data[:, :self.max_seq_len]
        
        input_dict = tokenizer(self.txt_list[index], padding = 'max_length', max_length = args.max_text_len, return_tensors = 'pt', return_attention_mask = False)
        output_text = torch.cat([input_dict['input_ids'], input_dict['token_type_ids'], ~(input_dict['input_ids']==0)], dim=0)
        
        return wav_data, input_dict['input_ids'], input_dict['token_type_ids'], ~(input_dict['input_ids']==0), self.label_list[index], self.valence_list[index], self.arousal_list[index]
        
    def __len__(self):
        return self.size


def _collate_fn(batch):    
    batches = list(filter(lambda x: x is not None, batch))
    batch = sorted(batches, key=lambda sample: sample[0].size(1), reverse=True)
    
    seq_lengths = [s[0].size(1) for s in batch]
    max_seq_size = max(seq_lengths)
    
    seqs = torch.zeros(len(batch), max_seq_size)
    input_ids = torch.zeros(len(batch), args.max_text_len).to(torch.long)
    token_ids = torch.zeros(len(batch), args.max_text_len).to(torch.long)
    mask_ids = torch.zeros(len(batch), args.max_text_len).to(torch.long)
    targets = torch.zeros(len(batch), 7).to(torch.long)
    valences = torch.zeros(len(batch), 1)
    arousals = torch.zeros(len(batch), 1)        
        
        
    for x in range(len(batch)):
        sample = batch[x]
        tensor = sample[0]
        input_id = sample[1]
        token_id = sample[2]
        mask_id = sample[3]
        target = sample[4]
        
        valence = torch.FloatTensor([sample[5]])
        arousal = torch.FloatTensor([sample[6]])
        seq_length = tensor.size(1)
        seqs[x].narrow(0, 0, seq_length).copy_(tensor.squeeze())
        input_ids[x].narrow(0, 0, args.max_text_len).copy_(input_id.squeeze())
        token_ids[x].narrow(0, 0, args.max_text_len).copy_(token_id.squeeze())
        mask_ids[x].narrow(0, 0, args.max_text_len).copy_(mask_id.squeeze())
        targets[x].narrow(0, 0, len(target)).copy_(target)
        valences[x].narrow(0, 0, len(valence)).copy_(valence)
        arousals[x].narrow(0, 0, len(arousal)).copy_(arousal)
    
    return seqs, input_ids, token_ids, mask_ids, targets, valences, arousals

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


class multimodal_classifier(nn.Module):
    def __init__(self, extractor, language_model, emo_labels, regress_labels, dropout_prob=0.1):
        super(multimodal_classifier, self).__init__()

        self.extractor = extractor
        self.lm = AutoModel.from_pretrained(language_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.wav_layer = nn.Linear(512, 768)
        self.output_layer = nn.Linear(768, emo_labels)
        self.valence_layer = nn.Linear(768, regress_labels)
        self.arousal_layer = nn.Linear(768, regress_labels)                

    def forward(self, wav, input_ids, attention_mask, token_type_ids):
        extracted_text = self.lm(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = extracted_text[1]
        pooled_output = self.dropout(pooled_output)
        
        extracted_wav = self.extractor(wav)                
        last_hidden_states = extracted_wav.extract_features
        last_hidden_states = self.dropout(last_hidden_states) #B, Seq, 512
        
        output_wav = self.wav_layer(last_hidden_states)
        
        output_add = self.output_layer(output_wav[:, -1] + pooled_output)
        output_val = self.valence_layer(output_wav[:, -1] + pooled_output)
        output_aro = self.arousal_layer(output_wav[:, -1] + pooled_output)        
        
        return F.softmax(output_add, dim=-1), output_val, output_aro
                

tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
extractor = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
multimodal_ser_model = multimodal_classifier(extractor, 'klue/bert-base', args.num_labels, args.regress)
multimodal_ser_model.cuda()
######################################################

#data_loader
train_dataset = SpectrogramDataset(path_list=train_fold, max_seq_len=int(args.max_seq_len*16000), tokenizer=tokenizer)
test_dataset = SpectrogramDataset(path_list=test_fold, max_seq_len=int(args.max_seq_len*16000), tokenizer=tokenizer)
train_sampler = RandomSampler(train_dataset)

train_dataloader = AudioDataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=8, pin_memory=True, sampler=train_sampler)
test_dataloader = AudioDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
print('len of train {} test {}'.format(len(train_dataloader), len(test_dataloader)))
optimizer = AdamW(multimodal_ser_model.parameters(), lr = args.lr,  eps = 1e-8)

criterion1 = nn.MultiLabelSoftMarginLoss()
criterion2 = nn.MSELoss()

for epoch_i in range(args.epochs):
    #train
    
    multimodal_ser_model.train()
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.epochs))
    print('Training...')

    total_loss = 0
    train_loss = []
    for step, (speech, text, segs, mask, label, val, aro) in enumerate(train_dataloader):
        optimizer.zero_grad()
        
        speech = speech.cuda()
        text = text.cuda()
        segs = segs.cuda()
        mask = mask.cuda()
        labels = label.cuda()
        vals = val.cuda()
        aros = aro.cuda()

        outputs, val_outputs, aro_outputs = multimodal_ser_model(speech, text, segs, mask)

        loss_emo = criterion1(outputs, labels)
        loss_val = criterion2(val_outputs.squeeze(dim=-1), vals.squeeze(dim=-1))
        loss_aro = criterion2(aro_outputs.squeeze(dim=-1), aros.squeeze(dim=-1))        
        loss = loss_emo + loss_val + loss_aro
        
        total_loss += loss.item()
        train_loss.append(total_loss/(step+1))        

        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)

    print(f'  Average training loss: {avg_train_loss:.2f}')   

    #validation
    with torch.no_grad():
        multimodal_ser_model.eval()
        print('Running evaluation...')

        val_acc_sum = 0
        targets_list = []
        preds_list = []
        
        val_targets_list = []
        aro_targets_list = []
        
        val_preds_list = []
        aro_preds_list = []
        for speech, text, segs, mask, label, val, aro in test_dataloader:

            speech = speech.cuda()
            text = text.cuda()
            segs = segs.cuda()
            mask = mask.cuda()
            labels = label.cuda()
            vals = val.cuda()
            aros = aro.cuda()
            
            outputs, val_outputs, aro_outputs = multimodal_ser_model(speech, text, segs, mask)
            
            
            targets = labels.detach().cpu().numpy()
            preds = outputs.detach().cpu().numpy() > 0.5
            val_acc = np.equal(targets, preds).sum()/7
            val_acc_sum += val_acc
            targets_list.append(targets)
            preds_list.append(preds)
            
            
            val_targets = vals.detach().cpu().numpy()
            val_targets = np.squeeze(val_targets)
            val_preds = val_outputs.detach().cpu().numpy()
            val_preds = np.squeeze(val_preds)
            val_targets_list.append(val_targets)
            val_preds_list.append(val_preds)
            
            aro_targets = aros.detach().cpu().numpy()
            aro_targets = np.squeeze(aro_targets)
            aro_preds = aro_outputs.detach().cpu().numpy()
            aro_preds = np.squeeze(aro_preds)
            aro_targets_list.append(aro_targets)
            aro_preds_list.append(aro_preds)
            
            

        
        targets_list = np.concatenate(targets_list, axis = 0)
        preds_list = np.concatenate(preds_list, axis = 0)
        f1_scores = f1_score(targets_list, preds_list, average="micro") * 100.0
        total_acc = (preds_list == targets_list).mean() * 100.0
        
        
        val_targets_list = np.concatenate(val_targets_list, axis = 0)
        val_preds_list = np.concatenate(val_preds_list, axis = 0)        
        val_pearsonrs = pearsonr(val_preds_list, val_targets_list)[0] * 100.0
        
        t_val_targets_list = torch.tensor(val_targets_list)
        t_val_preds_list = torch.tensor(val_preds_list)
        val_concordance_cc = concordance_cc(t_val_preds_list, t_val_targets_list)[0] * 100.0

        aro_targets_list = np.concatenate(aro_targets_list, axis = 0)
        aro_preds_list = np.concatenate(aro_preds_list, axis = 0)
        aro_pearsonrs = pearsonr(aro_preds_list, aro_targets_list)[0] * 100.0
        
        t_aro_targets_list = torch.tensor(aro_targets_list)
        t_aro_preds_list = torch.tensor(aro_preds_list)
        aro_concordance_cc = concordance_cc(t_aro_preds_list, t_aro_targets_list)[0] * 100.0
        

    print(f'  f1_score: {f1_scores:.4f}')
    print(f'  total_acc: {total_acc:.4f}')
    
    print(f'  val pearsonr: {val_pearsonrs:.4f}')
    print(f'  aro pearsonr: {aro_pearsonrs:.4f}')
    print(f'  val concordance_cc: {val_concordance_cc:.4f}')
    print(f'  aro concordance_cc: {aro_concordance_cc:.4f}')
    
