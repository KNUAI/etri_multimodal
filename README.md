# etri_multimodal
## 1.코드 설명
python: 3.8.12
numpy: 1.19.5
pandas: 1.3.4
torch: 1.10.0
torchaudio: 0.10.0
transformers: 4.12.2
sklearn: 1.0.1
scipy: 1.7.1
argparse: 1.1

## 2.코드 실행방식에 대한 설명
text_model 기반 감정(종류) 학습/평가 모델 실행
```
python text_classification.py --batch_size 32 --lr 1e-5 --num_fold 1
```

text_model 기반 감정(정도) 학습/평가 모델 실행
```
python text_regression.py --batch_size 32 --lr 1e-5 --num_fold 1
```

text_model 기반 감정(종류) 학습/평가 모델 실행
```
python text_total.py --batch_size 32 --lr 1e-5 --num_fold 1
```
