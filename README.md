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
사용데이터: KEMDy19
```
--dataset 19 --dataset_dir '/your_dir/dataset/KEMDy19/
```

사용데이터: KEMDy20
```
--dataset 20 --dataset_dir '/your_dir/dataset/KEMDy20/
```

5-fold train/valid = 8:2 적용
첫번째 fold 적용
```
--num_fold 1
```
두번째 fold 적용
```
--num_fold 2
```
세번째 fold 적용
```
--num_fold 3
```
네번째 fold 적용
```
--num_fold 4
```
다섯번째 fold 적용
```
--num_fold 5
```

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

speech_model 기반 감정(종류) 학습/평가 모델 실행
```
python speech_classification.py --batch_size 32 --lr 1e-5 --num_fold 1
```

speech_model 기반 감정(정도) 학습/평가 모델 실행
```
python speech_regression.py --batch_size 32 --lr 1e-5 --num_fold 1
```

speech_model 기반 감정(종류) 학습/평가 모델 실행
```
python speech_total.py --batch_size 32 --lr 1e-5 --num_fold 1
```

multimodal_model 기반 감정(종류) 학습/평가 모델 실행
```
python multimodal_classification.py --batch_size 32 --lr 1e-5 --num_fold 1
```

multimodal_model 기반 감정(정도) 학습/평가 모델 실행
```
python multimodal_regression.py --batch_size 32 --lr 1e-5 --num_fold 1
```

multimodal_model 기반 감정(종류) 학습/평가 모델 실행
```
python multimodal_total.py --batch_size 32 --lr 1e-5 --num_fold 1
```
