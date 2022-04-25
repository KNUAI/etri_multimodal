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
    
    KEMDy19 데이터셋의 경우 텍스트/음성의 매핑이 불가능한 12개의 파일을 제외하고 20,554개를 사용
    KEMDy20 데이터셋은 13,462개 모두를 사용
    
    5-fold 방식을 적용하여 각 데이터셋을 train:test=8:2의 비율로 5번의 교차검증을 수행함
    
    텍스트의 전처리는 KLUE 데이터셋으로 학습된 BERT 기반의 bert-base tokenizer를 사용하여 주어진 데이터셋의 모든 텍스트 데이터를 벡터화함
    음성의 전처리는 facebook의 wav2vec2-base-960h을 사용하여 주어진 데이터셋의 모든 음성 데이터를 5초로 샘플링하고 벡터화함

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

text_model 기반 감정(종류 및 정도) 학습/평가 모델 실행
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

speech_model 기반 감정(종류 및 정도) 학습/평가 모델 실행
```
python speech_total.py --batch_size 32 --lr 1e-5 --num_fold 1
```

multimodal_model(concat) 기반 감정(종류) 학습/평가 모델 실행
```
python multimodal_classification_concat.py --batch_size 32 --lr 1e-5 --num_fold 1
```

multimodal_model(concat) 기반 감정(정도) 학습/평가 모델 실행
```
python multimodal_regression_concat.py --batch_size 32 --lr 1e-5 --num_fold 1
```

multimodal_model(concat) 기반 감정(종류 및 정도) 학습/평가 모델 실행
```
python multimodal_total_concat.py --batch_size 32 --lr 1e-5 --num_fold 1
```

multimodal_model(element_wise_add) 기반 감정(종류) 학습/평가 모델 실행
```
python multimodal_classification_element_wise_add.py --batch_size 32 --lr 1e-5 --num_fold 1
```

multimodal_model(element_wise_add) 기반 감정(정도) 학습/평가 모델 실행
```
python multimodal_regression_element_wise_add.py --batch_size 32 --lr 1e-5 --num_fold 1
```

multimodal_model(element_wise_add) 기반 감정(종류 및 정도) 학습/평가 모델 실행
```
python multimodal_total_element_wise_add.py --batch_size 32 --lr 1e-5 --num_fold 1
```
