# farmsall-image-recognition.git

## 소개

이 저장소는 팜한농의 팜스올 사업의 일부입니다. 특히, 이미지를 통한 작물의 병해를 인지하는 분류 모델과 Out-of-distribution 문제 해결을 포함합니다.

## 필수 패키지
-   ```pip install transformers pytorch-ood lightning tensorboard```
- 또는 *requirements.txt* 파일의 패키지 리스트를 참고하여 설치할 것

## 사용법

0. Global 변수들
> - config.py  
공통으로 사용하는 변수들이 정의됨.

1. 데이터 준비
> -  prepare_data_resize.py  
학습 및 평가에 사용할 데이터셋을 준비/변환하는 코드. 원본데이터로부터 모델학습에 필요한 데이터셋 구조로 변경하기 위한 목적임.

2. EDA
> - inspect_dataset.ipynb  
이미지 분류모델과 OOD detector의 학습/평가에 사용되는 데이터셋들의 분포를 확인함

3. 병해분류 모델 학습
> - train_clf_cnn.py  
CNN기반의 모델을 학습. pytorch로 부터 CNN기반의 pre-trained 모델을 다운로드하여 fine-tuning함
> - train_clf_cnn.py  
ViT기반의 모델을 학습. HuggingFace로 부터 Vision Transformer기반의 pre-trained 모델을 다운로드하여 fine-tuning함

4. OOD 분류 모델 학습
> - train_clf_ood.py  
pytorch-ood 프로젝트를 활용하여, 다양한 OOD detector를 fit하고, OOD분류 threshold를 학습, 평가함

5. Evaluation
> - ```tensorboard --logdir=logs/farmsall --bind_all```  
이미지 classifier의 평가 지표, 모델 graph 및 confusion matrix는 텐서보드에서 확인함

> - report_ood_farms.ipynb  
OOD detector모델의 성능 평가 결과를 확인함. 모델이 저장된 logs 폴더의 경로를 설정후 실행함.

> - report_farms.ipynb  
이미지 classifier와 OOD detector를 같이 고려하여, 딸기8종+OOD를 함께 고려한 평가 결과.

> - predict.py  
이미지 분류기 predict 예시 코드


5. 모델 사용
> - predict_one.ipynb  
실제로 이미지 1개가 주어졌을때, 분류 결과를 얻는 예제 코드들.  
각 모델들의 predict 함수를 제공함 : ```predict_img```, ```predict_ood```, ```ensemble_ood```, ```predict```


## 폴더 구조 및 설명
> - ```data```  
>   - clf_loaders.py  
    이미지 classifier 학습/평가에 사용할 Dataloader를 정의. 신규 데이터를 정의할 경우 수정 필요
>   - ood_loader.py  
    OOD detector 학습/평가에 사용할 Dataloader를 정의. 신규 데이터를 정의할 경우 수정 필요

> - ```dataset```  
    config.py에서 정의된대로, 학습/평가에 사용할 데이터셋 폴더임. ```.gitignore```에 추가되어 있으므로 별도 관리해야함.

> - ```logs```  
    config.py에서 정의된대로, 모델, 학습/평가 결과를 저장하는 폴더임. ```.gitignore```에 추가되어 있으므로 별도 관리해야함.

> - ```org```  
    원본 데이터를 저장하는 폴더임. 공통저장소의 soft link로 만들어 사용할 것을 추천함. ```.gitignore```에 추가되어 있으므로 별도 관리해야함.

## Acknowledgement

이 저장소의 산출물은 팜한농 신사업팀의 발주로 진행되었습니다.
