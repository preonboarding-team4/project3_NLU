NLU
## NLU - 문장 유사도 계산 (STS)
### 과제 목표
한글 문장의 유사도 측정 모델 제작: 
두 한글 문장의 유사도를 점수 또는 Lable로 출력하는 모델
API를 통한 서비스화
### 과제 정보
기간: 3.7~24
참여자:[이강희](https://kanghee.notion.site/Lee-Kang-Hee-Kay-a04dfb8d8eb24fd3b1ed215219154a3b, "Kangheelee cv"), 이승수, [최혜림](https://github.com/hyelimchoi1223), 최화랑, 황찬희

## 과제 산출물
기업과제3_4팀.ipynb
KLUE STS data를 이용한 모델 학습 및 검증
데이터 전처리를 통한 모델 성능 비교 및 최종 모델 선정
데이터 증강 시도?
기업과제3_4팀_dev_set_score.csv
KLUE STS 데이터셋 컬럼에 predict real label, predict binary labe 추가한 csv 파일
기업과제3_6팀_dev_set_score.ipynb
dev_set_score.csv 생성 과정 출력
dev_set_score의 예측값과 실제값 F1 score 와 Pearson's R 결과물 출력
API
Fast API를 이용하여 API서버 구현
[Fast API 공식 docs](https://fastapi.tiangolo.com/tutorial/bigger-applications/)를 참고하여 프로젝트 구조 생성
프로젝트 실행 편의를 위하여 직접 설치하여 실행하는 방법과 도커 이미지를 이용하는 방법 2가지 제공


## Process
모델
KRBERT sub-character
선정 이유:
학습 데이터, Tokenizing 방식에서 가장 선택 범위가 넓은 BERT 모델
sub-character wordPiece(한글의 음절 단위를 기준으로 tokenize)

하이퍼파라미터 튜닝
Auto-Parameter tuning:
NNI(Neural Network Intelligence)과 Early-stopping을 사용
Learning rate, Batch size, Optimizer, Epoch
직접 튜닝
# of Layer: Liner layer의 개수
Layer unit(size)s: Liner layer를 구성하는 unit의 수

평가지표
F1-Score, Pearson’s r

모델 성능
### 최종 파라미터
| Parameter | value |
|---|---|
| Epoch | 5 |
| Batch size | 32 |
| Learning rate | 2e-5 |
| Optimizer | NAdam |
| # of Layer | 1 |
| Layer unit | 768 |


### 최종 모델 성능
| F1 score | Pearson corr |
|---|---|
| 79.03    | 86.14        |


리뷰 및 분석 
input
``` 
유사도 측정 모델 결과: 
"sentence1": "숙소 위치는 찾기 쉽고 일반적인 한국의 반지하 숙소입니다.",  
"sentence2": "숙박시설의 위치는 쉽게 찾을 수 있고 한국의 대표적인 반지하 숙박시설입니다."
```  
output
```
"label": 3.9,  
"real label": 3.917365074157715,  
"binary label": 1
```
