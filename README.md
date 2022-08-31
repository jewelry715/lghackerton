## LG Hackerton 자율주행 센서의 안테나 성능 예측
- 데이터 및 태스크 유형 : 정형데이터 - 회귀 
- 기간 : 2022-08-01 ~ 2022-08-26
- 사용 툴 : Colab, Jupyter, Pycharm
- 사용 모델 : H2O(AutoML), Pycaret(AutoML), KNN, LightGBM, XGBoost, CatBoost, MLP
- 리더보드 순위 : Top 6%
## 작업 목록(채송화)
1. 상관분석 그래프&수치화 결과
-왼쪽부터 x_01,x_02,x_03,x_04 순으로 진행
-y05:안테나 gain 평균(각도2)
![Untitled](https://user-images.githubusercontent.com/56911278/187566449-83292981-3eaf-4ae7-8f79-b75daf4f8cfd.png)

-y06:신호대 잡음비(각도1)
![Untitled (1)](https://user-images.githubusercontent.com/56911278/187566491-15ccea05-2c33-4fb5-b24d-7bb314e23b87.png)

-y07:안테나 gain 평균(각도3)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b3cdfe7a-9728-4e0c-9638-3bac29c0dda5/Untitled.png)

-y08:신호대 잡음비(각도2)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/02039177-3a61-4b03-b75e-fb47894c0e73/Untitled.png)

⇒y변수 전부 다 검사 통과된 상태

⇒y_05(안테나 gain 평균 각도2)에서는 x_13(각 안테나 패드 위치(높이) 차이)이 0.06의 큰 양의 상관관계를 띄는 편

⇒y_06(신호대 잡음비 각도1)은 x_19(스크류 체결 시 분당 회전수 4)이 가장 큰 음의 상관관계(0.063676), x_30이 제일 양의 상관관계(0.044)

⇒y_07(신호대 잡음비 각도2)에선 x_14(1번 안테나 패드 위치)에서 높은 상관성(0.059)을 띔,음의 상관관계는 x_19(-0.093032)가 가장 음의 상관관계를 띔

⇒y_08(신호대 잡음비 각도3)은 x_40(하우징 pcb 안착부 3치수)에서 상관성이 높음(0.14),음의 상관관계가 큰 값은 x_16(0.091816)

⇒그래프를 통해서 확실한 양과 음의 상관관계가 나오지 않음.

⇒y_05~y_07은 0.1이상의 양의 상관관계가 나오지 않고 yf8에서 0.1 이상이 나옴.

⇒양의 상관관계 기준 스크류 삽입깊이1,3(x_19,x_21)이 3개의 변수에 영향을 미치는 것으로 나옴

⇒양의 상관관계 기준 rf1부분 smt 납량,스크류 삽입 깊이4(x_50,x_33)도 2개의 변수에 영향을 미침.

⇒양과 음의 상관관계를 합쳐 3개의 변수에 영향을 미치는 변수는 x_29,x_28,x_25,x_19,x_22,x_06,x_32,

⇒모든 변수에 영향을 미치는 변수는 x_40,x_21

## 2. 모델 설계

-XGboost 사용

- 1,2,3,4 검사 통과여부는 모두 통과이므로 모델 설계시 이 요소는 생략
- 뚜렷한 상관관계가 나오지 않음

 1) y_05:X_49,X_03,x_05,X_08,X_07이 눈에 띈다.
-처음에 상관분석할 때 가장 영향을 준 x_13은 feature_importance가 19

- **rmse: 2.497128**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9c671c34-b910-4766-917e-f29c8412616f/Untitled.png)

2)y_06

 - X_07,x_49가 눈에 띔
 - X_19이 가장 음의 상관계수를 띄지만 여기서는 3위다.
 - X_30이 제일 양의 상관계수를 띄고 여기선 7위

**rmse: 1.813664**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/78229306-0b04-4039-b11d-d04872bfba2d/Untitled.png)

3)y_07

 - X_14에서 높은 상관성을 가짐->하지만 x_14의 중요도는 7
 - x_03,x_07,x_19에서 많이 보임.
 - x_19는 가장 높은 음의 상관계수를 띄는데 3위

**rmse: 0.408422**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2e5627ac-ea54-48b8-854f-63d5ffc95335/Untitled.png)

4)y_08

 - X_20에서 높은 상관성을 가짐->x_20에서는 중요도가 21
 - x_49,x_08에서 가장 높은편.

**rmse: 0.628281**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c412733c-4cd2-4346-9fcd-c155054fbdc4/Untitled.png)

- NRMSE

 - 검사통과여부 제외

  - MultiOutputRegressor 내에 XGBoost 내 XGBRegressor 사용

  - train_test_split으로 test_size를 0.3으로 함

  - train_test_split으로 나온 y_test와 예측값을 사용해 구한 결과 1.97이 나옴

 - 3개 변수 추가 제거(X_02,X_10,X_11): 

  - xgboost default값으로 할 경우 score:1.994

  - 방열 재료 2~3 무게가 0에 수렴함 => 모델 설계시 제거 고민중.feature_importance 순위권에도 거의 없음. 상관관계에서도 양과 음의 상위권에도 없음.

  - X_02도 한쪽으로 치우쳐져 있어서 제거 고민중

  - 검사 통과 여부와 3개의 변수를 더 제거한 후 score: 1.965

  - 그 이후로 Bayesian 튜닝 후 1.9659654193784164 →**1.9265165989248936(2차,(init_points = 3, n_iter = 10),target이 낮은 값에서 나온 파라미터로 활용, 데이콘 제출:1.9497846282)**

1. CatBoost

bayesian을 사용했을 때  ****

**{'bagging_temperature': 0.8965172915507091, 'bootstrap_type': 'Bayesian'}**

**1.9310962426408733 (데이콘 제출 땐 1.9510456453)**

따라서 1개는 optuna, 1개는 default값으로 사용

- optuna 사용 시 :

Best Score: 2.0434492912929
Best trial: {'max_depth': 5, 'bagging_temperature': 85.63992831863465, 'boosting_type': 'Ordered', 'random_strength': 0.300057986515936, 'reg_lambda': 7.323512758601547, 'min_child_samples': 35}

- default 사용 시: **1.9527952782736016**
- dacon 제출 결과 성능은 1.9552511094  > xgboost+파라미터 조정값(1.9727111263)

현재까지 리더보드 점수:

![리더보드.PNG](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5a403724-d34c-4205-bef0-7995c3d45e13/%EB%A6%AC%EB%8D%94%EB%B3%B4%EB%93%9C.png)

⇒xgboost보다 catboost에서 잘 나오는 편

3.KFold+(XGBoost,CatBoost)⇒kfold:5

xgb_params={
'objective':'reg:linear',
'eval_metric':'rmse'
}

1차시도(XGBoost): **1.9470746103894498→1.9492543742192563→1.9799553311702724→1.980339900237043**

→**1.991171846276644 ⇒오히려 score값이 증가함**

⇒평균:1.9694

1차시도(CatBoost):

cat_params={
'eval_metric':'RMSE',
'random_seed':2022
}

**1.9286611507321547→1.9397017548117697→1.9639106517617715→1.9655876942994777→**

**1.9760517687242012⇒score 값 역시 증가함**

평균:1.9543726899287115

⇒catboost에서 성능이 잘 나옴

2차시도

- (XGBoost+Bayesian): **14.188518518158697 ⇒BayesSearchCV시도**
- XGBoost+optuna: 1.98381859566627

⇒역시 score값이 증가하는 편

- (catBoost+optuna):1.9824312717202783⇒
- (catBoost+Bayesian): **1.9466725187124188(데이콘 제출:1.9467756036)**

 - bayesian: **{'bagging_temperature': 0.8965172915507091, 'bootstrap_type': 'Bayesian'} 적용**

MAE 활용

 - cross_val_score:cv10⇒xgboost:**0.6966217068893391,catboost:0.6893753175621163**

⇒catboost가 성능이 잘 나오는편

1. Feature Engineering
- XGBoost default+screw평균(x34~x37지움)+PolynomialFeatures(degree=2): 1.96
- CatBoost default+screw 평균+PolynomialFeatures(degree=2): 1.957707123022563

5.h2o

- 제출결과:1.9460230139 (screw평균(x34~x37지움)을 변수로 추가한 상태,polynomial x)
- n-fold:5로 시도→1.9443267706⇒현재 최고기록
- 곧 Feature Engineering으로 진행할 예정
- Feature Engineering 시도→h2o+polynomial⇒성능이 안좋음
