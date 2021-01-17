##############################  Python(資料清洗)  ###################################
import pandas as pd
import numpy as np
import math
plt.style.use('ggplot')

# 資料讀取
dat1 = pd.read_csv('train_buy_info.csv')
dat2 = pd.read_csv('train_cust_info.csv')
dat3 = pd.read_csv('train_tpy_info.csv')
dat4 = pd.read_csv('test_buy_x_info.csv')
dat5 = pd.read_csv('test_cust_x_info.csv')
dat6 = pd.read_csv('test_tpy_x_info.csv')

df1 = pd.merge(dat1, dat2, how='left', on = ['CUST_ID'])
df1 = pd.merge(df1,dat3, how='left', on = ['CUST_ID'])

df2 = pd.merge(dat4, dat5, how='left', on = ['CUST_ID'])
df2 = pd.merge(df2,dat6, how='left', on = ['CUST_ID'])
df = pd.concat([df1,df2],ignore_index=True)

aa = df[['BUY_TYPE','BUY_TPY1_NUM_CLASS', 'BUY_TPY2_NUM_CLASS', 'BUY_TPY3_NUM_CLASS',
       'BUY_TPY4_NUM_CLASS', 'BUY_TPY5_NUM_CLASS', 'BUY_TPY6_NUM_CLASS','BUY_TPY7_NUM_CLASS']]

aa['y'] = df['BUY_TYPE'].copy().apply({'a':1, 'e':2, 'b':3,'c':4,'f':5,'d':6,'g':7}.get)
aa['B1'] = aa.BUY_TPY1_NUM_CLASS.copy().apply({'A':1, 'B':2, 'C':3,'D':4,'E':5,'F':6,'G':7}.get)
aa['B2'] = aa.BUY_TPY2_NUM_CLASS.copy().apply({'A':1, 'B':2, 'C':3,'D':4,'E':5,'F':6,'G':7}.get)
aa['B3'] = aa.BUY_TPY3_NUM_CLASS.copy().apply({'A':1, 'B':2, 'C':3,'D':4,'E':5,'F':6,'G':7}.get)
aa['B4'] = aa.BUY_TPY4_NUM_CLASS.copy().apply({'A':1, 'B':2, 'C':3,'D':4,'E':5,'F':6,'G':7}.get)
aa['B5'] = aa.BUY_TPY5_NUM_CLASS.copy().apply({'A':1, 'B':2, 'C':3,'D':4,'E':5,'F':6,'G':7}.get)
aa['B6'] = aa.BUY_TPY6_NUM_CLASS.copy().apply({'A':1, 'B':2, 'C':3,'D':4,'E':5,'F':6,'G':7}.get)
aa['B7'] = aa.BUY_TPY7_NUM_CLASS.copy().apply({'A':1, 'B':2, 'C':3,'D':4,'E':5,'F':6,'G':7}.get)

aa.drop( ['BUY_TPY1_NUM_CLASS', 'BUY_TPY2_NUM_CLASS', 'BUY_TPY3_NUM_CLASS',
       'BUY_TPY4_NUM_CLASS', 'BUY_TPY5_NUM_CLASS',
'BUY_TPY6_NUM_CLASS','BUY_TPY7_NUM_CLASS']  ,axis=1,inplace=True )

a1 = []
for i in range(0,aa.shape[0]):
    a1.append( np.argmin( np.array( aa.iloc[i,2:9]) ) +1    )
aa['pred'] = a1
aa[['B1','B2','B3','B4','B5','B6','B7']] = abs(aa[['B1','B2','B3','B4','B5','B6','B7']]-7)

aa[['B1','B2','B3','B4','B5','B6','B7','y']][aa.y.notna()]
bb=[]
for i in  range(0,aa.shape[0]):
    if int(aa.ix[i,'B1']==0)+int(aa.ix[i,'B2']==0)+int(aa.ix[i,'B3']==0)+\
    int(aa.ix[i,'B4']==0)+int(aa.ix[i,'B5']==0)+int(aa.ix[i,'B6']==0)+int(aa.ix[i,'B7']==0) >5 : bb.append(1)
    else: bb.append(0)

aa['pred_control'] = bb*aa.pred
aa = aa[['BUY_TYPE', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7','y', 'pred','pred_control']]

# train data  (233894 rows x 47 columns)
df[df.BUY_TYPE.notna()]

# test data   (5610 rows x 47 columns)
df[df.BUY_TYPE.isna()]

df.BUY_TPY1_NUM_CLASS = abs(df.BUY_TPY1_NUM_CLASS.apply({'A':1, 'B':2, 'C':3,'D':4,'E':5,'F':6,'G':7}.get)-7 )
df.BUY_TPY2_NUM_CLASS = abs(df.BUY_TPY2_NUM_CLASS.apply({'A':1, 'B':2, 'C':3,'D':4,'E':5,'F':6,'G':7}.get)-7 )
df.BUY_TPY3_NUM_CLASS = abs(df.BUY_TPY3_NUM_CLASS.apply({'A':1, 'B':2, 'C':3,'D':4,'E':5,'F':6,'G':7}.get)-7 )
df.BUY_TPY4_NUM_CLASS = abs(df.BUY_TPY4_NUM_CLASS.apply({'A':1, 'B':2, 'C':3,'D':4,'E':5,'F':6,'G':7}.get)-7 )
df.BUY_TPY5_NUM_CLASS = abs(df.BUY_TPY5_NUM_CLASS.apply({'A':1, 'B':2, 'C':3,'D':4,'E':5,'F':6,'G':7}.get)-7 )
df.BUY_TPY6_NUM_CLASS = abs(df.BUY_TPY6_NUM_CLASS.apply({'A':1, 'B':2, 'C':3,'D':4,'E':5,'F':6,'G':7}.get)-7 )
df.BUY_TPY7_NUM_CLASS = abs(df.BUY_TPY7_NUM_CLASS.apply({'A':1, 'B':2, 'C':3,'D':4,'E':5,'F':6,'G':7}.get)-7 )

df.AGE = df.AGE.apply( {'a':1,'b':2, 'c':3, 'd':4, 'e':5, 'f':6, 'g':7, 'h':8, 'i':9, 'j':1,'k':11, 'l':12, \
                        'm':13, 'n':14, 'o':15, 'p':16, 'q':17}.get)
df.SEX = df.SEX.apply( {'a':0,'b':1}.get)
df.OCCUPATION = df.OCCUPATION.str[0]
del df['BUY_YEAR']

df.STATUS1 = df.STATUS1.fillna('b').apply( {'a':0,'b':1}.get)
df.STATUS2 = df.STATUS2.fillna('b').apply( {'a':0,'b':1}.get)
df.STATUS3 = df.STATUS3.fillna('b').apply( {'a':0,'b':1}.get)
df.STATUS4 = df.STATUS4.fillna('a').apply( {'a':0,'b':1}.get)

model =    pd.concat([aa.y,
           pd.get_dummies(df.EDUCATION,drop_first=True,prefix='EDUCATION'),
           pd.get_dummies(df.MARRIAGE,drop_first=True,prefix='MARRIAGE'),
           pd.get_dummies(df.CITY_CODE,drop_first=True,prefix='CITY_CODE'),
           pd.get_dummies(df.OCCUPATION,drop_first=True,prefix='OCCUPATION'),
           pd.get_dummies(df.BEHAVIOR_1,prefix='BEHAVIOR_1'),
           pd.get_dummies(df.BEHAVIOR_2,prefix='BEHAVIOR_2'),
           pd.get_dummies(df.BEHAVIOR_3,prefix='BEHAVIOR_3'),
           pd.get_dummies(df.CHARGE_WAY,prefix='CHARGE_WAY'),
           pd.get_dummies(df.IS_APP,prefix='IS_APP'),
           pd.get_dummies(df.IS_EMAIL,prefix='IS_EMAIL'),
           pd.get_dummies(df.IS_MAJOR_INCOME,prefix='IS_MAJOR_INCOME',drop_first=True),
           pd.get_dummies(df.IS_NEWSLETTER,prefix='IS_NEWSLETTER'),
           pd.get_dummies(df.IS_PHONE,prefix='IS_PHONE'),
           pd.get_dummies(df.IS_SPECIALMEMBER,prefix='IS_SPECIALMEMBER'),
           pd.get_dummies(df.PARENTS_DEAD,prefix='PARENTS_DEAD',drop_first=True),
           pd.get_dummies(df.REAL_ESTATE_HAVE,prefix='REAL_ESTATE_HAVE',drop_first=True),
           pd.get_dummies(df.BUY_MONTH,prefix='BUY_MONTH',drop_first=True),
           df.BUDGET,
           df.BUY_TPY1_NUM_CLASS,    
           df.BUY_TPY2_NUM_CLASS,   
           df.BUY_TPY3_NUM_CLASS,   
           df.BUY_TPY4_NUM_CLASS,   
           df.BUY_TPY5_NUM_CLASS,   
           df.BUY_TPY6_NUM_CLASS,   
           df.BUY_TPY7_NUM_CLASS,   
           df.SEX,                   
           df.STATUS1,            
           df.STATUS2,          
           df.STATUS3,          
           df.STATUS4,          
           df.WEIGHT,   
           df.CHILD_NUM,
           df.HEIGHT,
           df.AGE], axis=1)
model.to_csv(df.csv',index=0')

##############################  R(遺失值處理)  ###################################
library(tidyverse);library(Metrics);library(mice)
df =  read.csv(df.csv')
mice.data <- mice(df1[,-1],
                  m = 1,           # 產生1個被填補好的資料表
                  maxit = 50,      # max iteration
                  method = "cart", # 使用CART決策樹，進行遺漏值預測
                  seed = 188)      # set.seed()，令抽樣每次都一樣
new_data1 = complete(mice.data, 1) # 1st data
new_data = cbind(y=df1[,1],new_data1)

###########################  Python(建模XGboost)  #################################
# 用R 套件mice補遺失值匯入
from sklearn import preprocessing
import xgboost as xgb
from xgboost import XGBClassifier

new_data = pd.read_csv('new_data.csv')
new_data.CHILD_NUM = preprocessing.scale( new_data.CHILD_NUM)
features = [f for f in new_data.columns if f not in ['y'] ]
new_data.to_csv('new_data(1).csv')

model_1 = XGBClassifier(max_depth=8, learning_rate=0.05, n_estimators=700,silent=False, 
                        objective='multi:softprob', booster='gbtree', metrics='merror',
                        n_jobs=3, nthread=4, min_child_weight=2,subsample=0.8, 
                        colsample_bytree=0.7,num_class=7,seed=123,num_boost_round=700)

model_1.fit(new_data[new_data.y.notna()].iloc[:,1:],new_data[new_data.y.notna()].iloc[:,0]-1 , eval_metric='merror',verbose=True )

model_1.predict(new_data[new_data.y.isna()].iloc[:,1:])

raw_pred_prob = model_1.predict_proba(new_data.iloc[:,1:])
raw_pred = model_1.predict(new_data.iloc[:,1:])

mix_1 = pd.DataFrame({'y':new_data.y,
                          'yhat':raw_pred+1,
                          'prob_1':raw_pred_prob[:,0],
                          'prob_2':raw_pred_prob[:,1], 
                          'prob_3':raw_pred_prob[:,2], 
                          'prob_4':raw_pred_prob[:,3], 
                          'prob_5':raw_pred_prob[:,4], 
                          'prob_6':raw_pred_prob[:,5], 
                          'prob_7':raw_pred_prob[:,6]
                      } )
mix_1.to_csv('xgboost_prob.csv',index=0,na_rep='NA')

 ########################### R(建模RandomForest)  #################################
library(randomForest)
df = read.csv('new_data(1).csv')
train_index = setdiff(1:239504,which(is.na(df$y)))
data = df[train_index,-1]
data$y = factor(data$y)
model_rf = randomForest(y~. , data=data , ntree=100 , mtry = 21) 
prob = predict(model_rf,df[,-c(1,2)],type="prob")
class = predict(model_rf,df[,-c(1,2)])
rf_9Xn = cbind(df$y,class,prob)
write.csv(rf_9Xn,file="F:/R/randomforest.csv")

###########################  R(建模SVM)  #########################################
library(tidyverse); library(e1071)

# load data
df = read.csv('new_data(1).csv')
df = transform(df,y = factor(y))
dat = df %>% select(-X) 

# training index
train_pp = setdiff(1:239504, which(is.na(df$y)))

# model
model = svm(y~., data = dat[train_pp,], kernel = "radial", cost = 11, probability = TRUE)

# predict
pred = predict(model, dat[,-1], probability = TRUE)
write.csv(pred ,file="F:/R/svm.csv")

###########################  R(集成分析)  #########################################
library(tidyverse);library(glmnet);library(caret);library(nnet);library(xgboost)
model1 = read.csv('randomforest.csv')
model1 = model1[,-1] 
colnames(model1)=c('y','yhat','x1','x2','x3','x4','x5','x6','x7')
model2 = read.csv('xgboost.csv')
model3 = read.csv('svm.csv')
model3 = model3[,-1] 
colnames(model3)=c('y','yhat','v1','v2','v3','v4','v5','v6','v7')
X1 = cbind(model1[,c(1,3:9)],model2[3:9],model3[3:9])

# 非NA index
notna_index = which(as.numeric(is.na(X1$y))==0)

model_1 <- nnet::multinom(y ~.-1, data = X1[notna_index,])
predicted.classes_1 <- model_1 %>% predict(X1[,-1])

dtrain = xgb.DMatrix(data= as.matrix(X1[notna_index,-1]),label =  X1[notna_index,1] )

xgb_params = list(subsample=0.8,max_depth=4,eta=0.03,
                  eval_metric = "merror",objective = "multi:softprob",num_class = 8)
cv_model2 = xgb.cv(params=xgb_params,  data= as.matrix(X1[notna_index,-1]),
                   label =  X1[notna_index,1] ,nrounds = 100,nfold = 10)
model_2 = xgboost(dtrain,subsample=0.8,max_depth=4,eta=0.03,
                  nrounds = which.min(cv_model2$evaluation_log$test_merror_mean),
                  eval_metric = "merror",objective = "multi:softmax",num_class = 8)
predicted.classes_2 = predict(model_2,as.matrix(X1[,-1] ))

qq1 = predicted.classes_2
qq1[qq1==1] = 'a'
qq1[qq1==2] = 'e'
qq1[qq1==3] = 'b'
qq1[qq1==4] = 'c'
qq1[qq1==5] = 'f'
qq1[qq1==6] = 'd'
qq1[qq1==7] = 'g'

submit =  data.frame('CUST_ID' = df$CUST_ID[229505:239504],
                     'BUY_TYPE'= qq1[229505:239504])
write.csv(submit, file = "submit.csv",row.names=FALSE)

################################################################################
