## 主約

# 套件
library("tidyverse");library(dummies);library(xgboost)

# 讀檔
df1_train = read.csv('TRAIN_MAIN.csv')
df2_train = read.csv('TRAIN_ADD.csv')
df1_test = read.csv('TEST_MAIN.csv')
df2_test = read.csv('TEST_ADD.csv')
df3 = read.csv('CUST.csv')
df4 = read.csv('SUBMMIT_TABLE.csv')

# tibble
df1_train = df1_train %>% as.tibble()
df2_train= df2_train %>% as.tibble()
df1_test = df1_test %>% as.tibble()
df2_test = df2_test %>% as.tibble()
df3 = df3 %>% as.tibble()
df4 = df4 %>% as.tibble()

# 移除一模一樣資料
df1_train = distinct(df1_train)

# NA 處理
df3$PURPOSE_SAFE[is.na(df3$PURPOSE_SAFE)] = 0
df3$PURPOSE_LOAN[is.na(df3$PURPOSE_LOAN)] = 1
df3$PURPOSE_EDU[is.na(df3$PURPOSE_EDU)] = 1
df3$PURPOSE_SAVING[is.na(df3$PURPOSE_SAVING)] = 1
df3$PURPOSE_TAX[is.na(df3$PURPOSE_TAX)] = 1
df3$PURPOSE_OTHER[is.na(df3$PURPOSE_OTHER)] = 1

df3$DISEASE1[is.na(df3$DISEASE1)] = 1
df3$DISEASE2[is.na(df3$DISEASE2)] = 1
df3$DISEASE3[is.na(df3$DISEASE3)] = 1
df3$DISEASE4[is.na(df3$DISEASE4)] = 1

# 利用BMI找出異常值並轉為NA
df3 %>% mutate(BMI = WEIGHT / ((HEIGHT/100)^2)) %>% select(CUST_ID,HEIGHT,WEIGHT,AGE,SEX,BMI) %>%
  filter(BMI > 50 | BMI <= 10 | is.nan(BMI)) %>% arrange(BMI) 

index = df3 %>% mutate(BMI=WEIGHT/((HEIGHT/100)^2)) %>% select(CUST_ID,HEIGHT,WEIGHT,AGE,SEX,BMI) %>%
  filter(BMI > 50 | BMI <= 10 | is.nan(BMI)) %>% select(CUST_ID) %>% as.matrix() %>% as.numeric()

df3[which(df3$CUST_ID %in% index ), c("HEIGHT","WEIGHT")] = NA # 異常BMI
rm(index)

# 將主約之商品中類轉成大類
big = function(i){
  if (i == 1|i == 5|i == 13|i == 15|i == 2|i == 14|i == 16){return(1)}
  else if (i == 3|i == 4|i == 7|i == 8){return(2)}
  else if (i == 6){return(3)}
  else if (i == 9){return(4)}
  else if (i == 10){return(5)}
  else if (i == 17){return(6)}
  else if (i == 12){return(7)}
}

#### 主約train模型 ####

df1_train = df1_train %>% mutate(BIG_TYPE = sapply(PRODUCT_TYPE,big))

# 分出CUST的train-test
custid_train = df1_train$CUST_ID %>% unique() # cust_id in TRAIN_MAIN
df3_train = df3 %>% filter(CUST_ID %in% custid_train) # train part of CUST


# TRAIN_MAIN 的最新一筆與過去紀錄
df1_train_new = df1_train %>% filter(POLICY_ID %in% df3_train$POLICY_ID) %>% 
  arrange(CUST_ID) %>% select(POLICY_ID,CUST_ID,BIG_TYPE)
df1_train_past = df1_train %>% filter(!(POLICY_ID %in% df3_train$POLICY_ID))


## 建模方法:合併資料 

M_train = left_join(df1_train,
                    select(df3,everything(),-c("POLICY_ID", "BUY_YEAR", "BUY_MONTH", 
                                               "AGE", "SEX")),
                    by="CUST_ID") # 併上去，保留舊的資料

# M_train %>% group_by(CUST_ID) %>% count() %>% ungroup %>% select(n) %>% table() # 最新+過去資料的筆數

## 合併資料

# 過去紀錄中大類買過的個數總和
buy_type_sum = df1_train_past %>% group_by(CUST_ID) %>% 
  summarise(PAST_BUY_TYPE1 = sum((BIG_TYPE == 1)),
            PAST_BUY_TYPE2 = sum((BIG_TYPE == 2)),
            PAST_BUY_TYPE3 = sum((BIG_TYPE == 3)),
            PAST_BUY_TYPE4 = sum((BIG_TYPE == 4)),
            PAST_BUY_TYPE5 = sum((BIG_TYPE == 5)),
            PAST_BUY_TYPE6 = sum((BIG_TYPE == 6)),
            PAST_BUY_TYPE7 = sum((BIG_TYPE == 7)))

X = left_join(df3_train,buy_type_sum,by="CUST_ID") %>% arrange(CUST_ID)
y_train = df1_train_new %>% arrange(CUST_ID) %>% select(CUST_ID,BIG_TYPE) %>% arrange(CUST_ID)

# 只有一筆購買紀錄的會被記成NA 改成0
X$PAST_BUY_TYPE1[is.na(X$PAST_BUY_TYPE1)] = 0
X$PAST_BUY_TYPE2[is.na(X$PAST_BUY_TYPE2)] = 0
X$PAST_BUY_TYPE3[is.na(X$PAST_BUY_TYPE3)] = 0
X$PAST_BUY_TYPE4[is.na(X$PAST_BUY_TYPE4)] = 0
X$PAST_BUY_TYPE5[is.na(X$PAST_BUY_TYPE5)] = 0
X$PAST_BUY_TYPE6[is.na(X$PAST_BUY_TYPE6)] = 0
X$PAST_BUY_TYPE7[is.na(X$PAST_BUY_TYPE7)] = 0

# 過去紀錄中大類保額累計
# 考慮保額單位和幣值

# 將保額單位為"7.單位"的獨立出來加總
unit_7 = df1_train_past %>% group_by(CUST_ID) %>%
  summarise(unit7 = sum(MAIN_AMOUNT_UNIT==7) )


# 改資料

# 將CURRENCY依據匯率改值
df1_train_past$CURRENCY = as.numeric(df1_train_past$CURRENCY )
df1_train_past$CURRENCY[df1_train_past$CURRENCY==1] = 21.5
df1_train_past$CURRENCY[df1_train_past$CURRENCY==2] = 4.3
df1_train_past$CURRENCY[df1_train_past$CURRENCY==3] = 1.0
df1_train_past$CURRENCY[df1_train_past$CURRENCY==4] = 30.5
# df1_train_past$CURRENCY = as.factor(df1_train_past$CURRENCY)

# 保額單位依數量大小改值
df1_train_past$MAIN_AMOUNT_UNIT[df1_train_past$MAIN_AMOUNT_UNIT==1] = 1
df1_train_past$MAIN_AMOUNT_UNIT[df1_train_past$MAIN_AMOUNT_UNIT==3] = 1000
df1_train_past$MAIN_AMOUNT_UNIT[df1_train_past$MAIN_AMOUNT_UNIT==4] = 10000
df1_train_past$MAIN_AMOUNT_UNIT[df1_train_past$MAIN_AMOUNT_UNIT==7] = 0

# 過去紀錄之保額累計
amount_sum = df1_train_past %>% group_by(CUST_ID) %>%
  summarise(AMOUNT_SUM1 = sum((BIG_TYPE==1)*MAIN_AMOUNT*MAIN_AMOUNT_UNIT*CURRENCY),
            AMOUNT_SUM2 = sum((BIG_TYPE==2)*MAIN_AMOUNT*MAIN_AMOUNT_UNIT*CURRENCY),
            AMOUNT_SUM3 = sum((BIG_TYPE==3)*MAIN_AMOUNT*MAIN_AMOUNT_UNIT*CURRENCY),
            AMOUNT_SUM4 = sum((BIG_TYPE==4)*MAIN_AMOUNT*MAIN_AMOUNT_UNIT*CURRENCY),
            AMOUNT_SUM5 = sum((BIG_TYPE==5)*MAIN_AMOUNT*MAIN_AMOUNT_UNIT*CURRENCY),
            AMOUNT_SUM6 = sum((BIG_TYPE==6)*MAIN_AMOUNT*MAIN_AMOUNT_UNIT*CURRENCY),
            AMOUNT_SUM7 = sum((BIG_TYPE==7)*MAIN_AMOUNT*MAIN_AMOUNT_UNIT*CURRENCY))

# 過去紀錄之幣別累計
CURRENCY_sum = df1_train_past %>% group_by(CUST_ID) %>% 
  summarise(CURRENCY_1 =  sum(CURRENCY==1.0),
            CURRENCY_2 =  sum(CURRENCY==4.3),
            CURRENCY_3 =  sum(CURRENCY==30.5),
            CURRENCY_4 =  sum(CURRENCY==21.5))

# 過去紀錄之保費累計(考慮幣別)
PREMIUM_sum = df1_train_past %>% group_by(CUST_ID) %>% 
  summarise(PREMIUM_1 = sum((BIG_TYPE==1)*MAIN_PREMIUM*CURRENCY),
            PREMIUM_2 = sum((BIG_TYPE==2)*MAIN_PREMIUM*CURRENCY),
            PREMIUM_3 = sum((BIG_TYPE==3)*MAIN_PREMIUM*CURRENCY),
            PREMIUM_4 = sum((BIG_TYPE==4)*MAIN_PREMIUM*CURRENCY),
            PREMIUM_5 = sum((BIG_TYPE==5)*MAIN_PREMIUM*CURRENCY),
            PREMIUM_6 = sum((BIG_TYPE==6)*MAIN_PREMIUM*CURRENCY),
            PREMIUM_7 = sum((BIG_TYPE==7)*MAIN_PREMIUM*CURRENCY))

# 過去紀錄之繳別累計
PAY_FREQ = df1_train_past %>% group_by(CUST_ID) %>% 
  summarise(PAY_FREQ1 = sum(MAIN_PAY_FREQ==1),
            PAY_FREQ2 = sum(MAIN_PAY_FREQ==2),
            PAY_FREQ3 = sum(MAIN_PAY_FREQ==3),
            PAY_FREQ4 = sum(MAIN_PAY_FREQ==4),
            PAY_FREQ5 = sum(MAIN_PAY_FREQ==5),
            PAY_FREQ6 = sum(MAIN_PAY_FREQ==6))

# 過去紀錄之繳費管道累計
PAY_WAY = df1_train_past %>% group_by(CUST_ID) %>% 
  summarise(PAY_WAY1 = sum(MAIN_PAY_WAY==1),
            PAY_WAY2 = sum(MAIN_PAY_WAY==2),
            PAY_WAY3 = sum(MAIN_PAY_WAY==3),
            PAY_WAY4 = sum(MAIN_PAY_WAY==4),
            PAY_WAY5 = sum(MAIN_PAY_WAY==6))

# 過去紀錄之銷售通路累計
SALE_CHANNEL = df1_train_past %>% group_by(CUST_ID) %>% 
  summarise(SALE_CHANNEL1 = sum(MAIN_SALE_CHANNEL==0),
            SALE_CHANNEL2 = sum(MAIN_SALE_CHANNEL==1),
            SALE_CHANNEL3 = sum(MAIN_SALE_CHANNEL==2),
            SALE_CHANNEL4 = sum(MAIN_SALE_CHANNEL==3),
            SALE_CHANNEL5 = sum(MAIN_SALE_CHANNEL==4))

# 過去紀錄之購買年份累計
BUY_YEAR  = df1_train_past %>% group_by(CUST_ID) %>% 
  summarise(BUY_YEAR1 = sum(BUY_YEAR==2006),
            BUY_YEAR2 = sum(BUY_YEAR==2007),
            BUY_YEAR3 = sum(BUY_YEAR==2008),
            BUY_YEAR4 = sum(BUY_YEAR==2009),
            BUY_YEAR5 = sum(BUY_YEAR==2010),
            BUY_YEAR6 = sum(BUY_YEAR==2011),
            BUY_YEAR7 = sum(BUY_YEAR==2012),
            BUY_YEAR8 = sum(BUY_YEAR==2013),
            BUY_YEAR9 = sum(BUY_YEAR==2014),
            BUY_YEAR10 = sum(BUY_YEAR==2015),
            BUY_YEAR11 = sum(BUY_YEAR==2016),
            BUY_YEAR12 = sum(BUY_YEAR==2017),
            BUY_YEAR13 = sum(BUY_YEAR==2018) )

# MAIN_PERIOD
# 將主約保險年期資料切成5類
add1 = df1_train_past$MAIN_PERIOD
add1[add1<20] = 1
add1[(add1>=20) & (add1<40) ] = 2
add1[(add1>=40) & (add1<95) ] = 3
add1[(add1>=95) & (add1<200) ] = 4
add1[(add1>=200)] = 5
df1_train_past$add1 = add1

# 過去紀錄之主約保險年期類別累計
MAIN_PERIOD =  df1_train_past %>% group_by(CUST_ID) %>% 
  summarise(MAIN_PERIOD1 = sum(add1==1),
            MAIN_PERIOD2 = sum(add1==2),
            MAIN_PERIOD3 = sum(add1==3),
            MAIN_PERIOD4 = sum(add1==4),
            MAIN_PERIOD5 = sum(add1==5))

# MAIN_PAY_PERIOD
# 將主約繳費年期切為8類
add2 = df1_train_past$MAIN_PAY_PERIOD
add2[add2<6] = 1
add2[(add2>=6) & (add2<14) ] = 2
add2[(add2>=14) & (add2<19) ] = 3
add2[(add2>=19) & (add2<24) ] = 4
add2[(add2>=24) & (add2<40) ] = 5
add2[(add2>=40) & (add2<95) ] = 6
add2[(add2>=95) & (add2<101) ] = 7
add2[(add2>=101)] = 8
df1_train_past$add2 = add2

# 過去紀錄之主約繳費年期類別累計
MAIN_PAY_PERIOD =  df1_train_past %>% group_by(CUST_ID) %>% 
  summarise(MAIN_PAY_PERIOD1 = sum(add2==1),
            MAIN_PAY_PERIOD2 = sum(add2==2),
            MAIN_PAY_PERIOD3 = sum(add2==3),
            MAIN_PAY_PERIOD4 = sum(add2==4),
            MAIN_PAY_PERIOD5 = sum(add2==5),
            MAIN_PAY_PERIOD6 = sum(add2==6),
            MAIN_PAY_PERIOD7 = sum(add2==7),
            MAIN_PAY_PERIOD8 = sum(add2==8))
rm(add1,add2)

# 將所有新變數與原資料合併
X1 = left_join(X,unit_7)
X2 = left_join(X1,amount_sum)
X3 = left_join(X2,CURRENCY_sum)
X4 = left_join(X3,PREMIUM_sum)
X5 = left_join(X4,PAY_FREQ)
X6 = left_join(X5,PAY_WAY)
X7 = left_join(X6,SALE_CHANNEL)
X8 = left_join(X7,BUY_YEAR)
X9 = left_join(X8,MAIN_PERIOD)
X10 = left_join(X9,MAIN_PAY_PERIOD)
rm(X1,X2,X3,X4,X5,X6,X7,X8,X9)

# 區分只有一筆購買紀錄以及有兩筆以上購買紀錄者
colSums(is.na(X10))[(colSums(is.na(X10))>0) &(colSums(is.na(X10))<144436)]
colSums(is.na(X10))[(colSums(is.na(X10))==144436)]

# 把新變數假如是NA的都改成0(只有一筆資料者)
X10 = data.frame(X10)
for(i in 38:98) {
  X10[is.na(X10[,i]),i] = 0
}

# dummy化
model = cbind(X10[,-c(12:15,22,23)],
              SMOKES=dummy(X10$SMOKES)[,-1],
              ARCEA=dummy(X10$ARCEA)[,-1],
              LIQUEUR=dummy(X10$LIQUEUR)[,-1],
              EDUCATION=dummy(X10$EDUCATION)[,-1],
              CAREER=dummy(X10$CAREER)[,-1],
              VETERAN_STATUS=dummy(X10$VETERAN_STATUS)[,-1])

for (i in 1:ncol(model)) {
  model[,i] = as.numeric(model[,i])
}

#### 主約test模型 ####
# 以下處理皆與上面相同!!
df1_test = df1_test %>% mutate(BIG_TYPE = sapply(PRODUCT_TYPE,big))
df3_test = df3 %>% filter(!(CUST_ID %in% custid_train))

# 過去紀錄中大類買過的個數
buy_type_sum = df1_test %>% group_by(CUST_ID) %>% 
  summarise(PAST_BUY_TYPE1 = sum((BIG_TYPE == 1)),
            PAST_BUY_TYPE2 = sum((BIG_TYPE == 2)),
            PAST_BUY_TYPE3 = sum((BIG_TYPE == 3)),
            PAST_BUY_TYPE4 = sum((BIG_TYPE == 4)),
            PAST_BUY_TYPE5 = sum((BIG_TYPE == 5)),
            PAST_BUY_TYPE6 = sum((BIG_TYPE == 6)),
            PAST_BUY_TYPE7 = sum((BIG_TYPE == 7)))

X = left_join(df3_test,buy_type_sum,by="CUST_ID") %>% arrange(CUST_ID)
y_test = df1_train_new %>% arrange(CUST_ID) %>% select(CUST_ID,BIG_TYPE) %>% arrange(CUST_ID)

X$PAST_BUY_TYPE1[is.na(X$PAST_BUY_TYPE1)] = 0
X$PAST_BUY_TYPE2[is.na(X$PAST_BUY_TYPE2)] = 0
X$PAST_BUY_TYPE3[is.na(X$PAST_BUY_TYPE3)] = 0
X$PAST_BUY_TYPE4[is.na(X$PAST_BUY_TYPE4)] = 0
X$PAST_BUY_TYPE5[is.na(X$PAST_BUY_TYPE5)] = 0
X$PAST_BUY_TYPE6[is.na(X$PAST_BUY_TYPE6)] = 0
X$PAST_BUY_TYPE7[is.na(X$PAST_BUY_TYPE7)] = 0

# 過去紀錄中大類保額累計
# consider CURRENCY and UNIT

unit_7 = df1_test %>% group_by(CUST_ID) %>%
  summarise(unit7 = sum(MAIN_AMOUNT_UNIT==7) )


# 改資料

df1_test$CURRENCY = as.numeric(df1_test$CURRENCY )
df1_test$CURRENCY[df1_test$CURRENCY==1] = 21.5
df1_test$CURRENCY[df1_test$CURRENCY==2] = 4.3
df1_test$CURRENCY[df1_test$CURRENCY==3] = 1.0
df1_test$CURRENCY[df1_test$CURRENCY==4] = 30.5
# df1_train_past$CURRENCY = as.factor(df1_train_past$CURRENCY)

df1_test$MAIN_AMOUNT_UNIT[df1_test$MAIN_AMOUNT_UNIT==1] = 1
df1_test$MAIN_AMOUNT_UNIT[df1_test$MAIN_AMOUNT_UNIT==3] = 1000
df1_test$MAIN_AMOUNT_UNIT[df1_test$MAIN_AMOUNT_UNIT==4] = 10000
df1_test$MAIN_AMOUNT_UNIT[df1_test$MAIN_AMOUNT_UNIT==7] = 0


amount_sum = df1_test %>% group_by(CUST_ID) %>%
  summarise(AMOUNT_SUM1 = sum((BIG_TYPE==1)*MAIN_AMOUNT*MAIN_AMOUNT_UNIT*CURRENCY),
            AMOUNT_SUM2 = sum((BIG_TYPE==2)*MAIN_AMOUNT*MAIN_AMOUNT_UNIT*CURRENCY),
            AMOUNT_SUM3 = sum((BIG_TYPE==3)*MAIN_AMOUNT*MAIN_AMOUNT_UNIT*CURRENCY),
            AMOUNT_SUM4 = sum((BIG_TYPE==4)*MAIN_AMOUNT*MAIN_AMOUNT_UNIT*CURRENCY),
            AMOUNT_SUM5 = sum((BIG_TYPE==5)*MAIN_AMOUNT*MAIN_AMOUNT_UNIT*CURRENCY),
            AMOUNT_SUM6 = sum((BIG_TYPE==6)*MAIN_AMOUNT*MAIN_AMOUNT_UNIT*CURRENCY),
            AMOUNT_SUM7 = sum((BIG_TYPE==7)*MAIN_AMOUNT*MAIN_AMOUNT_UNIT*CURRENCY))

CURRENCY_sum = df1_test %>% group_by(CUST_ID) %>% 
  summarise(CURRENCY_1 =  sum(CURRENCY==1.0),
            CURRENCY_2 =  sum(CURRENCY==4.3),
            CURRENCY_3 =  sum(CURRENCY==30.5),
            CURRENCY_4 =  sum(CURRENCY==21.5))


PREMIUM_sum = df1_test %>% group_by(CUST_ID) %>% 
  summarise(PREMIUM_1 = sum((BIG_TYPE==1)*MAIN_PREMIUM*CURRENCY),
            PREMIUM_2 = sum((BIG_TYPE==2)*MAIN_PREMIUM*CURRENCY),
            PREMIUM_3 = sum((BIG_TYPE==3)*MAIN_PREMIUM*CURRENCY),
            PREMIUM_4 = sum((BIG_TYPE==4)*MAIN_PREMIUM*CURRENCY),
            PREMIUM_5 = sum((BIG_TYPE==5)*MAIN_PREMIUM*CURRENCY),
            PREMIUM_6 = sum((BIG_TYPE==6)*MAIN_PREMIUM*CURRENCY),
            PREMIUM_7 = sum((BIG_TYPE==7)*MAIN_PREMIUM*CURRENCY))

PAY_FREQ = df1_test %>% group_by(CUST_ID) %>% 
  summarise(PAY_FREQ1 = sum(MAIN_PAY_FREQ==1),
            PAY_FREQ2 = sum(MAIN_PAY_FREQ==2),
            PAY_FREQ3 = sum(MAIN_PAY_FREQ==3),
            PAY_FREQ4 = sum(MAIN_PAY_FREQ==4),
            PAY_FREQ5 = sum(MAIN_PAY_FREQ==5),
            PAY_FREQ6 = sum(MAIN_PAY_FREQ==6))

PAY_WAY = df1_test %>% group_by(CUST_ID) %>% 
  summarise(PAY_WAY1 = sum(MAIN_PAY_WAY==1),
            PAY_WAY2 = sum(MAIN_PAY_WAY==2),
            PAY_WAY3 = sum(MAIN_PAY_WAY==3),
            PAY_WAY4 = sum(MAIN_PAY_WAY==4),
            PAY_WAY5 = sum(MAIN_PAY_WAY==6))

SALE_CHANNEL = df1_test %>% group_by(CUST_ID) %>% 
  summarise(SALE_CHANNEL1 = sum(MAIN_SALE_CHANNEL==0),
            SALE_CHANNEL2 = sum(MAIN_SALE_CHANNEL==1),
            SALE_CHANNEL3 = sum(MAIN_SALE_CHANNEL==2),
            SALE_CHANNEL4 = sum(MAIN_SALE_CHANNEL==3),
            SALE_CHANNEL5 = sum(MAIN_SALE_CHANNEL==4))

BUY_YEAR  = df1_test %>% group_by(CUST_ID) %>% 
  summarise(BUY_YEAR1 = sum(BUY_YEAR==2006),
            BUY_YEAR2 = sum(BUY_YEAR==2007),
            BUY_YEAR3 = sum(BUY_YEAR==2008),
            BUY_YEAR4 = sum(BUY_YEAR==2009),
            BUY_YEAR5 = sum(BUY_YEAR==2010),
            BUY_YEAR6 = sum(BUY_YEAR==2011),
            BUY_YEAR7 = sum(BUY_YEAR==2012),
            BUY_YEAR8 = sum(BUY_YEAR==2013),
            BUY_YEAR9 = sum(BUY_YEAR==2014),
            BUY_YEAR10 = sum(BUY_YEAR==2015),
            BUY_YEAR11 = sum(BUY_YEAR==2016),
            BUY_YEAR12 = sum(BUY_YEAR==2017),
            BUY_YEAR13 = sum(BUY_YEAR==2018) )

# MAIN_PERIOD
add1 = df1_test$MAIN_PERIOD
add1[add1<20] = 1
add1[(add1>=20) & (add1<40) ] = 2
add1[(add1>=40) & (add1<95) ] = 3
add1[(add1>=95) & (add1<200) ] = 4
add1[(add1>=200)] = 5
df1_test$add1 = add1

MAIN_PERIOD =  df1_test %>% group_by(CUST_ID) %>% 
  summarise(MAIN_PERIOD1 = sum(add1==1),
            MAIN_PERIOD2 = sum(add1==2),
            MAIN_PERIOD3 = sum(add1==3),
            MAIN_PERIOD4 = sum(add1==4),
            MAIN_PERIOD5 = sum(add1==5))

# MAIN_PAY_PERIOD
add2 = df1_test$MAIN_PAY_PERIOD
add2[add2<6] = 1
add2[(add2>=6) & (add2<14) ] = 2
add2[(add2>=14) & (add2<19) ] = 3
add2[(add2>=19) & (add2<24) ] = 4
add2[(add2>=24) & (add2<40) ] = 5
add2[(add2>=40) & (add2<95) ] = 6
add2[(add2>=95) & (add2<101) ] = 7
add2[(add2>=101)] = 8
df1_test$add2 = add2

MAIN_PAY_PERIOD =  df1_test %>% group_by(CUST_ID) %>% 
  summarise(MAIN_PAY_PERIOD1 = sum(add2==1),
            MAIN_PAY_PERIOD2 = sum(add2==2),
            MAIN_PAY_PERIOD3 = sum(add2==3),
            MAIN_PAY_PERIOD4 = sum(add2==4),
            MAIN_PAY_PERIOD5 = sum(add2==5),
            MAIN_PAY_PERIOD6 = sum(add2==6),
            MAIN_PAY_PERIOD7 = sum(add2==7),
            MAIN_PAY_PERIOD8 = sum(add2==8))
rm(add1,add2)

# 合併新變數
X1 = left_join(X,unit_7)
X2 = left_join(X1,amount_sum)
X3 = left_join(X2,CURRENCY_sum)
X4 = left_join(X3,PREMIUM_sum)
X5 = left_join(X4,PAY_FREQ)
X6 = left_join(X5,PAY_WAY)
X7 = left_join(X6,SALE_CHANNEL)
X8 = left_join(X7,BUY_YEAR)
X9 = left_join(X8,MAIN_PERIOD)
X20 = left_join(X9,MAIN_PAY_PERIOD)
rm(X1,X2,X3,X4,X5,X6,X7,X8,X9)


colSums(is.na(X20))[(colSums(is.na(X20))>0) &(colSums(is.na(X20))<3983)]
colSums(is.na(X20))[(colSums(is.na(X20))==3983)]

X20 = data.frame(X20)
for(i in 38:98) {
  X20[is.na(X20[,i]),i] = 0
}

# dummy化
model_test = cbind(X20[,-c(12:15,22,23)],
                   SMOKES=dummy(X20$SMOKES)[,-1],
                   ARCEA=dummy(X20$ARCEA)[,-1],
                   LIQUEUR=dummy(X20$LIQUEUR)[,-1],
                   EDUCATION=dummy(X20$EDUCATION)[,-1],
                   CAREER=dummy(X20$CAREER)[,-1],
                   VETERAN_STATUS=dummy(X20$VETERAN_STATUS)[,-1])

for (i in 1:ncol(model_test)) {
  model_test[,i] = as.numeric(model_test[,i])
}



model$CAREER.X1022 = 0


################################
# 最終model之解釋變數以及反應變數
## final model
# X_train
model = model[c(1:126,131,127:130)] 
# y_train
y_test

# X_test
model_test
# y_test(預測)
y_test
#################################

###附約###

# 附約
df2_train
df2_test

# 主約(full data)
model
model_test


# 附約之中類轉大類
small = function(i){
  if (i == 7|i == 6|i == 9|i == 10){return(1)}
  else if (i == 17){return(2)}
  else if (i == 12){return(3)}
}


df2_train = df2_train %>% mutate(small_TYPE = sapply(ADD_PRODUCT_TYPE,small))
df2_test  = df2_test  %>% mutate(small_TYPE = sapply(ADD_PRODUCT_TYPE,small))

df2 = full_join(df2_train,df2_test)

########################################################
# 將保額單位為7者獨立出來做累計
add_unit_7 = df2 %>% mutate(time = BUY_YEAR+BUY_MONTH/13)%>%  group_by(CUST_ID) %>%
  summarise(add_unit7 = sum((ADD_AMOUNT_UNIT==7)*(time!=max(time)) ))

# 將單位換成實際值
df2$ADD_AMOUNT_UNIT[df2$ADD_AMOUNT_UNIT==1] = 1
df2$ADD_AMOUNT_UNIT[df2$ADD_AMOUNT_UNIT==3] = 1000
df2$ADD_AMOUNT_UNIT[df2$ADD_AMOUNT_UNIT==4] = 10000
df2$ADD_AMOUNT_UNIT[df2$ADD_AMOUNT_UNIT==7] = 0

# 過去紀錄之附約保額累計(要乘保額單位)
df2%>% mutate(time = BUY_YEAR+BUY_MONTH/13)%>% group_by(CUST_ID) %>% 
  summarise(ADD_AMOUNT_SUM1 = sum((small_TYPE==1)*ADD_AMOUNT*ADD_AMOUNT_UNIT*(time!=max(time))),
            ADD_AMOUNT_SUM2 = sum((small_TYPE==2)*ADD_AMOUNT*ADD_AMOUNT_UNIT*(time!=max(time))),
            ADD_AMOUNT_SUM3 = sum((small_TYPE==3)*ADD_AMOUNT*ADD_AMOUNT_UNIT*(time!=max(time))))

# 過去紀錄之附約保費累計
add_premium_sum = df2 %>% mutate(time = BUY_YEAR+BUY_MONTH/13) %>% group_by(CUST_ID) %>%
  summarise(ADD_PREMIUM_SUM1 = sum((small_TYPE==1)*ADD_PREMIUM*(time!=max(time))),
            ADD_PREMIUM_SUM2 = sum((small_TYPE==2)*ADD_PREMIUM*(time!=max(time))),
            ADD_PREMIUM_SUM3 = sum((small_TYPE==3)*ADD_PREMIUM*(time!=max(time))))


# 連續型處理 ADD_PAY_PERIOD ADD_PERIOD ADD_GIVE_PERIOD
############## 觀察繳費年期與保險年期與附約大類的關係以便做切割
# 切2,5,9,14,40,59,62,69
table("type"= df2$small_TYPE,"period"=df2$ADD_PAY_PERIOD)

# 切2,6.5,9,14,54,59,90,101
table("type"= df2$small_TYPE,"period"=df2$ADD_PERIOD)

# 不考慮
table("type"= df2$small_TYPE,"period"=df2$ADD_GIVE_PERIOD)

table(df2$ADD_PAY_PERIOD,df2$small_TYPE)
plot(df2$ADD_PAY_PERIOD,df2$small_TYPE,xlab="附約繳費年期",ylab="附約購買大類",main="附約繳費年期vs附約購買大類")
abline(v=2,col="blue")
abline(v=5,col="blue")
abline(v=9,col="blue")
abline(v=14,col="blue")
abline(v=40,col="blue")
abline(v=59,col="blue")
abline(v=62,col="blue")
abline(v=69,col="blue")

plot(df2$ADD_PAY_PERIOD,df2$small_TYPE,xlab="附約保險年期",ylab="附約購買大類",main="附約保險年期vs附約購買大類")
abline(v=2,col="blue")
abline(v=6.5,col="blue")
abline(v=9,col="blue")
abline(v=14,col="blue")
abline(v=54,col="blue")
abline(v=59,col="blue")
abline(v=90,col="blue")
abline(v=101,col="blue")
##############
# 附約繳費年期做切割
# 切2,5,9,14,40,59,62,69
ADD_1 = df2$ADD_PAY_PERIOD
ADD_1[ADD_1<2] = 1
ADD_1[(ADD_1>=2) & (ADD_1<5) ] = 2
ADD_1[(ADD_1>=5) & (ADD_1<9) ] = 3
ADD_1[(ADD_1>=9) & (ADD_1<14) ] = 4
ADD_1[(ADD_1>=14) & (ADD_1<40) ] = 5
ADD_1[(ADD_1>=40) & (ADD_1<59) ] = 6
ADD_1[(ADD_1>=59) & (ADD_1<62) ] = 7
ADD_1[(ADD_1>=62) & (ADD_1<69) ] = 8
ADD_1[(ADD_1>=69)] = 9
df2$ADD_1 = ADD_1

# 過去紀錄之繳費年期紀錄累計
ADD_PAY_PERIOD_sum =  df2 %>% mutate(time = BUY_YEAR+BUY_MONTH/13) %>% group_by(CUST_ID) %>% 
  summarise(ADD_PAY_PERIOD_1 = sum((ADD_1==1)*(time!=max(time))),
            ADD_PAY_PERIOD_2 = sum((ADD_1==2)*(time!=max(time))),
            ADD_PAY_PERIOD_3 = sum((ADD_1==3)*(time!=max(time))),
            ADD_PAY_PERIOD_4 = sum((ADD_1==4)*(time!=max(time))),
            ADD_PAY_PERIOD_5 = sum((ADD_1==5)*(time!=max(time))),
            ADD_PAY_PERIOD_6 = sum((ADD_1==6)*(time!=max(time))),
            ADD_PAY_PERIOD_7 = sum((ADD_1==7)*(time!=max(time))),
            ADD_PAY_PERIOD_8 = sum((ADD_1==8)*(time!=max(time))),
            ADD_PAY_PERIOD_9 = sum((ADD_1==9)*(time!=max(time))))


# 附約保險年期做切割
# 切2,6.5,9,14,54,59,90,101
ADD_2 = df2$ADD_PERIOD
ADD_2[ADD_2<2] = 1
ADD_2[(ADD_2>=2) & (ADD_2<6.5) ] = 2
ADD_2[(ADD_2>=6.5) & (ADD_2<9) ] = 3
ADD_2[(ADD_2>=9) & (ADD_2<14) ] = 4
ADD_2[(ADD_2>=14) & (ADD_2<54) ] = 5
ADD_2[(ADD_2>=54) & (ADD_2<59) ] = 6
ADD_2[(ADD_2>=59) & (ADD_2<90) ] = 7
ADD_2[(ADD_2>=90) & (ADD_2<101) ] = 8
ADD_2[(ADD_2>=101)] = 9
df2$ADD_2 = ADD_2

# 過去紀錄之保險年期紀錄累計
ADD_PERIOD_sum =  df2 %>% mutate(time = BUY_YEAR+BUY_MONTH/13) %>% group_by(CUST_ID) %>% 
  summarise(ADD_PERIOD_1 = sum((ADD_2==1)*(time!=max(time))),
            ADD_PERIOD_2 = sum((ADD_2==2)*(time!=max(time))),
            ADD_PERIOD_3 = sum((ADD_2==3)*(time!=max(time))),
            ADD_PERIOD_4 = sum((ADD_2==4)*(time!=max(time))),
            ADD_PERIOD_5 = sum((ADD_2==5)*(time!=max(time))),
            ADD_PERIOD_6 = sum((ADD_2==6)*(time!=max(time))),
            ADD_PERIOD_7 = sum((ADD_2==7)*(time!=max(time))),
            ADD_PERIOD_8 = sum((ADD_2==8)*(time!=max(time))),
            ADD_PERIOD_9 = sum((ADD_2==9)*(time!=max(time))))


as.tibble(model)
as.tibble(model_test)

# 合併新變數以及原本主約的解釋變數
# XX_train
XX1 = left_join(model,y_test)
XX2 = left_join(XX1,add_amount_sum)
XX3 = left_join(XX2,add_premium_sum)
XX4 = left_join(XX3,ADD_PAY_PERIOD_sum)
XX5 = left_join(XX4,ADD_PERIOD_sum)
XX6 = left_join(XX5,add_unit_7)
rm(XX1,XX2,XX3,XX4,XX5)
XX6[which(is.na(XX6$add_unit7) ),133:157] = 0

# XX_test (注意BIGTYPE 要帶之前預測值)
XX1_ = left_join(model_test,y_test)
XX2_ = left_join(XX1_,add_amount_sum)
XX3_ = left_join(XX2_,add_premium_sum)
XX4_ = left_join(XX3_,ADD_PAY_PERIOD_sum)
XX5_ = left_join(XX4_,ADD_PERIOD_sum)
XX6_ = left_join(XX5_,add_unit_7)
rm(XX1_,XX2_,XX3_,XX4_,XX5_)
XX6_[which(is.na(XX6_$add_unit7) ),133:157] = 0


df2 %>% group_by(CUST_ID) %>% select(CUST_ID,small_TYPE,BUY_YEAR,BUY_MONTH) %>% 
  arrange(CUST_ID,desc(BUY_YEAR),desc(BUY_MONTH))

# 三大類最新一筆是否有購買
# y1,y2,y3
yyy = df2%>% mutate(time = BUY_YEAR+BUY_MONTH/13) %>% select(CUST_ID,small_TYPE,time) %>% 
  arrange(desc(CUST_ID),desc(time)) %>%  group_by(CUST_ID) %>% 
  summarise( BUY_TYPE_1 =  as.numeric( sum((time==max(time))*(small_TYPE==1)) >0.5) ,
             BUY_TYPE_2 =  as.numeric( sum((time==max(time))*(small_TYPE==2)) >0.5) ,
             BUY_TYPE_3 =  as.numeric( sum((time==max(time))*(small_TYPE==3)) >0.5) )


left_join(select(XX6,CUST_ID),yyy) %>% dim


y1_train = left_join(select(XX6,CUST_ID),yyy)$BUY_TYPE_1
y1_train[is.na(y1_train)] = 0

y2_train = left_join(select(XX6,CUST_ID),yyy)$BUY_TYPE_2
y2_train[is.na(y2_train)] = 0

y3_train = left_join(select(XX6,CUST_ID),yyy)$BUY_TYPE_3
y3_train[is.na(y3_train)] = 0

##########以下為cross validation找最佳參數#########

library(xgboost)

# model 1
xgb.param1 = list(subsample=0.8,colsample_bytree=0.8,max_depth=5,eta=0.03)

xgbcv1 = xgb.cv(params = xgb.param1,
                data= as.matrix(XX6)  ,
                label = y1_train ,
                nfold = 8,nrounds = 100,
                eval_metric="error",objective="binary:logistic")


# model 2   "a= 0.05 ,b= 9 ,test_error= 0.1503705"
for(a in c(0.03,0.04,0.05,0.06)){
  for (b in c(10,11,12,13,14) ) {
    xgb.param2 = list(subsample=0.8,colsample_bytree=0.8,max_depth=b,eta=a)
    xgbcv2 = xgb.cv(params = xgb.param2,
                    data= as.matrix(XX6)  ,
                    label = y2_train ,verbose = F,
                    nfold = 8,nrounds = 100,
                    eval_metric="error",objective="binary:logistic")
    print(paste("a=",a,",b=",b,",test_error=",min(xgbcv2$evaluation_log$test_error_mean) ))
    print("---------------------------------------------------")
  }
}

# model 3  
#"a= 0.04 ,b= 15 ,test_error= 0.173952
for(a in c(0.04)){
  for (b in c(15) ) {
    xgb.param3 = list(subsample=0.8,colsample_bytree=0.8,max_depth=b,eta=a)
    xgbcv3 = xgb.cv(params = xgb.param3,
                    data= as.matrix(XX6)  ,
                    label = y3_train ,print_every_n = 10,
                    nfold = 8,nrounds = 100,
                    eval_metric="error",objective="binary:logistic")
    print(paste("a=",a,",b=",b,",test_error=",min(xgbcv3$evaluation_log$test_error_mean) ))
    print("---------------------------------------------------")
  }
}

####### 最終模型 ########

# final result (main)
K1 = as.tibble(data.frame(CUST_ID=model_test$CUST_ID,"pred"= pred_main))


K2 = as.tibble(data.frame(CUST_ID=XX6_$CUST_ID,"pred1"= pred_add1))
K3 = as.tibble(data.frame(CUST_ID=XX6_$CUST_ID,"pred2"= pred_add2))
K4 = as.tibble(data.frame(CUST_ID=XX6_$CUST_ID,"pred3"= pred_add3))


KK = left_join(K1,K2)
KK = left_join(KK,K3)
KK = left_join(KK,K4)

AA = left_join(as.tibble(select(df4,POLICY_ID,CUST_ID)),KK,by="CUST_ID") 
colnames(AA) = colnames(df4)

write.csv(AA,"aa.csv")

#### 隨機抽取驗證集
valid = sample(1:nrow(XX6),floor(nrow(XX6)*0.7 ))

##################################################################
colnames(XX6_) = colnames(XX6) 

#######以下為利用最佳參數跑model#######
# model 1 
xgb.param1 = list(subsample=0.8,colsample_bytree=0.8,max_depth=15,eta=0.03)

xgbcv1 = xgb.cv(params = xgb.param1,
                data= as.matrix(XX6)  ,
                label = y1_train ,
                nfold = 5,nrounds = 100,
                eval_metric="error",objective="binary:logistic")

xgb1 = xgboost(xgb.param1,data= as.matrix(XX6)  ,
               label = y1_train,
               nrounds = 300 ,
               eval_metric="error",objective="binary:logistic")





# model 2   "a= 0.05 ,b= 9 ,test_error= 0.1503705"
xgb.param2 = list(subsample=0.8,colsample_bytree=0.8,max_depth=9,eta=0.05)
xgbcv2 = xgb.cv(params = xgb.param2,
                data= as.matrix(XX6)  ,
                label = y2_train ,verbose = F,
                nfold = 8,nrounds = 100,
                eval_metric="error",objective="binary:logistic")

xgb2 = xgboost(xgb.param2,data= as.matrix(XX6)  ,
               label = y2_train,
               nrounds = 300 ,
               eval_metric="error",objective="binary:logistic")


pred_add2 = as.numeric(predict(xgb2,as.matrix(XX6_))>=0.5)



# model 3  
#"a= 0.04 ,b= 15 ,test_error= 0.173952
xgb.param3 = list(subsample=0.8,colsample_bytree=0.8,max_depth=15,eta=0.04)
xgbcv3 = xgb.cv(params = xgb.param3,
                data= as.matrix(XX6)  ,
                label = y3_train ,print_every_n = 10,
                nfold = 8,nrounds = 100,
                eval_metric="error",objective="binary:logistic")


xgb3 = xgboost(xgb.param3,data= as.matrix(XX6)  ,
               label = y3_train,
               nrounds = 300 ,
               eval_metric="error",objective="binary:logistic")


pred_add3 = as.numeric(predict(xgb3,as.matrix(XX6_))>=0.5)









###############################
###############################
###############################
###############################
###############################
#valid 
############以下為對驗證集做預測#############

### 附約
xgb__1 = xgboost(xgb.param1,data= as.matrix(XX6[valid,])  ,
                 label = y1_train[valid],
                 nrounds = 300 ,
                 eval_metric="error",objective="binary:logistic")

### confusion matrix ###
table("true"=y1_train[-valid],as.numeric(predict(xgb__1,as.matrix(XX6[-valid,])  )>0.5) )



xgb__2 = xgboost(xgb.param2,data= as.matrix(XX6[valid,])  ,
                 label = y2_train[valid],
                 nrounds = 300 ,
                 eval_metric="error",objective="binary:logistic")

### confusion matrix ###
table("true"=y2_train[-valid],as.numeric(predict(xgb__2,as.matrix(XX6[-valid,])  )>0.5) )


xgb__3 = xgboost(xgb.param3,data= as.matrix(XX6[valid,])  ,
                 label = y3_train[valid],
                 nrounds = 300 ,
                 eval_metric="error",objective="binary:logistic")

### confusion matrix ###
table("true"=y3_train[-valid],as.numeric(predict(xgb__3,as.matrix(XX6[-valid,])  )>0.5) )



### 主約

xgb__ = xgboost(xgb.param,data= as.matrix(model[valid,])  ,
              label = left_join(select(model,CUST_ID),y_test)$BIG_TYPE[valid] ,
              nrounds = 300 ,
              eval_metric="merror",objective="multi:softmax",num_class=8)

### confusion matrix ###
table("true"=left_join(select(model,CUST_ID),y_test)$BIG_TYPE[-valid],
      "pred"=(predict(xgb__,as.matrix(model[-valid,]))))

### 變數重要性 ###
importance_matrix1 = xgb.importance(model=xgb__)
print(importance_matrix1)
xgb.plot.importance(importance_matrix = importance_matrix1[c(1:5)])

importance_matrix2 = xgb.importance(model=xgb__1)
print(importance_matrix2)
xgb.plot.importance(importance_matrix = importance_matrix2[1:5])

importance_matrix3 = xgb.importance(model=xgb__2)
print(importance_matrix3)
xgb.plot.importance(importance_matrix = importance_matrix3[1:5])

importance_matrix4 = xgb.importance(model=xgb__3)
print(importance_matrix4)
xgb.plot.importance(importance_matrix = importance_matrix4[1:5])


 
