```{r code, eval=FALSE}

require(data.table)
require(TunePareto)
require(glmnet)
library(dplyr)
library(caret)


setwd("C:/Users/dsa/Desktop/582 Dosyalar/582 Proje")

testStart=as.Date('2018-12-21')
trainStart=as.Date('2012-07-15')
rem_miss_threshold=0.01 #parameter for removing bookmaker odds with missing ratio greater than this threshold

#Functions provided by our instructor

source('data_preprocessing.r')
source('feature_extraction.r')
source('performance_metrics.r')
source('train_models.r')


# read data
setwd("C:/Users/dsa/Downloads")
matches_raw=readRDS("df9b1196-e3cf-4cc7-9159-f236fe738215_matches.rds")
odd_details_raw=readRDS("df9b1196-e3cf-4cc7-9159-f236fe738215_odd_details.rds")


# preprocess matches
matches=matches_data_preprocessing(matches_raw)

# preprocess odd data
odd_details=details_data_preprocessing(odd_details_raw,matches)
odd_details[Match_Date>testStart]

# extract open and close odd type features from multiple bookmakers
features=extract_features.openclose(matches,odd_details,pMissThreshold=rem_miss_threshold,trainStart,testStart)


# divide data based on the provided dates 
matchinfo=matches[,c(2,3,4,5,10)]
matchinfo=matchinfo[complete.cases(matchinfo)]

homescore=rep(0,nrow(matchinfo))
awayscore=rep(0,nrow(matchinfo))

matchinfo1=cbind(matchinfo,homescore,awayscore)

for(i in 1:nrow(matchinfo1)){
  if(matchinfo1$Match_Result[i]=="Tie")
  {
    matchinfo1$homescore[i]=1
    matchinfo1$awayscore[i]=1
  }
  if(matchinfo1$Match_Result[i]=="Home")
  {
    matchinfo1$homescore[i]=2
    matchinfo1$awayscore[i]=0
  }
  if(matchinfo1$Match_Result[i]=="Away")
  {
    matchinfo1$homescore[i]=0
    matchinfo1$awayscore[i]=2
  }
}

#Finding scores of the teams from their last 5 games

teams_unique<-unique(as.vector(as.matrix(matches[,c("Home", "Away")])))
teams_unique<-as.factor(teams_unique)

df=data.frame()

for(i in 1:length(teams_unique)){
  matchfiltered=matchinfo1%>%filter(Home==teams_unique[i]|Away==teams_unique[i])%>% arrange(desc(Match_Date))
  dim(matchfiltered)
  row_number=nrow(matchfiltered)-5
  
  last5=rep(NA,nrow(matchfiltered))
  
  for(j in 1:row_number)
  {
    score=0
    for(k in 1:5)
    {
      if(matchfiltered[(j+k),]$Home==teams_unique[i])
      {
        score=score+matchfiltered[(j+k),]$homescore
      }
      else if(matchfiltered[(j+k),]$Away==teams_unique[i])
      {
        score=score+matchfiltered[(j+k),]$awayscore
      }
    }
    matchfiltered$last5[j]=score
  }
  matchfiltered$last5[(row_number+1):nrow(matchfiltered)]=0
  last5_dummy=cbind(matchfiltered$matchId,as.character(teams_unique[i]),matchfiltered$last5)
  df=rbind(df,last5_dummy)
}

colnames(df)=c("matchId","team","last5_score")

df$matchId=as.character(df$matchId)
df$team=as.character(df$team)

matches_new=matches%>%left_join(df,by=c("matchId","Home"="team"))
colnames(matches_new)[15]=c("home_last5score")
matches_new2=matches_new%>%left_join(df,by=c("matchId","Away"="team"))
colnames(matches_new2)[16]=c("away_last5score")


features=merge(features,matches_new2[,c(2,15,16)],by="matchId")

##

train_features=features[Match_Date>=trainStart & Match_Date<testStart] 
test_features=features[Match_Date>=testStart] 

# run glmnet on train data with tuning lambda parameter based on RPS and return predictions based on lambda with minimum RPS
#predictions=train_glmnet(train_features, test_features,not_included_feature_indices=c(1:5), alpha=1,nlambda=50, tune_lambda=TRUE,nofReplications=2,nFolds=10,trace=T)

train_features=train_features[complete.cases(train_features)]
#test_features=test_features[complete.cases(test_features)]

traindata_main=train_features[,-c(1:5)]
testdata_main = test_features[,-c(1:5)]
train.class_main=train_features$Match_Result

traindata_withclass=cbind(traindata_main,train.class_main)
names(traindata_withclass)
library(randomForest)
set.seed(123)
#res <- tuneRF(x=traindata_withclass[,-93], y=as.factor(traindata_withclass$train.class_main),mtryStart=10,ntreeTry=500,stepFactor=2.5,improve=0.01)
#saveRDS(res,"rf_proje.rds")
res<-readRDS("rf_proje.rds")
# Look at results
print(res)

# Find the mtry value that minimizes OOB Error
mtry_opt <- res[,"mtry"][which.min(res[,"OOBError"])]

#10 fold Cross Validation

fold_indices=generateCVRuns(as.factor(traindata_withclass$train.class_main),1,10,stratified=TRUE)
errs_rf<-rep(NA,10)

# for (i in 1:10) {
#   Replication=fold_indices[[1]]
#   testindices=Replication[[i]]
#   train=traindata_withclass[-testindices,]        
#   test=traindata_withclass[testindices,]
#   train_class<-as.factor(train$train.class_main)
#  
#   rf_model = randomForest(train[,-93],train_class,ntree=500,mtry = mtry_opt,nodesize=5) 
#   predict_rf=predict(rf_model,test,type="class")
#   conf.mat_rf<-table(as.factor(test$train.class_main),predict_rf)
#   errs_rf[i] <- 1-sum(diag(conf.mat_rf))/sum(conf.mat_rf)
# }
# saveRDS(errs_rf,"errors_rf.rds")
errs_rf<-readRDS("errors_rf.rds")
print(paste("10-fold Cross Validation Error:",mean(errs_rf)))

model2 <- randomForest(traindata_withclass[,-93],as.factor(traindata_withclass$train.class_main),ntree=500,mtry=mtry_opt,nodesize=5)

pred <- predict(model2,traindata_withclass[,-93],type="prob")

## predict testfeatures
test_feat=merge(test_features,matches[,2:4],by="matchId")

#Fill the NA Scores in Testdata with the Last Scores of Home and Away Teams

for(i in 1:nrow(test_feat)){
  if(is.na(test_feat$home_last5score[i])){
    temp<-df%>%filter(team==test_feat$Home[i])
    test_feat$home_last5score[i]=temp$last5_score[1]
    temp<-df%>%filter(team==test_feat$Away[i])
    test_feat$away_last5score[i]=temp$last5_score[1]
  }
}


testdata_withoutclass = test_feat[,-c(1:5,98,99)]

pred <- predict(model2,testdata_withoutclass,type="class")

table(as.factor(test_features$Match_Result),pred)
pred.prob <- predict(model2,testdata_withoutclass,type="prob")
pred.prob


## train class labels considered as output
train_class=data.table(finaltable2)
train_class[,pred_id:=1:.N]
train_class_outcomes=data.table::dcast(train_class,pred_id~Match_Result,value.var='pred_id')

train_class_outcomes[,pred_id:=NULL]
train_class_outcomes[is.na(train_class_outcomes)]=0
train_class_outcomes[train_class_outcomes>0]=1
setcolorder(train_class_outcomes,c('Home','Tie','Away'))

##using RPS
RPS_matrix<- function(probs,outcomes){
  probs=as.matrix(probs)
  outcomes=as.matrix(outcomes)
  probs=t(apply(t(probs), 2, cumsum))
  outcomes=t(apply(t(outcomes), 2, cumsum))
  RPS = apply((probs-outcomes)^2,1,sum) / (ncol(probs)-1)
  return(RPS)
}

##RPS results for ordered
RPS_results=RPS_matrix(pred.prob[,c(2,3,1)],train_class_outcomes)
RPS_results=data.table(RPS_results)
mean(RPS_results$RPS_results)

RPS=predictions$cv_stats$meanRPS_min
RPS

#Poisson Model


library(skellam)
setwd("C:/Users/dsa/Downloads")
matches=readRDS("df9b1196-e3cf-4cc7-9159-f236fe738215_matches.rds")

odd_details=readRDS("df9b1196-e3cf-4cc7-9159-f236fe738215_odd_details.rds")

matches=unique(matches)

matches<-data.table(matches)
odd_details<-data.table(odd_details)
#transform unix time to date time

matches[,Match_DateTime:=as.POSIXct(date,tz="UTC",origin = as.POSIXct("1970-01-01",tz="UTC"))]
matches[,Match_Hour := format(strptime(Match_DateTime,"%Y-%m-%d %H:%M:%OS"),'%H')]
matches[,Match_Hour := as.numeric(Match_Hour)]
matches[,Match_Date := as.Date(Match_DateTime,format="%Y-%m-%d")]


matches[,c("HomeGoals","AwayGoals"):=tstrsplit(score,':')]


#transform characters to numeric for scores
matches$HomeGoals=as.numeric(matches$HomeGoals)
matches[,AwayGoals:=as.numeric(AwayGoals)]

new<-matches%>%select(matchId,Match_DateTime,home,away,HomeGoals,AwayGoals)%>%filter(!is.na(HomeGoals))
new<-new%>%filter(home!=c("manchester united","manchester-utd","manchester city","crystal palace")&&away!=c("manchester united","manchester-utd","manchester city","crystal palace"))
deneme<-new%>%filter(!home %in% c("manchester united","manchester-utd","manchester city","crystal palace")&away==c("manchester united","manchester-utd","manchester city","crystal palace"))
testdata<-matches%>%select(matchId,Match_DateTime,home,away,HomeGoals,AwayGoals)%>%filter(is.na(HomeGoals))
skellam::dskellam(0,mean(new$HomeGoals),mean(new$AwayGoals))

data.frame(avg_home_goals = mean(new$HomeGoals),
           avg_away_goals = mean(new$AwayGoals))

poisson_model <- 
  rbind(
    data.frame(goals=new$HomeGoals,
               team=new$home,
               opponent=new$away,
               home=1),
    data.frame(goals=new$AwayGoals,
               team=new$away,
               opponent=new$home,
               home=0)) %>%
  glm(goals ~ home + team +opponent, family=poisson(link=log),data=.)
summary(poisson_model)

simulate_match <- function(foot_model, homeTeam, awayTeam, max_goals=10){
  home_goals_avg <- predict(foot_model,
                            data.frame(home=1, team=homeTeam, 
                                       opponent=awayTeam), type="response")
  away_goals_avg <- predict(foot_model, 
                            data.frame(home=0, team=awayTeam, 
                                       opponent=homeTeam), type="response")
  dpois(0:max_goals, home_goals_avg) %o% dpois(0:max_goals, away_goals_avg) 
}

new<-new%>%mutate(home_win_prob=NA,away_win_prob=NA,tie_prob=NA)

for(i in 1:nrow(new)){
  temp<-simulate_match(poisson_model, new$home[i], new$away[i], max_goals=10)
  new$home_win_prob[i]<-sum(temp[lower.tri(temp)])
  new$away_win_prob[i]<-sum(temp[upper.tri(temp)])
  new$tie_prob[i]<-sum(diag(temp))
}

new<-new%>%mutate(result_guess=NA)
for(i in 1:nrow(new)){
  if(new$home_win_prob[i]>new$away_win_prob[i]&new$home_win_prob[i]>new$tie_prob[i]){
    new$result_guess[i]="Home"
  }else if(new$away_win_prob[i]>new$home_win_prob[i]&new$away_win_prob[i]>new$tie_prob[i]){
    new$result_guess[i]="Away"
  }else if(new$tie_prob[i]>new$home_win_prob[i]&new$tie_prob[i]>new$away_win_prob[i]){
    new$result_guess[i]="Tie"
  }
  
}


testdata<-testdata%>%mutate(home_win_prob=NA,away_win_prob=NA,tie_prob=NA)

for(i in 1:nrow(testdata)){
  temp<-simulate_match(poisson_model, testdata$home[i], testdata$away[i], max_goals=4)
  testdata$home_win_prob[i]<-sum(temp[lower.tri(temp)])
  testdata$away_win_prob[i]<-sum(temp[upper.tri(temp)])
  testdata$tie_prob[i]<-sum(diag(temp))
}

testdata<-testdata%>%mutate(result_guess=NA)
for(i in 1:nrow(testdata)){
  if(testdata$home_win_prob[i]>testdata$away_win_prob[i]&testdata$home_win_prob[i]>testdata$tie_prob[i]){
    testdata$result_guess[i]="Home"
  }else if(testdata$away_win_prob[i]>testdata$home_win_prob[i]&testdata$away_win_prob[i]>testdata$tie_prob[i]){
    testdata$result_guess[i]="Away"
  }else if(testdata$tie_prob[i]>testdata$home_win_prob[i]&testdata$tie_prob[i]>testdata$away_win_prob[i]){
    testdata$result_guess[i]="Tie"
  }
  
}

odd_test<-odd_details%>%filter(betType=="1x2",bookmaker=="bet365",oddtype==c("odd1","odd2","oddX"))%>%select(matchId,odd,oddtype,date)
odd_test<-data.table(odd_test)
odd_test[,odd_DateTime:=as.POSIXct(date,tz="UTC",origin = as.POSIXct("1970-01-01",tz="UTC"))]
odd_test[,odd_Hour := format(strptime(odd_DateTime,"%Y-%m-%d %H:%M:%OS"),'%H')]
odd_test[,odd_Hour := as.numeric(odd_Hour)]
odd_test[,odd_Date := as.Date(odd_DateTime,format="%Y-%m-%d")]

feature_odd_details=odd_test[,list(Odd_Open=odd[1],Odd_Close=odd[.N]),list(matchId,oddtype)]

feature_odd_details_test = merge(testdata, feature_odd_details,by="matchId")

feature_odd_details_test<-feature_odd_details_test%>%mutate(prob_bookmaker=1/Odd_Close)

details<-dcast(feature_odd_details_test,matchId~oddtype,value.var = c("prob_bookmaker"))

final<-merge(feature_odd_details_test[,1:10],details,by="matchId")

final<-unique(final)

final<-final%>%mutate(result_bookmaker=NA)
final<-final%>%arrange(Match_DateTime)
for(i in 1:nrow(final)){
  if(is.na(final$odd1[i])){
    final$result_bookmaker[i]="Away"
  }else if(is.na(final$odd2[i])){
    final$result_bookmaker[i]="Home"
  }else if(is.na(final$oddX[i])){
    if(final$odd1[i]>final$odd2[i]){
      final$result_bookmaker[i]="Home"
    }else{
      final$result_bookmaker[i]="Away"
    }
  }else if(final$odd1[i]>final$odd2[i]&final$odd1[i]>final$oddX[i]){
    final$result_bookmaker[i]="Home"
  }else if(final$odd2[i]>final$odd1[i]&final$odd2[i]>final$oddX[i]){
    final$result_bookmaker[i]="Away"
  }else if(final$oddX[i]>final$odd1[i]&final$oddX[i]>final$odd2[i]){
    final$result_guess[i]="Tie"
  }
  
}


colnames(new)[7:9]<-c("Home","Away","Tie")

train_class<-new$result_guess
train_class=data.table(train_class)
train_class[,pred_id:=1:.N]
train_class_outcomes=data.table::dcast(train_class,pred_id~train_class,value.var='pred_id')
train_class_outcomes[,pred_id:=NULL]
train_class_outcomes[is.na(train_class_outcomes)]=0
train_class_outcomes[train_class_outcomes>0]=1
setcolorder(train_class_outcomes,c('Home','Tie','Away'))

##using RPS
RPS_matrix<- function(probs,outcomes){
  probs=as.matrix(probs)
  outcomes=as.matrix(outcomes)
  probs=t(apply(t(probs), 2, cumsum))
  outcomes=t(apply(t(outcomes), 2, cumsum))
  RPS = apply((probs-outcomes)^2,1,sum) / (ncol(probs)-1)
  return(RPS)
}