import pandas as pd 
import numpy as np
def load_data():
    return pd.read_csv('../data/train.csv')

def clean_data(data_df):
    data_df.duplicated().sum()
     ##0 count so no need for dropping duplicated rows
    data_df.drop_duplicates(inplace=True) 
    data_df['Fare']=pd.to_numeric(data_df['Fare'], errors='coerce') ##checking for nan values in Fare if added 
    data_df['Fare'].notna().value_counts() ##since no false values we have no nan's introduced
    data_df.drop(columns=['Cabin'],inplace=True) ##deletes 'Cabin' column from data_df and updates it on the place 
    overall_median=data_df['Age'].median() ##Age missing values  handled 
    data_df['Age'] = data_df['Age'].fillna(
    data_df.groupby(['Pclass', 'Sex'])['Age'].transform('median'))
##to handle nans now 
    data_df['Age'].fillna(overall_median,inplace=True)
    modes=data_df['Embarked'].mode() ##embarked missing values handled
    if(len(modes)>1):
        modes='S'
    else :
        modes=modes[0]
    data_df['Embarked'].fillna(modes,inplace=True)
    fare_median=data_df['Fare'].median() ##Fare missing values handled
    data_df['Fare'].fillna(fare_median,inplace=True)
    data_df.drop_duplicates(inplace=True) 
    return data_df

def engineer_features(data_df):
    
    data_df['Title']=data_df['Name'].apply(commontitle) ##rare titles added
    data_df.columns
    data_df['FamilySize']=data_df['SibSp']+data_df['Parch']+1
    data_df['FamilySize'].isna().sum() ##checking for null values
    data_df['isAlone']=1 ##is alone binary factor has been set now 
    data_df.loc[data_df['FamilySize']>1,'isAlone']=0
    data_df['isAlone'].value_counts()
    data_df['AgeBin'] = pd.qcut(data_df['Age'], 5, labels=False, duplicates='drop') ##age bin
    data_df['FareBin'] = pd.qcut(data_df['Fare'], 5, labels=False, duplicates='drop')## fare bin
    data_df.drop_duplicates(inplace=True) 
    return data_df


def encode_and_scale(data_df):
    #sex,embarked,Pclass,Title 

    data_df['Sex'] #has male and female rn so one shot --> sex_male, sex_female
    data_df['Sex_male']=0
    data_df['Sex_female']=0
    data_df.loc[data_df['Sex']=='male','Sex_male']=1
    data_df.loc[data_df['Sex']=='female','Sex_female']=1
    data_df['Sex'].value_counts() ##577 male 314 female 
    data_df['Sex_female'].value_counts() #314--> 1 (its working)
    data_df['Embarked'].value_counts()
    data_df['Embarked_S']=0
    data_df['Embarked_C']=0
    data_df['Embarked_Q']=0
    data_df.loc[data_df['Embarked']=='S','Embarked_S']=1
    data_df.loc[data_df['Embarked']=='C','Embarked_C']=1
    data_df.loc[data_df['Embarked']=='Q','Embarked_Q']=1
    data_df['Embarked'].value_counts() ## 646 S 168 C 77 Q
    data_df['Embarked_C'].value_counts() ##-->168 so works (its working) 
    data_df['Pclass_3']=0
    data_df['Pclass_1']=0
    data_df['Pclass_2']=0
    data_df.loc[data_df['Pclass']==3,'Pclass_3']=1
    data_df.loc[data_df['Pclass']==1,'Pclass_1']=1
    data_df.loc[data_df['Pclass']==2,'Pclass_2']=1
    data_df['Pclass_3'].value_counts()
    data_df['Pclass'].value_counts() ## it matches so its also working
    data_df['Title_Mr'] = 0
    data_df['Title_Miss'] = 0
    data_df['Title_Master'] = 0
    data_df['Title_rare'] = 0
    data_df.loc[data_df['Title'] == 'Mr', 'Title_Mr'] = 1
    data_df.loc[data_df['Title'] == 'Miss', 'Title_Miss'] = 1
    data_df.loc[data_df['Title'] == 'Master', 'Title_Master'] = 1
    data_df.loc[data_df['Title'] == 'rare', 'Title_rare'] = 1
    data_df['Title'].value_counts()
    data_df['Title_Miss'].value_counts() ## its working
    data_df.drop(columns=['Name','Ticket','PassengerId'],inplace=True)
    fare_cap = data_df['Fare'].quantile(0.99)
    age_cap = data_df['Age'].quantile(0.99)
    data_df.loc[data_df['Age']>age_cap,'Age']=age_cap
    data_df.loc[data_df['Fare']>fare_cap,'Fare']=fare_cap ##age cap and fare cap have been set individually based on 
    # Normalize Age so now its between 0 to 1 
    data_df['Age'] = (data_df['Age'] - data_df['Age'].min()) / (data_df['Age'].max() - data_df['Age'].min())

    # Normalize Fare so now its beetween 0 to 1
    data_df['Fare'] = (data_df['Fare'] - data_df['Fare'].min()) / (data_df['Fare'].max() - data_df['Fare'].min())
    data_df.drop_duplicates(inplace=True) 
    return data_df
## you can also do it in another way by using value_counts and then put normalise=True




    ##dropping columns {Name,Ticket,PassengerId}
def save_outputs(data_df):
    data_df.drop_duplicates(inplace=True) 
    data_df.to_csv('../output/cleaned.csv', index=False)





def commontitle(title):
    common=['Mr','Miss','Master']
    for titlemaybe in common:
        if titlemaybe in title:
            return titlemaybe
    return "rare"


    