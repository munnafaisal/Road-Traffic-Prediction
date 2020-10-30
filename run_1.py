import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from collections import Iterable
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVC

data = pd.read_csv('./traffic-studies-speed-reports-beta-1.csv')
path1 = './traffic-studies-speed-reports-beta-1.csv'


speed_list = ['SPEED_0_14', 'SPEED_15_19', 'SPEED_20_24', 'SPEED_25_29', 'SPEED_30_34', 'SPEED_35_39', 'SPEED_40_44', 'SPEED_45_49', 'SPEED_50_54', 'SPEED_55_59', 'SPEED_60_64', 'SPEED_65_69', 'SPEED_70_200']

labels = []
n_cl = 3
cl_data = {}
cl_tr_data = {}
tr_state = [0,1,2]

def get_sl_list(speed_list):

    l = []
    for i in speed_list:
        l.append(int(i.split('_')[2]))
    
    return l
    

def avg_speed(sp_limit, mx_index):
    
    sp = []
    
    for i in mx_index:
        sp.append(sp_limit[i])
    
    return sp 
    
def get_site_code_conversion(s_code, s_code_data):
    
    sc_int = []
    s_code = list(s_code)
    for i in s_code_data:
        
        k = [m for m, j in enumerate(s_code) if j == i]
        sc_int.append(k[0])
    
    return sc_int
    
    
def get_all_speed_data(speed_list, data):
    
    d = []
    for i in speed_list :
        d.append(list(data[i]))

    zipped_data = list(zip(
        d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11],d[12]))
    
    return zipped_data
    

def flatten(lis):

    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item

def get_data(train,path,site_codes,avg_spds):

    if train:

        data = pd.read_csv(path)
        features = list(data.columns)
        print(" feature ", features)
        avg_spds_cl = np.array(avg_spds).reshape(-1, 1)
        day_of_month_list = list(data.DAY_OF_MONTH)
        day_of_week = list(data.DAY_OF_WEEK)
        site_code  = site_codes
        time_hour = list(data.TIME)

        hour_conversion = []

        for i, d in enumerate(time_hour):
           
            m = d.split(':')
            k = [np.int(m[0]),np.int(m[1])]
            hour_conversion.append(k)

        zipped_data = list(zip(
            site_code,day_of_month_list,day_of_week,hour_conversion,avg_spds))

        for i, d in enumerate(zipped_data):
            zipped_data[i] = list(flatten(d))

        return zipped_data,avg_spds_cl,avg_spds

    else:

        data = pd.read_csv(path)
        features = list(data.columns)
        print(" feature ", features)
        day_of_month_list = list(data.DAY_OF_MONTH)
        day_of_week = list(data.DAY_OF_WEEK)
        site_code = site_codes
        time_hour = list(data.TIME)

        hour_conversion = []

        for i, d in enumerate(time_hour):
            m = d.split(' ')[1].split(':')
            k = [np.int(m[0]), np.int(m[1])]
            hour_conversion.append(k)

        zipped_data = list(zip(
            site_code, day_of_month_list, day_of_week, hour_conversion))

        for i, d in enumerate(zipped_data):
            zipped_data[i] = list(flatten(d))

        return zipped_data

for i in range(n_cl):

    cl_data[i] = []
    cl_tr_data[i] = []

def get_cluster_labels(travel_time):

    print(" clustering in Progress ")
    kmeans = KMeans(n_clusters=n_cl)
    kmeans = kmeans.fit(travel_time)

    labels = kmeans.predict(travel_time)
    return labels

def get_cluster_data(labels,travel_time_list):

    for index, (y,x) in enumerate(zip(labels,travel_time_list)):
        cl_data[y].append(x)

    return cl_data

def get_traffic_status(tr_state,cl_data):

    for i in range(3):
        tr_state[i] = np.mean(cl_data[i])


    tr_state = np.sort(tr_state)

    return tr_state

def get_trained_data(tr_state,tr_zipped_data):

    for i, tr in enumerate(tr_state):
        if i==0:
            r1 = 0
            r2 = tr_state[0]

        elif i == 1:
            r1 = tr_state[0]
            r2 = tr_state[1]
        else:
            r1 = tr_state[1]
            r2 = 100

        cl_tr_data[i] = [z[:-1] for z in tr_zipped_data if z[-1]>=r1 and z[-1]<=r2]
        #cl_tr_data[i] = [z[:-1] for z in tr_zipped_data if z[-1]>=r1 and z[-1]<=r2]

    return cl_tr_data

def get_label_data(cl_tr_data):

    l= []
    for i in range(3):
        l.append(list((np.ones(len(cl_tr_data[i])))*i))

    return l

def get_tr_data_list(cl_tr_data):

    ll = [v for k, v in cl_tr_data.items()]
    k = []

    for j in ll:
        for i in j:
            k.append(i)

    return k

def train_classifier(tr,lb):

    clf_1 = RandomForestClassifier(class_weight='balanced',n_jobs=4,verbose=True)
    clf_1.fit(tr, lb)

    # clf_2 = SVC(kernel='linear',
    #             class_weight='balanced',
    #             probability=True,verbose=True,max_iter=100)

    clf_2 = NuSVC(nu=0.3,kernel='rbf',gamma='auto',class_weight='balanced',verbose= True,probability=True)

    clf_2.fit(tr, lb)


    scores1 = cross_val_score(clf_1, tr, lb, cv=5, scoring='f1_macro')
    scores2 = cross_val_score(clf_2, tr, lb, cv=5, scoring='f1_macro')

    
    return clf_1, clf_2, scores1, scores2

def plot_scores(sc1,sc2,s1,s2):
    
    labels = []
    
    for i in range(len(sc1)):
        labels.append(str(i))

    plt.figure(200)
    plt.subplot(3, 1, 1)
    plt.bar(labels,sc1)
    plt.title('Cross Validation '+'Score :' + 'RFC' + ' ')

    plt.subplot(3, 1, 2)
    plt.bar(labels,sc2)
    plt.title('Cross Validation '+'Score :' + 'NuSVC' + ' ')

    plt.subplot(3, 1, 3)
    plt.bar(['RFC', 'NuSVC'], [s1,s2])
    plt.title('TEST SCORE')
    


# site_code,day_of_month_list,day_of_week,hour_conversion
def plot_prediction(c1,c2,test_data,site_code):

    names = ['High', 'Mediam', 'Low']
    t1 = str(test_data[0][3]) + ':'  + str(test_data[0][4])
    t2 = str(test_data[1][3]) + ':'  + str(test_data[1][4])
    
    z1 = site_code[test_data[0][0]]
    z2 = site_code[test_data[1][0]]

    
    plt.figure(300)
    plt.subplot(2, 2, 1)
    plt.bar(names, c1[0])
    plt.title('Classifier :'+ 'RFC' + ' Time : '+t1+ ' Day_Month :'+ str(test_data[0][1]) + ' Day_Week :'+ str(test_data[0][2]) +" Zone :" +z1)

    plt.subplot(2, 2, 2)
    plt.bar(names, c2[0])
    plt.title("Classifier :"+ 'NuSVC' + ' Time : '+t1+ ' Day_Month :'+ str(test_data[0][1]) + ' Day_Week :'+ str(test_data[0][2]) +" Zone :" +z1)

    plt.subplot(2, 2, 3)
    plt.bar(names, c1[1])
    plt.title('Classifier :' + 'RFC' + ' Time : '+t2+ ' Day_Month :'+ str(test_data[1][1]) + ' Day_Week :'+ str(test_data[1][2]) +" Zone :" +z2)

    plt.subplot(2, 2, 4)
    plt.bar(names, c2[1])
    plt.title("Classifier :" + 'NuSVC' + ' Time : '+t2+ ' Day_Month :'+ str(test_data[1][1]) + ' Day_Week :'+ str(test_data[1][2]) +" Zone :" +z2)


tr_data = get_all_speed_data(speed_list, data)
np_tr = np.array(tr_data)

max_idx = np.argmax(np_tr, axis=1)
speed_limit = get_sl_list(speed_list)

avg_speed_list = avg_speed(speed_limit,max_idx)
s_c = list(data.SITE_CODE.unique())
site_code = list(data.SITE_CODE)
s_c_con = get_site_code_conversion(s_code=s_c, s_code_data= site_code)

tr_zipped_data,travel_time,travel_time_list = get_data(train=True,path=path1,site_codes=s_c_con,avg_spds=avg_speed_list) #Read Data from TRAIN_SET.csv

labels = get_cluster_labels(travel_time)                                        #Get Cluster Travel Time label

cl_data = get_cluster_data(labels,travel_time_list)                             #Get Clustered Travel Time data

tr_state = get_traffic_status(tr_state,cl_data)                                 #Get Travel Time Threshold which will define Traffic status

print('\n Average Speed Threshold')
print(' HIGH :',tr_state[0],'MEDIUM :',tr_state[1], 'LOW :',tr_state[2])


        ###  Get Train Data ###

cl_tr_data = get_trained_data(tr_state,tr_zipped_data)
tr = get_tr_data_list(cl_tr_data)

        ### Get Label Data ###

lb = list(flatten(get_label_data(cl_tr_data)))

        ##### Train Classifier ######

X_train, X_test, y_train, y_test = train_test_split(tr, lb, test_size=0.2, random_state=True) #Train Test Split

c1, c2,sc1,sc2 = train_classifier(X_train, y_train)

test_data = X_test[0:2]

cc1 = c1.predict_proba(X_test[0:2])
cc2 = c2.predict_proba(X_test[0:2])

        #### Plot Prediction and Scores #####

plot_prediction(cc1,cc2,test_data,site_code)
plot_scores(sc1,sc2,c1.score(X_test,y_test),c2.score(X_test,y_test))
plt.show()
