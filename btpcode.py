# %% [code]


# %% [code] {"execution":{"iopub.status.busy":"2023-04-14T10:14:45.400846Z","iopub.execute_input":"2023-04-14T10:14:45.401249Z","iopub.status.idle":"2023-04-14T10:14:46.802095Z","shell.execute_reply.started":"2023-04-14T10:14:45.401213Z","shell.execute_reply":"2023-04-14T10:14:46.800604Z"}}
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.model_selection import cross_val_score
from sklearn.metrics import *
import numpy as np
from math import *
import pickle


import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import *
from scipy.special import erf as er
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ShuffleSplit
from sklearn.feature_selection import mutual_info_classif as mui
from sklearn.feature_selection import chi2 as chi2
from sklearn.feature_selection import f_classif
# from sklearn.feature_selection import mrmr_classif as mrmr
from sklearn.feature_selection import SelectKBest as skb
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot as plt
import timeit
import os



path = "C://Users//kumar//Desktop//Untitled Folder//datum"


#Your statements here



# %% [code] {"execution":{"iopub.status.busy":"2023-04-11T16:36:35.058561Z","iopub.execute_input":"2023-04-11T16:36:35.059187Z","iopub.status.idle":"2023-04-11T16:36:35.072376Z","shell.execute_reply.started":"2023-04-11T16:36:35.059146Z","shell.execute_reply":"2023-04-11T16:36:35.070889Z"}}

def initialize_feature_pop(npop, dim):
    popu = np.zeros((npop, dim))
    for i in range(npop):
        while (np.sum(popu[i]) == 0):
            popu[i, :] = np.random.randint(0, 2, size=(1, dim))
    return popu

def CostFunction(obj, dataframe, target,p_score):
    return fitness_function(obj, dataframe, target,p_score)

def Distance(obj1, obj2):
    return np.sqrt(np.sum((obj1 - obj2)**2))

def fitness_function(pop, datafr, targ,p_score):
    cost = np.zeros((pop.shape[0]))
    for i in range(pop.shape[0]):
        cols = [j for j in range(pop.shape[1]) if pop[i][j] == 1]
        new = np.array(datafr[:, cols])
        # xtrain, xtest, ytrain, ytest = train_test_split(new, targ, test_size=0.3)

        kf = KFold(n_splits=10, shuffle=False)
        # # score = []
        # # for train_index, test_index in kf.split(new):
        # #     x_train, x_test = new[train_index], new[test_index]
        # #     y_train, y_test = targ[train_index], targ[test_index]
        # #     clf1 = GaussianNB()
        # #     clf1.fit(x_train, y_train)
        # #     score.append(clf1.score(x_test, y_test))
        # clf1 = knn()
        # clf1.fit(xtrain, ytrain)
       
        # accuracys = clf1.score(xtest,ytest)
        score = cross_val_score(GaussianNB(), new, targ, cv = kf, scoring = 'accuracy', n_jobs=1)
        accuracys = np.mean(score)
        cost[i] = (1-accuracys)*0.9+0.1*(np.mean(pop[i]))+0.001*(1-np.mean(p_score[pop[i]==1]));
    return cost


# %% [code] {"execution":{"iopub.status.busy":"2023-04-11T16:36:35.074052Z","iopub.execute_input":"2023-04-11T16:36:35.074431Z","iopub.status.idle":"2023-04-11T16:36:35.088276Z","shell.execute_reply.started":"2023-04-11T16:36:35.074392Z","shell.execute_reply":"2023-04-11T16:36:35.086713Z"}}
def V_1(changes, original):
    mutation_rate = 0.05
    chang = np.array(changes)
    orig = np.array(original)
    mask = np.array(abs(er((sqrt(pi)/2)*chang)))
    # print(abs(er((sqrt(pi)/2)*1)))
    # print(np.max(mask))
    r = np.random.uniform(0, 1)
    # print(r)
    for i in range(orig.shape[0]):
        if (mask[i] >= r):
            orig[i] = 1
        else :
            orig[i]=0
    for i in range(orig.shape[0]):
        r = np.random.uniform(0,1)
        if r < mutation_rate:
            orig[i] = 1 - orig[i]
    # while (np.sum(orig, dtype=np.int32) == 0):
    #     orig = np.random.randint(0, 2, size=orig.shape)
    if(np.sum(orig,dtype=np.int32)==0):
        return original
  # print(original)
    return orig
def crossover(hawk1, hawk2):
    crossover_rate = 0.1
    r = np.random.uniform(0,1)
    new1, new2 = hawk1, hawk2
    if r > crossover_rate: return new1, new2
    else:
        Hawk1, Hawk2 = np.array(hawk1), np.array(hawk2)
        new1, new2 = np.zeros_like(Hawk1), np.zeros_like(Hawk2) 
        # print('1 = ' ,Hawk1)
        # print('2 = ', Hawk2)
        for i in range(Hawk1.shape[1]):
            rr = np.random.uniform(0,1)
            if rr >= 0.5:
                new1[0][i], new2[0][i] = Hawk1[0][i], Hawk2[0][i]
            else:
                new1[0][i], new2[0][i] = Hawk2[0][i], Hawk1[0][i]
        while(np.sum(new1[0], dtype=np.int32)==0):
            new1 = np.random.randint(0, 2, size = new1.shape)
        while(np.sum(new2[0], dtype=np.int32)==0):
            new2 = np.random.randint(0, 2, size = new1.shape)
        return new1, new2
def remove_Categorical(df):
    new_df = df.copy(deep = False)
    allcols = []
    for col in new_df.columns:
        if new_df[col].dtype == 'object':
            allcols.append(col)
        elif new_df[col].min()!=new_df[col].max():
            new_df[col]=(new_df[col]-new_df[col].min())/(new_df[col].max()-new_df[col].min())
        else:
            new_df[col]=new_df[col]+1
            new_df[col]=new_df[col]//new_df[col].min()
    for col in allcols:
        le = preprocessing.LabelEncoder()
        new_df[col] = le.fit_transform(new_df[col])
    return new_df



files=[  'Mobile.csv']
totalres=pd.DataFrame({},columns=["Name","Accuracy","fet","exectime"])

for fil in files:
    filesname=fil;
    fet_num=[];
    accuracys=[];
    accuracysfortable=[];
    exec_tim=[];

# extension="data"
    try:
            df = pd.read_csv(filesname,header=None)
    except:
            continue
        
    for j in range(20):
        print(j)
        print()
        print()
        print(filesname)
        # data_name = filesname
        # data1_name = data_name.lower() + "data"
        
        try:
            df = pd.read_csv(filesname,header=None)
        except:
            continue
        

        print(df.shape)
        nfeatures = df.shape[1] - 1
        df=df.sample(frac=1)
        
        DF = df.drop(nfeatures, axis=1)
        DF = remove_Categorical(DF)
        ta = df[nfeatures]
        
        
    
        X_kbest = np.array(DF)
    
        le = preprocessing.LabelEncoder()
        ta = le.fit_transform(ta)
        ta=np.array(ta);
        lat=np.array(ta.reshape((-1,1)));
        df = DF
        # newdfsave=np.array(df);
        # newdfsave=np.concatenate((newdfsave,lat),axis=1);
        # df.to_csv(filesname+"new.csv",index=False,header=False);
    # print(df.head())
        f_score,p_score=f_classif(X_kbest,ta)
        p_score=np.array(p_score)
        print(p_score,f_score)
        p_score=(p_score-p_score.min())/(p_score.max()-p_score.min())
        p_score[np.isnan(p_score)] = 0
# %% [code] {"execution":{"iopub.status.busy":"2023-04-11T16:36:36.473253Z","iopub.execute_input":"2023-04-11T16:36:36.473686Z","iopub.status.idle":"2023-04-11T16:36:50.166336Z","shell.execute_reply.started":"2023-04-11T16:36:36.473646Z","shell.execute_reply":"2023-04-11T16:36:50.165277Z"}}
        VarNumber = X_kbest.shape[1]  # Dimensionality of our problem
        MaxFes = 110  # Maximum number of generations
        nPop = 10  # Number of population in each iteration
# Randomly choosing the number of FireHawks in an iteration
        HN = 3
        w = 1
# Counters
        Iter = 0
        FEs = 0
        start = timeit.default_timer()

        Pop = initialize_feature_pop(nPop, VarNumber)
        FEs += nPop
        Cost = CostFunction(Pop, X_kbest, ta,p_score)

# %% [code] {"execution":{"iopub.status.busy":"2023-04-11T16:36:50.167880Z","iopub.execute_input":"2023-04-11T16:36:50.168184Z","iopub.status.idle":"2023-04-11T16:36:50.181459Z","shell.execute_reply.started":"2023-04-11T16:36:50.168159Z","shell.execute_reply":"2023-04-11T16:36:50.180078Z"}}
        SortOrder = np.argsort(Cost)
        Cost = np.sort(Cost)
# print(SortOrder)
        Pop = Pop[SortOrder, :]
        BestPop = np.array(Pop[0, :])
        SP = np.mean(Pop, axis=0)

        FHPops = np.array(Pop[0:HN, :])  # Fire Hawks
        Pop2 = np.array(Pop[HN:, :])  # Prey
        PopNew = []
        for i in range(HN):
            nPop2 = Pop2.shape[0]
    # print("nPop2")
    # print(nPop2)
            if nPop2 < HN-i:
                break
            Dist = np.zeros((nPop2))
            for q in range(nPop2):
                Dist[q] = Distance(FHPops[i, :], Pop2[q, :])
            b = np.argsort(Dist)
            Dist = np.sort(Dist)
    # Randomly assigning alpha number of nearest preys to each fire hawk.
            alpha = np.random.randint(1, nPop2+1)
    # print("alpha")
    # print(alpha)
            PopNew.append(np.array(Pop2[b[0:alpha], :]))

            Pop2 = np.delete(Pop2, b[0:alpha], 0)
    # If every prey has been assigned to a fire hawk, then stop.
            if not np.any(Pop2):
                break

# If some prey are left to be assigned, assign them to the last fire hawk.
        if np.any(Pop2):
            PopNew[-1] = np.concatenate((PopNew[-1], Pop2), axis=0)
        GB = Cost[0]
        BestPos = np.array(BestPop)


# %% [code] {"execution":{"iopub.status.busy":"2023-04-11T16:36:50.183591Z","iopub.execute_input":"2023-04-11T16:36:50.184094Z","iopub.status.idle":"2023-04-11T17:26:33.599535Z","shell.execute_reply.started":"2023-04-11T16:36:50.184052Z","shell.execute_reply":"2023-04-11T17:26:33.598162Z"}}
        all_accuracies = []
        all_fitness = []
        while FEs < MaxFes:
            FEs = FEs+1
            Iter += 1
            PopTot = np.zeros_like(Pop)
            Cost = np.zeros_like(Cost)
        # print(BestPos)
            for i in range(len(PopNew)):
    
                PR = PopNew[i]
                FHl = np.array(FHPops[i, :])
                SPl = np.mean(PR, axis=0)
                r1, r2,q ,r4= np.random.uniform(0, 1), np.random.uniform(0, 1),np.random.uniform(0,1),np.random.uniform(0,1)
                FHnear = np.array(FHPops[np.random.randint(0, HN), :])
                # FHl_change=0
                if(q>=0.5):
                    FHl_change=SPl-r1*abs(SPl-2*r2*FHl)
                else:
                    FHl_change = (r1*BestPos-r2*FHnear)
                
                # print(np.max(FHl_change));
                FHl_new = np.array(V_1(FHl_change, FHl))
                # FHl_new,_=crossover(FHl_new,FHnear)

                FHl_new = np.reshape(FHl_new, (1, PopTot.shape[1]))
                FHl_new, FHnear = crossover(FHl_new, np.reshape(FHnear, (1, PopTot.shape[1])))
    
        # print(FHl_new.shape)
                if i == 0:
                    PopTot = FHl_new
                else:
                    PopTot = np.row_stack((PopTot, FHl_new))
            # print(BestPos, i, Iter)
            # print(PopTot.shape)
                for q in range(PR.shape[0]):
                    r1, r2 = np.random.uniform(0, 1), np.random.uniform(0, 1)
    
            # new position of fire hawks by Eq. 7
                    PRq_chang1 = (r1*FHl-r2*SPl)
            #         PRq_new1 = np.array(V_1(PRq_chang1, PR[q, :]))
    
            #         PRq_new1 = np.reshape(PRq_new1, (1, FHl_new.shape[1]))
            # # PRq_new1=np.clip(PRq_new1,VarMin,VarMax);
            #         PopTot = np.row_stack((PopTot, PRq_new1))
                    r1, r2 = np.random.uniform(0, 1), np.random.uniform(0, 1)
                    FHAlter = np.array(FHPops[np.random.randint(0, HN), :])
    
            # new position of fire hawks by Eq. 8
                    PRq_chang1 =PRq_chang1+ (r1*FHAlter-r2*SP)
                    PRq_new2 = np.array(V_1(PRq_chang1, PR[q, :]))
            # PRq_new2=PR[q,:]+(r1*FHAlter-r2*SP);
            # PRq_new2=np.clip(PRq_new2,VarMin,VarMax);
                    PRq_new2 = np.reshape(PRq_new2, (1, FHl_new.shape[1]))
                    PopTot = np.row_stack((PopTot, PRq_new2))
        # print(BestPos)
        # print(1)
        # print(PopTot)
        # for i in range(PopTot.shape[0]):
        #     if (np.sum(PopTot[i], dtype=np.int32) == 0):
        #         print("Yes")
        #         print(PopTot[i])
        #         while (np.sum(PopTot[i], dtype=np.int32) == 0):
        #             PopTot[i] = np.random.randint(0, 2, size=(1, VarNumber))
        # print(2)
            # print(PopTot)
            Cost = CostFunction(PopTot, X_kbest, ta,p_score)
    
            SortOrder = np.argsort(Cost)
            Cost = np.sort(Cost)
            Pop = np.array(PopTot[SortOrder, :])
            Pop = np.array(Pop[0:nPop])
            HN = 3
            BestPop = np.array(Pop[0])
    
            SP = np.mean(Pop, axis=0)
            FHPops = np.array(Pop[0:HN, :])  # Fire Hawks
            Pop2 = np.array(Pop[HN:, :])  # Preys
    
        # Again distance calculation of each prey from fire hawk and grouping the preys with them
            for i in range(HN):
                nPop2 = Pop2.shape[0]
                if nPop2 < HN:
                    break
                Dist = np.zeros((nPop2))
                for q in range(nPop2):
                    Dist[q] = Distance(FHPops[i, :], Pop2[q, :])
                b = np.argsort(Dist)
                Dist = np.sort(Dist)
                alfa = np.random.randint(1, nPop2+1)
                if i == 0:
                    PopNew = []
                PopNew.append(Pop2[b[0:alfa], :])
                Pop2 = np.delete(Pop2, b[0:alfa], 0)
                if not np.any(Pop2):
                    break
            if np.any(Pop2):
                PopNew[-1] = np.concatenate((PopNew[-1], Pop2), axis=0)
    
            if Cost[0] < GB:
                GB = Cost[0]
                BestPos = np.array(BestPop)
            all_accuracies.append(1-GB)
            all_fitness.append(GB)
            # print(Pop[0] == BestPos)
            print(BestPos)

         
    
            print("Iteration no.: {} --- Best Cost: {}".format(Iter, GB))
    
    
    # %% [code] {"execution":{"iopub.status.busy":"2023-04-11T17:26:33.600905Z","iopub.execute_input":"2023-04-11T17:26:33.601254Z","iopub.status.idle":"2023-04-11T17:26:33.725285Z","shell.execute_reply.started":"2023-04-11T17:26:33.601222Z","shell.execute_reply":"2023-04-11T17:26:33.723676Z"}}
        print(1-GB)
        print(BestPos)
        print(np.sum(BestPos))
    
        stop = timeit.default_timer()
    
        print(stop-start,"sdfa")
        colss = [j for j in range(BestPos.shape[0]) if BestPos[j] == 1]
        print(X_kbest.shape)
        newx = np.array(X_kbest[:, colss])
        print(newx.shape)
        exec_tim.append(stop-start);
        fet_num.append(np.sum(BestPos));
        targ=np.array(ta)
        print('Time: ', stop - start)  
        kfs = KFold(n_splits=10, shuffle=True)
        # score = []
        # for train_index, test_index in kf.split(newx):
        #     x_train, x_test = newx[train_index], newx[test_index]
        #     y_train, y_test = targ[train_index], targ[test_index]
        #     clf1 = knn() 
        #     clf1.fit(x_train, y_train)
        #     score.append(clf1.score(x_test, y_test))
        scores = cross_val_score(GaussianNB(), newx, targ, cv = kfs, scoring = 'accuracy', n_jobs=1)
        accs = np.mean(scores)
    
        accuracys.append([accs,all_accuracies,all_fitness])
        accuracysfortable.append(accs);
        print(accs)

        # for i in range(X_kbest.shape[1]):
        #     kf = KFold(n_splits=10, shuffle=True)
        #     nex=np.array(X_kbest[:,i]).reshape((-1,1))
        #     scoret = cross_val_score(knn(), nex, targ, cv = kf, scoring = 'accuracy', n_jobs=1)
        #     accis = np.mean(scoret)
        #     print(i,accis)

    
    
    
# totalres.to_csv("./results/resfin.csv",index=False);
# print(df.columns)
# selected = [df.columns[i] for i in colss]
# print(selected)
# newdf = df[selected]
# # label.to_csv("label1.csv", index=False)
# newdf.to_csv(data_name+"selected.csv", index=False)

# # %% [code] {"execution":{"iopub.status.busy":"2023-04-11T17:26:33.726931Z","iopub.execute_input":"2023-04-11T17:26:33.727678Z","iopub.status.idle":"2023-04-11T17:26:33.932782Z","shell.execute_reply.started":"2023-04-11T17:26:33.727641Z","shell.execute_reply":"2023-04-11T17:26:33.931369Z"}}
# plt.plot([i+1 for i in range(len(all_accuracies))],all_accuracies,  '-o')
# plt.xlabel("No. of iterations")
# plt.ylabel("Accuracy")
# plt.title("Accuracy vs iteration")
# plt.savefig(data_name + '-accuracies1.png')
# plt.show()


# # %% [code] {"execution":{"iopub.status.busy":"2023-04-11T17:26:33.935302Z","iopub.execute_input":"2023-04-11T17:26:33.935989Z","iopub.status.idle":"2023-04-11T17:26:34.118445Z","shell.execute_reply.started":"2023-04-11T17:26:33.935928Z","shell.execute_reply":"2023-04-11T17:26:34.117446Z"}}
# plt.plot([i+1 for i in range(len(all_fitness))],all_fitness,  '-o')
# plt.xlabel("No. of iterations")
# plt.ylabel("Fitness")
# plt.title("Fitness vs iteration")
# plt.savefig(data_name + '-fitness1.png')
# plt.show()


# %% [code]



# %% [code]


# %% [code]
