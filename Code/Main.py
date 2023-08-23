# ======================================================================================================================
# ======================= EEG for motor imagery  ===========================
# ====================== Presented by: Reza Saadatyar  =====================
# =================== E-mail: Reza.Saadatyar92@gmail.com  ==================
# ============================  2022-2023 ==================================
# The program will run automatically when you run code/file Main.py, and you do not need to run any of the other codes.
# ================================================== Import Libraries ==================================================
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from Filtering import filtering
from sklearn import model_selection
from Normalize import normalize_data
from CSP import common_spatial_pattern
from Preparing_data import preparing_data
from Spatial_Filter import spatial_filter
from Classification import classification
from Output_Training_Test_Network import output_network
# ============================================Step 1: Preparing the data ===============================================
# ----------------------------------------- Step 1.1: Load data & Time_trial -------------------------------------------
data_set = loadmat('BCICIV_calib_ds1a_100Hz.mat')
Key = data_set.keys()
Data = np.double(data_set['cnt']) * 0.1
N = Data.shape[1]

Nfo = data_set['nfo']
Fs = 100
Time = np.arange(Data.shape[0]) / Fs

Mrk = data_set['mrk']
Time_Trial = np.array(Mrk['pos'].tolist()).flatten()
# -------------------------------------------- Step 1.2: position_classes ----------------------------------------------
Name_class = Nfo['classes']
Name_channel = list(np.concatenate(list(np.array(Nfo['clab'].tolist()).flatten())))

X_position = list(np.array(Nfo['xpos'].tolist()).flatten())
Y_position = list(np.array(Nfo['ypos'].tolist()).flatten())
Position_XY = np.array([X_position, Y_position])  # Position electrode
# -------------------------------------------- Step 1.3: Labels --------------------------------------------------------
Labels = np.array(Mrk['y'].tolist()).flatten()
Data, Labels = preparing_data(Data, Labels)
# ======================================== Step 2: Filtering & Data scaling ============================================
# ---------------------------- Step 2.1: Band pass filtering to get beta and mu band information -----------------------
Data_Filter = filtering(Data, f_low=8, f_high=30, order=3, fs=100, btype='bandpass')      # btype:'low', 'high', 'bandpass', 'bandstop'
# ---------------------------------------------- Step 2.2: Data scaling ------------------------------------------------
# Data = normalize_data(Data, Type_Normalize='MinMaxScaler', Display_Figure='on')   # Type_Normalize:'MinMaxScaler', 'normalize'
# =============================== Step 3: Source localization using special filters ====================================
Type_Filter = "HL"     # 'CAR', 'LL', 'HL'
Display_Figure = "On"   # "On" , "Off"
Data_Filter = spatial_filter(Data_Filter, Position_XY, Fs, Type_Filter, Name_channel, Display_Figure)
# ============== Step 4: Separate trials: Number Samples each trial*number channel*number trial for SCP ================
Ltr = 4 * Fs
Data1 = np.zeros((Ltr, N, 100))
Data2 = np.zeros((Ltr, N, 100))
c1 = 0
c2 = 0
for i in range(0, len(Time_Trial)):
    Data_Trial = Data_Filter[Time_Trial[i]:Time_Trial[i] + Ltr, :]
    if Labels[i] == 0:   # Label 1
        Data1[:, :, c1] = Data_Trial
        c1 = c1 + 1
    elif Labels[i] == 1:  # Label -1
        Data2[:, :, c2] = Data_Trial
        c2 = c2 + 1

Label1 = np.zeros(np.shape(Data1)[2], dtype=int)
Label2 = np.ones(np.shape(Data2)[2], dtype=int)
data_train1, Data_Test1, Label_Train1, Label_Test1 = model_selection.train_test_split(Data1.T, Label1, test_size=0.3, random_state=0)
data_train2, Data_Test2, Label_Train2, Label_Test2 = model_selection.train_test_split(Data2.T, Label2, test_size=0.3, random_state=0)
# ======================= Step 3: Common Spatial Pattern (Feature Extraction & Selection) ==============================
W = common_spatial_pattern(data_train1, data_train2, m=1)
Feature_Data1 = np.zeros((2, np.shape(Data1)[2]))
Feature_Data2 = np.zeros((2, np.shape(Data2)[2]))
Data1 = np.concatenate((data_train1, Data_Test1,), axis=0)
Data2 = np.concatenate((data_train2, Data_Test2,), axis=0)

plt.figure(figsize=(16, 10))
for i in range(0, np.shape(Data1)[0]):
    X1 = Data1[i, :, :]
    X2 = Data2[i, :, :]
    Y1 = W.T.dot(X1)
    Y2 = W.T.dot(X2)
    Feature_Data1[:, i] = np.var(Y1, axis=1)
    Feature_Data2[:, i] = np.var(Y2, axis=1)

    plt.subplot(131)
    plt.plot(X1[53, :], X1[55, :], '.r', X2[53, :], X2[55, :], '.b')
    plt.title('EEG Before SCP; Part Training')
    plt.subplot(132)
    plt.plot(Y1[0, :], Y1[1, :], '.r', Y2[0, :], Y2[1, :], '.b')
    plt.title('EEG After SCP; Part Training')
plt.subplot(133)
plt.plot(Feature_Data1[0, :], Feature_Data1[1, :], '.r', Feature_Data2[0, :], Feature_Data2[1, :], '.b')
plt.title('Variance of EEG; Part Training')
plt.show()
# =========================================== Step 4: Classification  ==================================================
Data = np.concatenate((Feature_Data1, Feature_Data2), axis=1)
Labels = np.concatenate((Label1, Label2), axis=0)

model, type_class = classification(Data, Labels, type_class='NB', hidden_layer_mlp=(10,), max_iter=200, kernel_svm='rbf',
    c_svm=10, gamma_svm=0.7, max_depth=5, criterion_dt='entropy', n_estimators=500)
Accuracy_Train, Cr_Train, Accuracy_Test, Cr_Test = output_network(Data, Labels, model, type_class, k_fold=5)
"""
type_class: 'KNN', 'LR', 'MLP', 'SVM', 'DT', 'NB', 'RF', 'AdaBoost', 'XGBoost', 'LDA'
LR: LogisticRegression; MLP: Multilayer perceptron, SVM:Support Vector Machine; DT: Decision Tree; NB: Naive Bayes;
RF: Random Forest; AdaBoost; XGBoost; LDA: Linear Discriminant Analysis; KNN:K-Nearest Neighbors 
Parameters:
The number of hidden layers: hidden_layer_mlp; The number of epochs MLP: max_iter,
kernel_svm=‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’;  c_svm=Regularization parameter, 
gamma_svm=Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. 
max_depth=The maximum depth of the tree, random forest and XGBoost; criterion= 'gini', 'entropy', 'log_loss';
n_estimators:The number of trees in the forest.
"""


"""
Num_K = KNN_Optimal(Data_Train, Label_Train, Data_Test, Label_Test, N=21)      # Obtain optimal K
model = neighbors.KNeighborsClassifier(n_neighbors=Num_K, metric='minkowski')  # Train Network

Cr_Train, Cr_Test = Output_Network(Data_Train, Label_Train, Data_Test, Label_Test, model)
model = naive_bayes.GaussianNB()
model.fit(Data_Train.T, Label_Train)
Label_Pred = model.predict(Data_Test)
"""