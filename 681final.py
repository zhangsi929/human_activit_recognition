
# coding: utf-8

# In[2]:


# import training data
file = open("/Users/zhangsi929/Documents/681final/train/X_train.txt", "r")
train_x = file.readlines()
for i in range(0,len(train_x)):
    train_x[i] = train_x[i].split(' ')
    train_x[i] = list(filter(None,train_x[i]))
    train_x[i][len(train_x[i])-1]=train_x[i][len(train_x[i])-1].replace('\n','')
    for j in range(0,len(train_x[i])):
        train_x[i][j] = float(train_x[i][j])

file = open("/Users/zhangsi929/Documents/681final/train/Y_train.txt", "r")
train_y = file.readlines()
for i in range(0,len(train_y)):
    train_y[i] = int(train_y[i].replace('\n',''))
    
# import testing data
file = open("/Users/zhangsi929/Documents/681final/test/X_test.txt", "r")
test_x = file.readlines()
for i in range(0,len(test_x)):
    test_x[i] = test_x[i].split(' ')
    test_x[i] = list(filter(None,test_x[i]))
    test_x[i][len(test_x[i])-1]=test_x[i][len(test_x[i])-1].replace('\n','')
    for j in range(0,len(test_x[i])):
        test_x[i][j] = float(test_x[i][j])

file = open("/Users/zhangsi929/Documents/681final/test/Y_test.txt", "r")
test_y = file.readlines()
for i in range(0,len(test_y)):
    test_y[i] = int(test_y[i].replace('\n',''))


# In[3]:


# original svm
from sklearn import svm
from sklearn.metrics import confusion_matrix
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(train_x, train_y)
pred_y = clf.predict(test_x)
confusion_matrix(test_y, pred_y)


# In[80]:


# original decision tree
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_x, train_y)
pred_y = clf.predict(test_x)
confusion_matrix(test_y, pred_y)


# In[82]:


# random forest roc vs class
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
clf_rf = OneVsRestClassifier(RandomForestClassifier(n_estimators=70, max_depth=None, min_samples_split=2, random_state=0))
train_y_bi = label_binarize(train_y, classes=[1, 2, 3, 4, 5, 6])
test_y_bi = label_binarize(test_y, classes=[1, 2, 3, 4, 5, 6])
pred_y_rf = clf_rf.fit(pca_train_x, train_y_bi).predict_proba(pca_test_x)


# In[135]:


import matplotlib.pyplot as plt
fpr_rf = dict()
tpr_rf = dict()
roc_auc_rf = dict()
graph_label = ['walking', 'up_stairs', 'down_stairs', 'sitting', 'standing', 'laying']
for i in range(6):
    fpr_rf[i], tpr_rf[i], _ = roc_curve(test_y_bi[:, i], pred_y_rf[:, i])
    roc_auc_rf[i] = auc(fpr_rf[i], tpr_rf[i])
font = {'size'   : 22}
plt.rc('font', **font) 
plt.figure(figsize=(15,15))
line1,=plt.plot(fpr_rf[0],tpr_rf[0],'r--',label=graph_label[0]+ ' auc= ' + str(roc_auc_rf[0]))
line2,=plt.plot(fpr_rf[1],tpr_rf[1],'b--',label=graph_label[1]+ ' auc= ' + str(roc_auc_rf[1]))
line3,=plt.plot(fpr_rf[2],tpr_rf[2],'g--',label=graph_label[2]+ ' auc= ' + str(roc_auc_rf[2]))
line4,=plt.plot(fpr_rf[3],tpr_rf[3],'r+',label=graph_label[3]+ ' auc= ' + str(roc_auc_rf[3]))
line5,=plt.plot(fpr_rf[4],tpr_rf[4],'b+',label=graph_label[4]+ ' auc= ' + str(roc_auc_rf[4]))
line6,=plt.plot(fpr_rf[5],tpr_rf[5],'g+',label=graph_label[5]+ ' auc= ' + str(roc_auc_rf[5]))
plt.xlabel('fpr',fontsize=22)
plt.ylabel('tpr',fontsize=22)
plt.title('roc of different status',fontsize=22)
plt.legend(handles=[line1, line2, line3, line4, line5, line6])
#plt.savefig('/Users/zhaobi/Desktop/pca_svd.png')
plt.rc('font', **font)
plt.show()    


# In[ ]:


# linear svm roc vs class
clf_svm = OneVsRestClassifier(svm.SVC(kernel='linear', gamma=2,probability=True))
clf_svm = clf_svm.fit(pca_train_x, train_y_bi)
pred_y_svm = clf_svm.decision_function(pca_test_x)

fpr_svm = dict()
tpr_svm = dict()
roc_auc_svm = dict()

for i in range(6):
    fpr_svm[i], tpr_svm[i], _ = roc_curve(test_y_bi[:, i], pred_y_svm[:, i])
    roc_auc_svm[i] = auc(fpr_svm[i], tpr_svm[i])


# In[213]:


clf_svm_poly = OneVsRestClassifier(svm.SVC(C=1.0, kernel='poly', degree=3, gamma='auto'))
clf_svm_poly = clf_svm_poly.fit(pca_train_x, train_y_bi)
pred_y_svm_poly = clf_svm_poly.decision_function(pca_test_x)

fpr_svm_poly = dict()
tpr_svm_poly = dict()
roc_auc_svm_poly = dict()

for i in range(6):
    fpr_svm_poly[i], tpr_svm_poly[i], _ = roc_curve(test_y_bi[:, i], pred_y_svm_poly[:, i])
    roc_auc_svm_poly[i] = auc(fpr_svm_poly[i], tpr_svm_poly[i])
font = {'size'   : 22}
plt.rc('font', **font) 
plt.figure(figsize=(15,15))
line1,=plt.plot(fpr_svm_poly[0],tpr_svm_poly[0],'r--',label=graph_label[0] + ' auc= ' + str(roc_auc_svm_poly[0]))
line2,=plt.plot(fpr_svm_poly[1],tpr_svm_poly[1],'b--',label=graph_label[1]+ ' auc= ' + str(roc_auc_svm_poly[1]))
line3,=plt.plot(fpr_svm_poly[2],tpr_svm_poly[2],'g--',label=graph_label[2]+ ' auc= ' + str(roc_auc_svm_poly[2]))
line4,=plt.plot(fpr_svm_poly[3],tpr_svm_poly[3],'r+',label=graph_label[3]+ ' auc= ' + str(roc_auc_svm_poly[3]))
line5,=plt.plot(fpr_svm_poly[4],tpr_svm_poly[4],'b+',label=graph_label[4]+ ' auc= ' + str(roc_auc_svm_poly[4]))
line6,=plt.plot(fpr_svm_poly[5],tpr_svm_poly[5],'g+',label=graph_label[5]+ ' auc= ' + str(roc_auc_svm_poly[5]))
plt.xlabel('fpr',fontsize=22)
plt.ylabel('tpr',fontsize=22)
plt.title('roc of different status SVM-poly',fontsize=22)
plt.legend(handles=[line1, line2, line3, line4, line5, line6])
#plt.savefig('/Users/zhaobi/Desktop/pca_svd.png')
plt.rc('font', **font)
plt.show()


# In[205]:


clf_svm_rbf = OneVsRestClassifier(svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto'))
clf_svm_rbf = clf_svm_rbf.fit(pca_train_x, train_y_bi)
pred_y_svm_rbf = clf_svm_rbf.decision_function(pca_test_x)
fpr_svm_rbf = dict()
tpr_svm_rbf = dict()
roc_auc_svm_rbf = dict()
for i in range(6):
    fpr_svm_rbf[i], tpr_svm_rbf[i], _ = roc_curve(test_y_bi[:, i], pred_y_svm_rbf[:, i])
    roc_auc_svm_rbf[i] = auc(fpr_svm_rbf[i], tpr_svm_rbf[i])
font = {'size'   : 22}
plt.rc('font', **font) 
plt.figure(figsize=(15,15))
line1,=plt.plot(fpr_svm_rbf[0],tpr_svm_rbf[0],'r--',label=graph_label[0] + ' auc= ' + str(roc_auc_svm_rbf[0]))
line2,=plt.plot(fpr_svm_rbf[1],tpr_svm_rbf[1],'b--',label=graph_label[1]+ ' auc= ' + str(roc_auc_svm_rbf[1]))
line3,=plt.plot(fpr_svm_rbf[2],tpr_svm_rbf[2],'g--',label=graph_label[2]+ ' auc= ' + str(roc_auc_svm_rbf[2]))
line4,=plt.plot(fpr_svm_rbf[3],tpr_svm_rbf[3],'r+',label=graph_label[3]+ ' auc= ' + str(roc_auc_svm_rbf[3]))
line5,=plt.plot(fpr_svm_rbf[4],tpr_svm_rbf[4],'b+',label=graph_label[4]+ ' auc= ' + str(roc_auc_svm_rbf[4]))
line6,=plt.plot(fpr_svm_rbf[5],tpr_svm_rbf[5],'g+',label=graph_label[5]+ ' auc= ' + str(roc_auc_svm_rbf[5]))
plt.xlabel('fpr',fontsize=22)
plt.ylabel('tpr',fontsize=22)
plt.title('roc of different status SVM-rbf',fontsize=22)
plt.legend(handles=[line1, line2, line3, line4, line5, line6])
#plt.savefig('/Users/zhaobi/Desktop/pca_svd.png')
plt.rc('font', **font)
plt.show()


# In[228]:


font = {'size'   : 22}
plt.rc('font', **font) 
plt.figure(figsize=(15,15))
labels = ['linear', 'rbf', 'poly']
line1,=plt.plot(fpr_svm[4],tpr_svm[4],'r--',label=labels[0] + ', auc= ' + str(roc_auc_svm[4]))
line2,=plt.plot(fpr_svm_rbf[4],tpr_svm_rbf[4],'b--',label=labels[1]+ ', auc= ' + str(roc_auc_svm_rbf[4]))
line3,=plt.plot(fpr_svm_poly[4],tpr_svm_poly[4],'g--',label=labels[2]+ ', auc= ' + str(roc_auc_svm_poly[4]))
plt.xlabel('fpr',fontsize=22)
plt.ylabel('tpr',fontsize=22)
plt.title('roc of STANDING with different kernel',fontsize=22)
plt.legend(handles=[line1, line2, line3])
#plt.savefig('/Users/zhaobi/Desktop/pca_svd.png')
plt.rc('font', **font)
plt.show()


# In[219]:


roc_auc_svm_poly


# In[200]:


#plot roc of different status with svm_linear
font = {'size'   : 22}
plt.rc('font', **font) 
plt.figure(figsize=(15,15))
line1,=plt.plot(fpr_svm[0],tpr_svm[0],'r--',label=graph_label[0] + ' auc= ' + str(roc_auc_svm[0]))
line2,=plt.plot(fpr_svm[1],tpr_svm[1],'b--',label=graph_label[1]+ ' auc= ' + str(roc_auc_svm[1]))
line3,=plt.plot(fpr_svm[2],tpr_svm[2],'g--',label=graph_label[2]+ ' auc= ' + str(roc_auc_svm[2]))
line4,=plt.plot(fpr_svm[3],tpr_svm[3],'r+',label=graph_label[3]+ ' auc= ' + str(roc_auc_svm[3]))
line5,=plt.plot(fpr_svm[4],tpr_svm[4],'b+',label=graph_label[4]+ ' auc= ' + str(roc_auc_svm[4]))
line6,=plt.plot(fpr_svm[5],tpr_svm[5],'g+',label=graph_label[5]+ ' auc= ' + str(roc_auc_svm[5]))
plt.xlabel('fpr',fontsize=22)
plt.ylabel('tpr',fontsize=22)
plt.title('roc of different status SVM-linear',fontsize=22)
plt.legend(handles=[line1, line2, line3, line4, line5, line6])
#plt.savefig('/Users/zhaobi/Desktop/pca_svd.png')
plt.rc('font', **font)
plt.show()    


# In[5]:


# pca
from sklearn.decomposition import PCA as PCA
import matplotlib.pyplot as plt 
pca = PCA(n_components = 2)
pca_train_x = pca.fit_transform(train_x)
plt.figure(figsize=(15,15))
font = {'size'   : 22}
plt.rc('font', **font) 
unique = list(set(train_y))
colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
for i, u in enumerate(unique):
    xi = [pca_train_x[:,0][j] for j  in range(len(pca_train_x[:,0])) if train_y[j] == u]
    yi = [pca_train_x[:,1][j] for j  in range(len(pca_train_x[:,1])) if train_y[j] == u]
    labelx = ""
    if u == 1:
        labelx = "WALKING "
    elif u == 2:
        labelx = "WALKING_UPSTAIRS"
    elif u == 3:
        labelx = "WALKING_DOWNSTAIRS"
    elif u == 4:
        labelx = "SITTING "
    elif u == 5:
        labelx = "STANDING"
    elif u == 6:
        labelx = "LAYING "
    plt.scatter(xi, yi, c=colors[i], label = labelx, edgecolor='none', alpha=0.8,
            cmap=plt.cm.get_cmap('nipy_spectral', 1))
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title("PCA Scatter Plot for 6 Activities")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.);
# #plt.savefig('/Users/zhangsi929/Desktop/pca6.png')


# In[ ]:


# Feature Selection: Isomap + SVM
from sklearn import manifold, datasets
from sklearn.metrics import accuracy_score
n_neighbors = 10
pcs = [2, 5, 10, 50,100,150,200,250,300,400]
iso_pc = []
for pcs_num in pcs:
    isomap = manifold.Isomap(n_neighbors, n_components = pcs_num)
    iso_train_x = isomap.fit_transform(train_x)
    iso_test_x = isomap.transform(test_x)
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(iso_train_x, train_y)
    iso_pred_y = clf.predict(iso_test_x)
    iso_pc.append(accuracy_score(test_y,iso_pred_y,))
    

import matplotlib.pyplot as plt
font = {'size'   : 22}
plt.rc('font', **font) 
plt.figure(figsize=(15,15))
plt.plot(pcs,iso_pc)
plt.xlabel('n_components',fontsize=22)
plt.ylabel('accuracy',fontsize=22)
plt.title('isomap',fontsize=22)
#plt.savefig('/Users/zhangsi929/Desktop/iso+svm.png')
plt.rc('font', **font)
plt.grid(True)


# In[ ]:


# Feature Seletction: PCA + SVM 
from sklearn.decomposition import PCA as PCA
import matplotlib.pyplot as plt 
from sklearn import svm
from sklearn.metrics import accuracy_score
pcs = [2,5,10,50,100,150,200,250,300,400]
pca_pc = []
pca_pc_1 = []
pca_pc_2 = []
pca_pc_3 = []
pca_pc_4 = []
pca_pc_5 = []
pca_pc_6 = []
#plot the change of accuracy for each class
for pcs_num in pcs:
    pca = PCA(n_components = pcs_num)
    pca_train_x = pca.fit_transform(train_x)
    pca_test_x = pca.transform(test_x)
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(pca_train_x, train_y)
    pca_pred_y = clf.predict(pca_test_x) 
    test_y_1 = []
    pca_pred_y_1 = []
    test_y_2 = []
    pca_pred_y_2 = []
    test_y_3 = []
    pca_pred_y_3 = []
    test_y_4 = []
    pca_pred_y_4 = []
    test_y_5 = []
    pca_pred_y_5 = []
    test_y_6 = []
    pca_pred_y_6 = []
    for i in range(len(test_y)):
        if (test_y[i] == 1):
            test_y_1.append(test_y[i]);
            pca_pred_y_1.append(pca_pred_y[i])
        elif (test_y[i] == 2):
            test_y_2.append(test_y[i]);
            pca_pred_y_2.append(pca_pred_y[i])
        elif (test_y[i] == 3):
            test_y_3.append(test_y[i]);
            pca_pred_y_3.append(pca_pred_y[i])
        elif (test_y[i] == 4):
            test_y_4.append(test_y[i]);
            pca_pred_y_4.append(pca_pred_y[i])
        elif (test_y[i] == 5):
            test_y_5.append(test_y[i]);
            pca_pred_y_5.append(pca_pred_y[i])
        elif (test_y[i] == 6):
            test_y_6.append(test_y[i]);
            pca_pred_y_6.append(pca_pred_y[i])
    pca_pc.append(accuracy_score(test_y,pca_pred_y,));
    pca_pc_1.append(accuracy_score(test_y_1,pca_pred_y_1,))
    pca_pc_2.append(accuracy_score(test_y_2,pca_pred_y_2,))
    pca_pc_3.append(accuracy_score(test_y_3,pca_pred_y_3,))
    pca_pc_4.append(accuracy_score(test_y_4,pca_pred_y_4,))
    pca_pc_5.append(accuracy_score(test_y_5,pca_pred_y_5,))
    pca_pc_6.append(accuracy_score(test_y_6,pca_pred_y_6,))
    
import matplotlib.pyplot as plt
font = {'size'   : 22}
plt.rc('font', **font) 
plt.figure(figsize=(15,15))
line1, = plt.plot(pcs,pca_pc_1,label='Walking')
line2, = plt.plot(pcs,pca_pc_2,label='Walking up stairs')
line3, = plt.plot(pcs,pca_pc_3,label='Walking down Stairs')
line4, = plt.plot(pcs,pca_pc_4,label='Sitting')
line5, = plt.plot(pcs,pca_pc_5,label='Standing')
line6, = plt.plot(pcs,pca_pc_6,label='Laying')
plt.legend(handles=[line1, line2, line3, line4, line5, line6])
plt.xlabel('n_components',fontsize=22)
plt.ylabel('accuracy',fontsize=22)
plt.title('pca',fontsize=22)
#plt.savefig('/Users/zhangsi929/Desktop/pca+svm.png')
plt.rc('font', **font)
plt.grid(True)
#plot for general accuracy with all six activities
# font = {'size'   : 22}
# plt.rc('font', **font) 
# plt.figure(figsize=(15,15))
# plt.plot(pcs,pca_pc)
# plt.xlabel('n_components',fontsize=22)
# plt.ylabel('accuracy',fontsize=22)
# plt.title('pca',fontsize=22)
# #plt.savefig('/Users/zhangsi929/Desktop/pca+svm.png')
# plt.rc('font', **font)
# plt.grid(True)


# In[11]:


# svd + svm
from sklearn.decomposition import TruncatedSVD
pcs = [2,5,10,50,100,150,200,250,300,400]
svd_pc = []
for pcs_num in pcs:
    svd = TruncatedSVD(n_components=pcs_num, n_iter=7)
    svd_train_x = svd.fit(train_x).transform(train_x)
    svd_test_x = svd.transform(test_x)
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(svd_train_x, train_y)
    svd_pred_y = clf.predict(svd_test_x)
    svd_pc.append(accuracy_score(test_y,svd_pred_y))
    
import matplotlib.pyplot as plt
font = {'size'   : 22}
plt.rc('font', **font) 
plt.figure(figsize=(15,15))
plt.plot(pcs,svd_pc)
plt.xlabel('n_components',fontsize=22)
plt.ylabel('svm_accuracy',fontsize=22)
plt.title('svd',fontsize=22)
#plt.savefig('/Users/zhaobi/Desktop/svd.png')
plt.rc('font', **font)
plt.show()


# In[12]:


#svd vs pca vs isomap
import matplotlib.pyplot as plt
font = {'size'   : 22}
plt.rc('font', **font) 
plt.figure(figsize=(15,15))
line1, = plt.plot(pcs,svd_pc,'r--',label='SVD')
line2, = plt.plot(pcs,pca_pc,'b--',label='PCA')
line3, = plt.plot(pcs,iso_pc, 'y--',label='Isomap')
plt.xlabel('n_components',fontsize=22)
plt.ylabel('accuracy',fontsize=22)
plt.title('PCA vs SVD vs Isomap',fontsize=22)
plt.legend(handles=[line1, line2, line3])
#plt.savefig('/Users/zhaobi/Desktop/pca_svd.png')
plt.rc('font', **font)
plt.grid(True)


# In[81]:


from sklearn.decomposition import PCA
pca = PCA(n_components = 250)
pca_train_x = pca.fit_transform(train_x)
pca_test_x = pca.transform(test_x)


# In[35]:


# broadth vs ann performance
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
n_nodes = [6,9,12,15,20]
ann_acc_nodes = []
# Adding the input layer and the first hidden layer
for nodes in n_nodes:
    ann_classifier = Sequential()
    ann_classifier.add(Dense(units = nodes, kernel_initializer = 'uniform', activation = 'relu', input_dim = 250))
# Adding the second hidden layer
    ann_classifier.add(Dense(units = nodes, kernel_initializer = 'uniform', activation = 'relu'))
# Adding the output layer
    ann_classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'softmax'))
# Compiling the ANN
    ann_classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    ann_classifier.fit(np.array(pca_train_x), train_y_bi, batch_size = 10, epochs = 20)   
# fit data in ann classifier
    ann_pred = ann_classifier.predict(np.array(pca_test_x))
    ann_acc_nodes.append(accuracy_score(test_y_bi, ann_pred.round()))


# In[51]:



font = {'size'   : 22}
plt.rc('font', **font) 
plt.figure(figsize=(15,15))
plt.plot(n_nodes, ann_acc_nodes)
plt.xlabel('n_nodes',fontsize=22)
plt.ylabel('ann_acc',fontsize=22)
#plt.yticks(np.arange(0.8,1.05,0.05))
plt.title('ann',fontsize=22)
#plt.savefig('/Users/chenqi/Documents/681final/ann_nodes_origin.png')
plt.show()


# In[97]:


ann_classifier = Sequential()
ann_classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 250))
# Adding the second hidden layer
ann_classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))
# Adding the output layer
ann_classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'softmax'))
# Compiling the ANN
ann_classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
ann_classifier.fit(np.array(pca_train_x), train_y_bi, batch_size = 10, epochs = 20) 
# fit data in ann classifier
pred_y_ann = ann_classifier.predict_proba(pca_test_x)


# In[174]:


pred_origin_label = [0] * pred_y_ann.shape[0]
for i in range(0, pred_y_ann.shape[0]):
    for j in range(pred_y_ann.shape[1]):
        mx = pred_y_ann[i][0]
        if pred_y_ann[i][j] >= mx:
            mx = pred_y_ann[i][j]
            pred_origin_label[i] = j + 1


# In[175]:


confusion_matrix(test_y, pred_origin_label)


# In[86]:


fpr_ann = dict()
tpr_ann = dict()
roc_auc_ann = dict()
graph_label = ['walking', 'up_stairs', 'down_stairs', 'sitting', 'standing', 'laying']

for i in range(6):
    fpr_ann[i], tpr_ann[i], _ = roc_curve(test_y_bi[:, i], pred_y_ann[:, i])
    roc_auc_ann[i] = auc(fpr_ann[i], tpr_ann[i])
    font = {'size'   : 22}
    plt.rc('font', **font) 
    plt.figure(figsize=(15,15))
    plt.plot(fpr_ann[i],tpr_ann[i])
    plt.xlabel('FPR',fontsize=22)
    plt.ylabel('TPR',fontsize=22)
    plt.title(graph_label[i],fontsize=22)
        #plt.savefig('ROC(RF).png')
    plt.rc('font', **font)
    plt.show()


# In[137]:


plt.figure()
graph_label = ['walking', 'up_stairs', 'down_stairs', 'sitting', 'standing', 'laying']

for i in range(6):
    fpr = [fpr_rf[i], fpr_svm[i], fpr_ann[i]]
    tpr = [tpr_rf[i], tpr_svm[i], tpr_ann[i]]
    roc_auc = [roc_auc_rf[i], roc_auc_svm[i], roc_auc_ann[i]]
    
    font = {'size'   : 22}
    plt.rc('font', **font) 
    plt.figure(figsize=(15,15))
    line1,=plt.plot(fpr[0],tpr[0],'r--',label='rf' + ' auc = ' + str(roc_auc[0]))
    line2,=plt.plot(fpr[1],tpr[1],'b--',label='svm' + ' auc = ' + str(roc_auc[1]))
    line3,=plt.plot(fpr[2],tpr[2],'g--',label='ann' + ' auc = ' + str(roc_auc[2]))
    plt.xlabel('fpr',fontsize=22)
    plt.ylabel('tpr',fontsize=22)
    plt.title('roc of ' + graph_label[i],fontsize=22)
    plt.legend(handles=[line1, line2, line3])
    #plt.savefig('/Users/zhaobi/Desktop/pca_svd.png')
    plt.rc('font', **font)
    plt.show()    


# In[94]:


n_layers = [2,4,6,12]
ann_acc_layers = []
for layer in n_layers:
    ann_classifier = Sequential()
    ann_classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 250))
    for l in range(0, layer):
        ann_classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))
    ann_classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'softmax'))
# Compiling the ANN
    ann_classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    ann_classifier.fit(np.array(pca_train_x), train_y_bi, batch_size = 10, epochs = 20)
    ann_pred = ann_classifier.predict(np.array(pca_test_x))
    ann_acc_layers.append(accuracy_score(test_y_bi, ann_pred.round()))


# In[105]:


#find the best number of trees
from sklearn.ensemble import RandomForestClassifier
nTrees = [1,5,10,15,20,50,70,100,150,200,300,400]
rf_pc = []
for ntree in nTrees:
    clf = RandomForestClassifier(n_estimators=ntree, max_depth=None, min_samples_split=2, random_state=0)
    clf = clf.fit(pca_train_x, train_y)
    rf_pred_y = clf.predict(pca_test_x)
    rf_pc.append(accuracy_score(test_y, rf_pred_y))

import matplotlib.pyplot as plt
font = {'size'   : 22}
plt.rc('font', **font) 
plt.figure(figsize=(15,15))
plt.plot(nTrees, rf_pc)
plt.xlabel('nTrees',fontsize=22)
plt.ylabel('accuracy',fontsize=22)
plt.title('random forest',fontsize=22)
#plt.savefig('/Users/zhaobi/Desktop/rf.png')
plt.show()


# In[198]:


# cm of rf
confusion_matrix(test_y, rf_pred_y)

import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix

font = {'size'   : 12}
plt.rc('font', **font) 
plt.figure(figsize=(100,100))
fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(test_y, rf_pred_y))
fig.set_size_inches(5, 5)
plt.title('Random Forest',fontsize=22)
plt.show()


# In[110]:


#select the best kernel
svm_pc = []
svm_pred_y = []
for kernel in ('poly', 'rbf','linear'):
    clf = svm.SVC(kernel=kernel, gamma=2,probability=True)
    clf = clf.fit(pca_train_x, train_y)
    tmp_pred_y = clf.predict(pca_test_x)
    svm_pred_y.append(tmp_pred_y)
    svm_pc.append(accuracy_score(test_y,tmp_pred_y))


# In[199]:


rf_pc


# In[176]:


# cm of svmlinear
confusion_matrix(test_y, svm_pred_y[2])


# In[121]:


clf = svm.SVC(C=1.0, kernel='poly', degree=3, gamma='auto')
clf = clf.fit(pca_train_x, train_y)
tmp_pred_y = clf.predict(pca_test_x)
svm_pred_y[3] = tmp_pred_y
svm_pc[3] = accuracy_score(test_y,tmp_pred_y)


# In[122]:


svm_pc


# In[123]:


confusion_matrix(test_y, svm_pred_y[3])


# In[195]:


from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
font = {'size'   : 12}
plt.rc('font', **font) 
plt.figure(figsize=(100,100))
fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(test_y, svm_pred_y[2]))
fig.set_size_inches(5, 5)
plt.title('svm-linear',fontsize=22)
plt.show()


# In[196]:


from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
font = {'size'   : 12}
plt.rc('font', **font) 
plt.figure(figsize=(100,100))
fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(test_y, svm_pred_y[0]))
fig.set_size_inches(5, 5)
plt.title('svm-polynomial',fontsize=22)
plt.show()


# In[197]:


from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
font = {'size'   : 12}
plt.rc('font', **font) 
plt.figure(figsize=(100,100))
fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(test_y, svm_pred_y[1]))
fig.set_size_inches(5, 5)
plt.title('svm-rbf',fontsize=22)
plt.show()


# In[177]:


# sfs
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
clf = svm.SVC(decision_function_shape='ovo')
sfs = SFS(clf, 
           k_features=30, 
           forward=True, 
           floating=False, 
           verbose=2,
           scoring='accuracy',
           cv=0)
train_x_arr = np.asarray(train_x)
train_y_arr = np.asarray(train_y)
sfs = sfs.fit(train_x_arr, train_y_arr)


# In[ ]:


#plot sfs
fig1 = plot_sfs(sfs.get_metric_dict(), kind='std_dev')
plt.ylim([0.8, 1])
plt.title('Sequential Forward Selection (w. StdDev)')
plt.grid()
plt.show()

