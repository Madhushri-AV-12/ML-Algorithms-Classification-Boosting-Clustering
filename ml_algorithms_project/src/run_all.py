from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, mean_absolute_error,
                             mean_squared_error, r2_score, silhouette_score)

ROOT = Path(__file__).resolve().parents[1]
DATA, RESULTS, PLOTS = ROOT/'data', ROOT/'results', ROOT/'plots'
for p in (RESULTS, PLOTS): p.mkdir(parents=True, exist_ok=True)
all_metrics = []

# 1. Classification: Social Network Ads
ads = pd.read_csv(DATA/'Social_Network_Ads.csv')
# User ID is an identifier, not a useful predictive feature.
X = ads.drop(columns=['Purchased', 'User ID'])
y = ads['Purchased']
cat = X.select_dtypes(include=['object']).columns.tolist()
num = X.select_dtypes(exclude=['object']).columns.tolist()
pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat), ('num', 'passthrough', num)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
classifiers = {
    'RandomForestClassifier': RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced'),
    'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=2, random_state=42)
}
class_results = {}
for name, model in classifiers.items():
    pipe = Pipeline([('preprocess', pre), ('model', model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    m = {'task':'classification','dataset':'Social_Network_Ads.csv','algorithm':name,
         'accuracy':float(accuracy_score(y_test,pred)), 'precision_weighted':float(precision_score(y_test,pred,average='weighted',zero_division=0)),
         'recall_weighted':float(recall_score(y_test,pred,average='weighted',zero_division=0)), 'f1_weighted':float(f1_score(y_test,pred,average='weighted',zero_division=0))}
    all_metrics.append(m); class_results[name]=m
    pd.DataFrame(classification_report(y_test,pred,output_dict=True,zero_division=0)).T.to_csv(RESULTS/f'{name.lower()}_classification_report.csv')
    cm=confusion_matrix(y_test,pred)
    fig,ax=plt.subplots(figsize=(5,4)); im=ax.imshow(cm); ax.set(title=f'{name} — Confusion Matrix',xlabel='Predicted',ylabel='Actual',xticks=[0,1],yticks=[0,1]);
    for (i,j),v in np.ndenumerate(cm): ax.text(j,i,str(v),ha='center',va='center')
    fig.tight_layout(); fig.savefig(PLOTS/f'{name.lower()}_confusion_matrix.png',dpi=160); plt.close(fig)
    pd.DataFrame({'actual':y_test.to_numpy(),'predicted':pred}).to_csv(RESULTS/f'{name.lower()}_predictions.csv',index=False)

# 2. Boosting regression: Insurance charges
ins = pd.read_csv(DATA/'insurance_pre.csv')
Xr, yr = ins.drop(columns=['charges']), ins['charges']
catr = Xr.select_dtypes(include=['object']).columns.tolist(); numr=Xr.select_dtypes(exclude=['object']).columns.tolist()
prer = ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),catr),('num','passthrough',numr)])
Xtr,Xte,ytr,yte=train_test_split(Xr,yr,test_size=0.2,random_state=42)
reg=Pipeline([('preprocess',prer),('model',GradientBoostingRegressor(n_estimators=250,learning_rate=0.03,max_depth=2,loss='squared_error',random_state=42))])
reg.fit(Xtr,ytr); rp=reg.predict(Xte)
regm={'task':'regression/boosting','dataset':'insurance_pre.csv','algorithm':'GradientBoostingRegressor',
      'mae':float(mean_absolute_error(yte,rp)),'rmse':float(np.sqrt(mean_squared_error(yte,rp))),'r2':float(r2_score(yte,rp))}
all_metrics.append(regm)
pd.DataFrame({'actual_charges':yte.to_numpy(),'predicted_charges':rp}).to_csv(RESULTS/'insurance_boosting_predictions.csv',index=False)
fig,ax=plt.subplots(figsize=(6,5)); ax.scatter(yte,rp,alpha=.7); lo=min(float(yte.min()),float(rp.min())); hi=max(float(yte.max()),float(rp.max())); ax.plot([lo,hi],[lo,hi],linestyle='--'); ax.set(title='Insurance Charges: Actual vs Predicted',xlabel='Actual charges',ylabel='Predicted charges'); fig.tight_layout(); fig.savefig(PLOTS/'insurance_boosting_actual_vs_predicted.png',dpi=160); plt.close(fig)

# 3. Clustering: Mall customers
mall=pd.read_csv(DATA/'Mall_Customers.csv')
features=['Annual Income (k$)','Spending Score (1-100)']
Xm=mall[features].copy(); scaler=StandardScaler(); Z=scaler.fit_transform(Xm)
# Select k from 2..8 using silhouette score, without using CustomerID or demographic labels.
cluster_scores=[]
for k in range(2,9):
    km=KMeans(n_clusters=k,random_state=42,n_init=20); labels=km.fit_predict(Z)
    cluster_scores.append({'k':k,'silhouette_score':float(silhouette_score(Z,labels)),'inertia':float(km.inertia_)})
best=max(cluster_scores,key=lambda z:z['silhouette_score']); kbest=best['k']
km=KMeans(n_clusters=kbest,random_state=42,n_init=20); labels=km.fit_predict(Z)
out=mall.copy(); out['Cluster']=labels; out.to_csv(RESULTS/'mall_customers_cluster_assignments.csv',index=False)
cluster_m={'task':'clustering','dataset':'Mall_Customers.csv','algorithm':'KMeans','selected_k':int(kbest),'silhouette_score':best['silhouette_score'],'rows_clustered':int(len(mall))}
all_metrics.append(cluster_m)
pd.DataFrame(cluster_scores).to_csv(RESULTS/'mall_k_selection_scores.csv',index=False)
fig,ax=plt.subplots(figsize=(7,5)); sc=ax.scatter(mall[features[0]],mall[features[1]],c=labels,s=45); ax.set(title=f'Mall Customer Segments (KMeans, k={kbest})',xlabel='Annual Income (k$)',ylabel='Spending Score (1–100)'); fig.tight_layout(); fig.savefig(PLOTS/'mall_customer_clusters.png',dpi=160); plt.close(fig)
summary={'classification':class_results,'insurance_boosting_regression':regm,'mall_clustering':cluster_m,'k_selection_scores':cluster_scores,
         'notes':['Metrics are from held-out test data for classification and regression.','KMeans k was selected by the highest silhouette score among k=2..8.','Results are dataset-specific and should not be interpreted as guarantees of real-world performance.']}
(RESULTS/'summary.json').write_text(json.dumps(summary,indent=2))
pd.DataFrame(all_metrics).to_csv(RESULTS/'metrics_summary.csv',index=False)
print('\n=== RESULTS ===')
print(pd.DataFrame(all_metrics).to_string(index=False))
print('\nK selection scores:')
print(pd.DataFrame(cluster_scores).to_string(index=False))
print(f'\nSelected K={kbest}; silhouette={best["silhouette_score"]:.4f}')
