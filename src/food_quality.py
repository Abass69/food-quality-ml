"""Reproducible measurement-based sensory quality classification."""
from pathlib import Path
import argparse, hashlib, json, platform, zipfile, urllib.request
import numpy as np
import pandas as pd
import sklearn
import joblib
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
FEATURES = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'wine_type']
LABELS = ['lower', 'standard', 'higher']
URL = 'https://archive.ics.uci.edu/static/public/186/wine+quality.zip'

def grades(scores):
    return np.where(np.asarray(scores) <= 5, 0, np.where(np.asarray(scores) == 6, 1, 2))

def load_data():
    path = ROOT / 'data/raw/wine-quality.zip'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(URL, path)
    frames = []
    with zipfile.ZipFile(path) as z:
        for kind, code in [('red', 0), ('white', 1)]:
            df = pd.read_csv(z.open(f'winequality-{kind}.csv'), sep=';')
            df['wine_type'] = code
            df['source_id'] = [f'{kind}:{i+1}' for i in range(len(df))]
            frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    df = raw.drop_duplicates(subset=FEATURES + ['quality']).reset_index(drop=True)
    groups = pd.util.hash_pandas_object(df[FEATURES], index=False).to_numpy()
    audit = {'raw_rows': len(raw), 'unique_rows': len(df), 'exact_duplicates_removed': len(raw)-len(df),
             'unique_measurement_profiles': len(np.unique(groups)), 'missing_values': int(df[FEATURES+['quality']].isna().sum().sum()),
             'archive_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'source_url': URL,
             'source_counts': raw.wine_type.value_counts().sort_index().to_dict(),
             'original_score_counts': df.quality.value_counts().sort_index().to_dict()}
    return df, groups, audit

def metrics(y, p):
    return {'accuracy': accuracy_score(y,p), 'balanced_accuracy': balanced_accuracy_score(y,p),
            'macro_f1': f1_score(y,p,labels=[0,1,2],average='macro',zero_division=0),
            'ordinal_band_mae': float(np.mean(np.abs(y-p)))}

def validate_input(df):
    if set(df.columns) != set(FEATURES):
        raise ValueError(f'CSV must have exactly these columns: {FEATURES}')
    df = df[FEATURES].apply(pd.to_numeric, errors='raise')
    if len(df) == 0 or not np.isfinite(df.to_numpy()).all():
        raise ValueError('Provide at least one row, with finite measurements and no missing values.')
    if not df.wine_type.isin([0,1]).all():
        raise ValueError('wine_type must be 0 (red) or 1 (white).')
    return df

def train():
    out = ROOT/'results'; out.mkdir(exist_ok=True)
    df, groups, audit = load_data()
    X, y = df[FEATURES], grades(df.quality)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr, te = next(outer.split(X,y,groups))
    assert not set(groups[tr]) & set(groups[te])
    cv = list(StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED+1).split(X.iloc[tr],y[tr],groups[tr]))
    for a,b in cv:
        assert not set(groups[tr][a]) & set(groups[tr][b])
    models = {
        'majority_baseline': DummyClassifier(strategy='most_frequent'),
        'logistic_regression': make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced',random_state=SEED)),
        'random_forest': RandomForestClassifier(n_estimators=300,min_samples_leaf=2,max_features='sqrt',class_weight='balanced_subsample',random_state=SEED,n_jobs=2),
        'gradient_boosting': HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=15,l2_regularization=1,random_state=SEED)
    }
    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X.iloc[tr], y[tr], cv=cv, scoring='f1_macro',n_jobs=1)
        cv_results[name] = {'mean_macro_f1':float(scores.mean()),'std_macro_f1':float(scores.std()),'fold_scores':scores.tolist()}
        print(name,cv_results[name],flush=True)
    selected = max(cv_results,key=lambda n: cv_results[n]['mean_macro_f1'])
    # Freeze model selection before inspecting test predictions.
    model = models[selected].fit(X.iloc[tr],y[tr])
    pred = model.predict(X.iloc[te])
    baseline = models['majority_baseline'].fit(X.iloc[tr],y[tr]).predict(X.iloc[te])
    rng = np.random.default_rng(SEED)
    ug = np.unique(groups[te]); members = {g:np.flatnonzero(groups[te]==g) for g in ug}
    boot=[]
    for _ in range(1000):
        ids = np.concatenate([members[g] for g in rng.choice(ug,len(ug),replace=True)])
        boot.append(f1_score(y[te][ids],pred[ids],labels=[0,1,2],average='macro',zero_division=0))
    report = {'seed':SEED,'selected_model':selected,'train_rows':len(tr),'test_rows':len(te),
              'audit':audit,'cross_validation':cv_results,'test_metrics':metrics(y[te],pred),
              'baseline_test_metrics':metrics(y[te],baseline),
              'macro_f1_cluster_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist(),
              'classification_report':classification_report(y[te],pred,labels=[0,1,2],target_names=LABELS,output_dict=True,zero_division=0),
              'confusion_matrix':confusion_matrix(y[te],pred,labels=[0,1,2]).tolist(),
              'subgroups':{kind:metrics(y[te][X.iloc[te].wine_type.to_numpy()==code],pred[X.iloc[te].wine_type.to_numpy()==code]) for kind,code in [('red',0),('white',1)]},
              'environment':{'python':platform.python_version(),'sklearn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__}}
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    assignments=df[['source_id']].copy(); assignments['group']=groups.astype(str); assignments['split']='train'; assignments.loc[te,'split']='test'
    assignments.to_csv(out/'split_assignments.csv',index=False)
    predictions=df.iloc[te][['source_id','quality','wine_type']].copy(); predictions['true_grade']=y[te]; predictions['predicted_grade']=pred
    predictions.to_csv(out/'test_predictions.csv',index=False)
    importance=permutation_importance(model,X.iloc[te],y[te],scoring='f1_macro',n_repeats=10,random_state=SEED,n_jobs=1)
    imp=pd.DataFrame({'feature':FEATURES,'mean_f1_drop':importance.importances_mean,'std':importance.importances_std}).sort_values('mean_f1_drop',ascending=False)
    imp.to_csv(out/'permutation_importance.csv',index=False)
    joblib.dump({'model':model,'features':FEATURES,'min':X.iloc[tr].min(),'max':X.iloc[tr].max()},out/'model.joblib',compress=3)
    X.iloc[te[:3]].to_csv(ROOT/'examples/measurements.csv',index=False)
    plot_results(report,imp,out)
    print(json.dumps({k:report[k] for k in ['selected_model','test_metrics','macro_f1_cluster_bootstrap_95ci']},indent=2))

def plot_results(report,imp,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    cm=np.asarray(report['confusion_matrix']); axes[0].imshow(cm,cmap='Blues')
    for i in range(3):
        for j in range(3): axes[0].text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else '#17324d')
    axes[0].set(xticks=range(3),yticks=range(3),xticklabels=LABELS,yticklabels=LABELS,xlabel='Predicted grade',ylabel='Observed grade',title='Held-out confusion matrix')
    top=imp.head(6).iloc[::-1]; axes[1].barh(top.feature,top.mean_f1_drop,xerr=top['std'],color='#16857b')
    axes[1].set(xlabel='Macro-F1 decrease after permutation',title='Predictive associations (not causal)')
    fig.savefig(out/'evaluation.png',dpi=180); plt.close(fig)

def predict(input_path,output_path):
    bundle=joblib.load(ROOT/'results/model.joblib') # Load only a trusted local artifact.
    X=validate_input(pd.read_csv(input_path))
    p=bundle['model'].predict(X)
    result=pd.DataFrame({'predicted_grade':[LABELS[i] for i in p],
      'outside_training_range':((X<bundle['min'])|(X>bundle['max'])).any(axis=1),
      'scope':'sensory quality only; not a food safety assessment'})
    result.to_csv(output_path,index=False)
    print(result.to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('train'); p=sub.add_parser('predict'); p.add_argument('--input',required=True); p.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.command=='train': train()
    else: predict(args.input,args.output)
