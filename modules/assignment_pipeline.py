"""Reproducible text classification pipeline for the course assignment.

Fit and select on train/validation only. Evaluate each selected family once on
held-out test data. Artifacts are regenerated; historical reports are untouched.
"""
from __future__ import annotations
import json, random, re, time, platform
from importlib.metadata import version
from collections import Counter
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
REGISTRY = {
 'ag_news': ('fancyzhx/ag_news', 10000, 2000, 96),
 'rotten_tomatoes': ('cornell-movie-review-data/rotten_tomatoes', None, None, 64),
 'banking77': ('mteb/banking77', None, None, 64),
}
MODEL_ID='BAAI/bge-small-en-v1.5'
TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
NEGATION = re.compile(r"\b(not|no|never|nothing|nobody|none|cannot|nor|without)\b|n't\b", re.I)

def runtime_metadata():
 return {'python':platform.python_version(), **{n:version(n) for n in ('numpy','pandas','scikit-learn','scipy','torch','transformers','datasets','h5py')}}

def seed_all(seed=SEED):
 import torch
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def words(s):
 return TOKEN.findall(str(s).lower())

def multiclass_brier(y, proba):
 """Mean over samples of sum_k (p_k - 1[y=k])^2."""
 y=np.asarray(y); proba=np.asarray(proba,dtype=np.float64)
 onehot=np.zeros_like(proba); onehot[np.arange(len(y)),y]=1.0
 return float(np.mean(np.sum((proba-onehot)**2,axis=1)))

def expected_calibration_error(y, proba, n_bins=15):
 """Top-label ECE over equal-width confidence bins."""
 proba=np.asarray(proba,dtype=np.float64); y=np.asarray(y)
 conf=proba.max(axis=1); correct=(proba.argmax(axis=1)==y).astype(float)
 edges=np.linspace(0,1,n_bins+1); ece=0.0
 for lo,hi in zip(edges[:-1],edges[1:]):
  sel=(conf>lo)&(conf<=hi)
  if sel.any(): ece+=sel.mean()*abs(correct[sel].mean()-conf[sel].mean())
 return float(ece)

def align_proba(classes, proba, n_classes):
 out=np.zeros((proba.shape[0],n_classes),dtype=np.float64)
 out[:,np.asarray(classes,dtype=int)]=proba
 return out

def metrics(y, pred, proba=None):
 p,r,f,_=precision_recall_fscore_support(y,pred,average='macro',zero_division=0)
 out={'accuracy':accuracy_score(y,pred),'precision_macro':p,'recall_macro':r,'f1_macro':f,
      'brier_score':np.nan,'ece':np.nan}
 if proba is not None:
  proba=np.clip(np.asarray(proba,dtype=np.float64),0.0,1.0)
  proba=proba/proba.sum(axis=1,keepdims=True).clip(min=1e-12)
  out['brier_score']=multiclass_brier(y,proba); out['ece']=expected_calibration_error(y,proba)
 return out

def negation_mask(texts):
 return np.array([bool(NEGATION.search(str(t).lower())) for t in texts])

def negation_rows(family, texts, y, pred):
 has=negation_mask(texts); correct=(np.asarray(pred)==np.asarray(y))
 return {'family':family,
         'n_negation':int(has.sum()),'n_other':int((~has).sum()),
         'acc_negation':float(correct[has].mean()) if has.any() else np.nan,
         'acc_other':float(correct[~has].mean()) if (~has).any() else np.nan,
         'gap':float(correct[has].mean()-correct[~has].mean()) if has.any() and (~has).any() else np.nan}

def load_splits(name, quick=False):
 from datasets import load_dataset
 dataset_id, cap_train, cap_test, max_len = REGISTRY[name]
 raw = load_dataset(dataset_id)
 train, test = raw['train'], raw['test']
 quality={}
 for key,part in [('train',train),('test',test)]:
  raw_text=part['text']; valid=[str(t).strip() for t in raw_text if t is not None and str(t).strip()]
  quality[key+'_missing_or_blank']=len(raw_text)-len(valid)
  quality[key+'_duplicate_texts']=len(valid)-len(set(valid))
 names = list(train.features['label'].names) if getattr(train.features['label'], 'names', None) else []
 if not names and 'label_text' in train.column_names:
  pairs = sorted({(int(i),str(s)) for i,s in zip(train['label'],train['label_text'])})
  names = [s for _,s in pairs]
 def extract(split):
  rows=[(str(t).strip(),int(y)) for t,y in zip(split['text'],split['label']) if t is not None and str(t).strip()]
  return [a for a,_ in rows], np.array([b for _,b in rows],dtype=np.int64)
 def cap(text,y,n,seed):
  if n is None or len(y)<=n: return text,y
  ix,_=train_test_split(np.arange(len(y)),train_size=n,stratify=y,random_state=seed)
  ix=np.sort(ix); return [text[i] for i in ix], y[ix]
 texts, labels=extract(train); texts,labels=cap(texts,labels, min(cap_train or 1500,1500) if quick else cap_train, SEED)
 tx,ty=extract(test); tx,ty=cap(tx,ty, min(cap_test or 600,600) if quick else cap_test, SEED+1)
 fit,val=train_test_split(np.arange(len(labels)),test_size=.15,stratify=labels,random_state=SEED)
 fit_set={texts[i] for i in fit}; val_set={texts[i] for i in val}; test_set=set(tx)
 quality.update({'fit_val_exact_overlap_before':len(fit_set & val_set),'fit_test_exact_overlap_before':len(fit_set & test_set),'val_test_exact_overlap_before':len(val_set & test_set)})
 original_fit,original_val=len(fit),len(val)
 val=np.array([i for i in val if texts[i] not in test_set],dtype=int)
 val_set={texts[i] for i in val}
 fit=np.array([i for i in fit if texts[i] not in test_set and texts[i] not in val_set],dtype=int)
 quality.update({'removed_fit_for_overlap':original_fit-len(fit),'removed_val_for_overlap':original_val-len(val),
                 'fit_test_exact_overlap':len({texts[i] for i in fit} & test_set)})
 if set(labels[fit]) != set(labels) or set(labels[val]) != set(labels):
  raise ValueError('Removing exact split overlaps left a class absent from fit or validation; revise the split.')
 return {'fit':([texts[i] for i in fit],labels[fit]), 'val':([texts[i] for i in val],labels[val]), 'test':(tx,ty), 'names':names, 'max_len':max_len,'source':dataset_id,'quality':quality}

def write_sparse_h5(path, X, y):
 with h5py.File(path,'w') as f:
  for k in ('fit','val','test'):
   m=X[k].tocsr(); g=f.create_group(k); g.create_dataset('data',data=m.data,compression='gzip'); g.create_dataset('indices',data=m.indices,compression='gzip'); g.create_dataset('indptr',data=m.indptr); g.attrs['shape']=m.shape
   f.create_dataset('y_'+k,data=y[k])

def read_sparse_h5(path):
 with h5py.File(path,'r') as f:
  return {k:sparse.csr_matrix((f[k]['data'][:],f[k]['indices'][:],f[k]['indptr'][:]),shape=tuple(f[k].attrs['shape'])) for k in ('fit','val','test')}

def vectorizer(stop,ngram,tokenization='regex'):
 extra={} if tokenization=='regex' else {'tokenizer':str.split,'token_pattern':None}
 return TfidfVectorizer(lowercase=True,stop_words='english' if stop else None,ngram_range=(1,ngram),max_features=25000,min_df=2,sublinear_tf=True,dtype=np.float32,**extra)

def make_classifier(kind,c):
 if kind=='nb': return MultinomialNB(alpha=1.0)
 if kind=='logreg': return LogisticRegression(C=c,max_iter=1500,random_state=SEED)
 return LinearSVC(C=c,random_state=SEED,dual='auto',max_iter=5000)

def select_sparse(split, feature_dir, tokenization='regex', classifiers=('nb','logreg','svm')):
 validation=[]; best=None
 n_classes=len(split.get('names') or []) or int(max(split['fit'][1].max(),split['val'][1].max(),split['test'][1].max())+1)
 for stop in (False,True):
  for ng in (1,2):
   vec=vectorizer(stop,ng,tokenization); fit=vec.fit_transform(split['fit'][0]); val=vec.transform(split['val'][0])
   for model,c in [(m,c) for m in classifiers for c in ((1,) if m=='nb' else (1,4))]:
    clf=make_classifier(model,c); t=time.perf_counter(); clf.fit(fit,split['fit'][1]); pred=clf.predict(val)
    proba=align_proba(clf.classes_,clf.predict_proba(val),n_classes) if hasattr(clf,'predict_proba') else None
    row={'family':'TF-IDF','model':model,'stop_words':stop,'max_ngram':ng,'C':c,'fit_sec':time.perf_counter()-t,**metrics(split['val'][1],pred,proba)}
    validation.append(row)
    if best is None or row['f1_macro']>best[0]['f1_macro']: best=(row,vec,clf)
 row,vec,clf=best
 X={'fit':vec.transform(split['fit'][0]),'val':vec.transform(split['val'][0]),'test':vec.transform(split['test'][0])}
 y={k:split[k][1] for k in X}; path=feature_dir/'tfidf.h5'; write_sparse_h5(path,X,y)
 X=read_sparse_h5(path)
 clf=make_classifier(row['model'],row['C']); clf.fit(X['fit'],y['fit'])
 pred=clf.predict(X['test'])
 proba=align_proba(clf.classes_,clf.predict_proba(X['test']),n_classes) if hasattr(clf,'predict_proba') else None
 return validation, row, pred, proba, {'vocabulary_size':len(vec.vocabulary_), 'feature_file':str(path)}

def encode_bge(split, feature_dir, padding, max_length, batch_size):
 import torch
 from transformers import AutoTokenizer, AutoModel
 tokenizer=AutoTokenizer.from_pretrained(MODEL_ID); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 model=AutoModel.from_pretrained(MODEL_ID).to(device); model.eval(); outputs={}
 for k in ('fit','val','test'):
  texts=split[k][0]; vectors=[]
  with torch.inference_mode():
   for start in range(0,len(texts),batch_size):
    # padding can be dynamic or fixed; both use the same attention mask and truncation.
    inp=tokenizer(texts[start:start+batch_size],padding='longest' if padding=='dynamic' else 'max_length',truncation=True,max_length=max_length,return_tensors='pt')
    inp={key:value.to(device) for key,value in inp.items()}
    z=model(**inp).last_hidden_state[:,0]; z=torch.nn.functional.normalize(z,p=2,dim=1)
    vectors.append(z.cpu().numpy().astype('float32'))
    if start == 0 or (start // batch_size + 1) % 50 == 0:
     print(f'  {k}: {min(start + batch_size, len(texts))}/{len(texts)}',flush=True)
  np.save(feature_dir/('bge_'+k+'.npy'),np.concatenate(vectors))
  outputs[k]=np.load(feature_dir/('bge_'+k+'.npy'),mmap_mode='r')
 del model
 if device.type=='cuda': torch.cuda.empty_cache()
 return outputs

def select_dense(split, X):
 validation=[]; best=None
 n_classes=len(split.get('names') or []) or int(max(split['fit'][1].max(),split['val'][1].max(),split['test'][1].max())+1)
 for c in (0.5,2.0,8.0):
  model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1500,random_state=SEED))
  t=time.perf_counter(); model.fit(X['fit'],split['fit'][1]); pred=model.predict(X['val'])
  proba=align_proba(model.classes_,model.predict_proba(X['val']),n_classes)
  row={'family':'BGE-small','model':'logreg','C':c,'fit_sec':time.perf_counter()-t,**metrics(split['val'][1],pred,proba)}; validation.append(row)
  if best is None or row['f1_macro']>best[0]['f1_macro']: best=(row,model)
 pred=best[1].predict(X['test']); proba=align_proba(best[1].classes_,best[1].predict_proba(X['test']),n_classes)
 return validation,best[0],pred,proba

def train_textcnn(split, max_length, epochs, batch_size):
 import torch
 from torch import nn
 from torch.utils.data import TensorDataset,DataLoader
 seed_all(); counts=Counter(w for t in split['fit'][0] for w in words(t)); vocab={w:i+2 for i,(w,n) in enumerate(counts.most_common(19998)) if n>=2}
 def encode(texts):
  x=np.zeros((len(texts),max_length),dtype=np.int64)
  for i,t in enumerate(texts):
   ids=[vocab.get(w,1) for w in words(t)][:max_length]
   if not ids: ids=[1]
   x[i,:len(ids)]=ids
  return torch.from_numpy(x)
 tensors={k:encode(split[k][0]) for k in ('fit','val','test')}
 class TextCNN(nn.Module):
  def __init__(self):
   super().__init__(); self.emb=nn.Embedding(len(vocab)+2,100,padding_idx=0); self.convs=nn.ModuleList([nn.Conv1d(100,64,k) for k in (3,4,5)]); self.dropout=nn.Dropout(.5); self.fc=nn.Linear(192,len(split['names']))
  def forward(self,x):
   z=self.emb(x).transpose(1,2); outputs=[]
   for conv in self.convs:
    h=torch.relu(conv(z)); outputs.append(h.amax(dim=2))
   return self.fc(self.dropout(torch.cat(outputs,dim=1)))
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=TextCNN().to(device); optimizer=torch.optim.Adam(model.parameters(),lr=.001); loss=nn.CrossEntropyLoss()
 def loader(k,shuffle=False):
  return DataLoader(TensorDataset(tensors[k],torch.tensor(split[k][1])),batch_size=batch_size,shuffle=shuffle,generator=torch.Generator().manual_seed(SEED))
 def predict(k):
  model.eval(); preds=[]; probs=[]
  with torch.no_grad():
   for x,_ in loader(k):
    logits=model(x.to(device)); preds.extend(logits.argmax(1).cpu().tolist())
    probs.append(torch.softmax(logits,dim=1).cpu().numpy())
  return np.array(preds), np.concatenate(probs,axis=0)
 best=-1.; best_state=None; history=[]; stale=0
 for epoch in range(epochs):
  model.train(); total=0
  for x,y in loader('fit',True):
   x,y=x.to(device),y.to(device); optimizer.zero_grad(); l=loss(model(x),y); l.backward(); optimizer.step(); total+=l.item()*len(y)
  pred,proba=predict('val'); m=metrics(split['val'][1],pred,proba); history.append({'epoch':epoch+1,'train_loss':total/len(tensors['fit']),**m})
  if m['f1_macro']>best:
   best=m['f1_macro']; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
  else:
   stale+=1
   if stale>=2: break
 model.load_state_dict(best_state)
 pred,proba=predict('test')
 return history,pred,proba,len(vocab)

def plot_eda(split,out,name):
 text,y=split['fit']; lens=np.array([len(words(t)) for t in text]); freq=Counter(w for t in text for w in words(t)); top=freq.most_common(15)
 fig,axes=plt.subplots(1,3,figsize=(15,4)); axes[0].hist(lens,bins=30); axes[0].set_title('Train: words per text'); axes[0].set_xlabel('Words')
 counts=np.bincount(y,minlength=len(split['names'])); axes[1].bar(range(len(counts)),counts); axes[1].set_title('Train class counts'); axes[1].set_xlabel('Class ID')
 axes[2].barh([w for w,_ in top][::-1],[n for _,n in top][::-1]); axes[2].set_title('Most frequent words'); fig.suptitle(name); fig.tight_layout(); fig.savefig(out/'eda.png',dpi=140); plt.close(fig)
 return {'n_fit':len(text),'n_val':len(split['val'][0]),'n_test':len(split['test'][0]),'classes':len(counts),'min_train_class':int(counts.min()),'max_train_class':int(counts.max()),'length_mean':float(lens.mean()),'length_p95':float(np.percentile(lens,95)),'vocabulary':len(freq),'top_words':top}

def run_dataset(name,root,quick=False,epochs=7,batch_size=32,padding='dynamic',tokenization='regex',classifiers=('nb','logreg','svm'),embeddings=('tfidf','bge')):
 seed_all(); root=Path(root); out=root/'reports'/name; features=root/'features'/name; out.mkdir(parents=True,exist_ok=True); features.mkdir(parents=True,exist_ok=True)
 split=load_splits(name,quick); print(f'[{name}] fit/val/test:',*(len(split[k][1]) for k in ('fit','val','test')),flush=True)
 eda=plot_eda(split,out,name); eda.update(split['quality']); (out/'eda.json').write_text(json.dumps(eda,indent=2),encoding='utf8')
 val=[]; rows=[]; neg=[]; info={}; history=[]
 if 'tfidf' in embeddings:
  val,choice,pred,proba,info=select_sparse(split,features,tokenization,classifiers)
  rows.append({'family':'TF-IDF','selection':json.dumps({k:choice[k] for k in ('model','stop_words','max_ngram','C')}),**metrics(split['test'][1],pred,proba)})
  neg.append(negation_rows('TF-IDF',split['test'][0],split['test'][1],pred)); np.save(out/'tfidf_predictions.npy',pred)
 np.save(features/'y_fit.npy',split['fit'][1]); np.save(features/'y_val.npy',split['val'][1]); np.save(features/'y_test.npy',split['test'][1])
 if 'bge' in embeddings:
  print(f'[{name}] BGE feature extraction',flush=True); X=encode_bge(split,features,padding,split['max_len'],batch_size)
  v,choice,pred,proba=select_dense(split,X); val+=v
  rows.append({'family':'BGE-small','selection':json.dumps({'model':'logreg','C':choice['C']}),**metrics(split['test'][1],pred,proba)})
  neg.append(negation_rows('BGE-small',split['test'][0],split['test'][1],pred)); np.save(out/'bge_predictions.npy',pred)
 print(f'[{name}] TextCNN',flush=True); history,pred,proba,vocab=train_textcnn(split,split['max_len'],2 if quick else epochs,batch_size)
 rows.append({'family':'TextCNN','selection':json.dumps({'early_stopping':'validation macro-F1','best_epoch':max(history,key=lambda x:x['f1_macro'])['epoch'],'vocab_size':vocab}),**metrics(split['test'][1],pred,proba)})
 neg.append(negation_rows('TextCNN',split['test'][0],split['test'][1],pred)); np.save(out/'textcnn_predictions.npy',pred)
 pd.DataFrame(val).to_csv(out/'validation.csv',index=False)
 if history: pd.DataFrame(history).to_csv(out/'textcnn_history.csv',index=False)
 results=pd.DataFrame(rows); results.to_csv(out/'test_results.csv',index=False)
 pd.DataFrame(neg).to_csv(out/'negation.csv',index=False)
 (out/'run_metadata.json').write_text(json.dumps({'dataset':split['source'],'label_names':split['names'],'seed':SEED,'quick':quick,'padding':padding,'tfidf_tokenization':tokenization,'classifiers':list(classifiers),'embeddings':list(embeddings),'max_length':split['max_len'],'batch_size':batch_size,'epochs_limit':(2 if quick else epochs),'environment':runtime_metadata(),'bge_model':MODEL_ID,'features':info,'data':eda},indent=2),encoding='utf8')
 write_error_analysis(split,out)
 print(results.to_string(index=False),flush=True); return results

def write_error_analysis(split,out):
 """Save per-class statistics and examples for all selected model families."""
 out=Path(out); y=split['test'][1]; names=split['names']; labels=np.arange(len(names))
 present=set(pd.read_csv(out/'test_results.csv')['family'])
 for family in ('tfidf','bge','textcnn'):
  if {'tfidf':'TF-IDF','bge':'BGE-small','textcnn':'TextCNN'}[family] not in present: continue
  path=out/(family+'_predictions.npy')
  if not path.exists(): continue
  pred=np.load(path)
  report=classification_report(y,pred,labels=labels,target_names=names,output_dict=True,zero_division=0)
  pd.DataFrame(report).T.to_csv(out/(family+'_classification_report.csv'))
  errors=[{'text':split['test'][0][i], 'true':names[y[i]], 'predicted':names[pred[i]]} for i in np.flatnonzero(pred!=y)[:30]]
  pd.DataFrame(errors,columns=['text','true','predicted']).to_csv(out/(family+'_errors.csv'),index=False)
  matrix=confusion_matrix(y,pred,labels=labels)
  np.save(out/(family+'_confusion_matrix.npy'),matrix)

def make_report(root,names,quick):
 import textwrap
 root=Path(root); reports=root/'reports'; rows=[]; neg_rows=[]
 for n in names:
  p=reports/n/'test_results.csv'
  if p.exists():
   t=pd.read_csv(p); t.insert(0,'dataset',n); rows.append(t)
  np_=reports/n/'negation.csv'
  if np_.exists():
   ntab=pd.read_csv(np_); ntab.insert(0,'dataset',n); neg_rows.append(ntab)
 if not rows: return
 data=pd.concat(rows,ignore_index=True); data.to_csv(reports/'comparison.csv',index=False)
 if neg_rows: pd.concat(neg_rows,ignore_index=True).to_csv(reports/'negation_comparison.csv',index=False)
 with PdfPages(reports/'assignment_report.pdf') as pdf:
  fig=plt.figure(figsize=(8.27,11.69))
  fig.text(.07,.95,'MACHINE LEARNING ASSIGNMENT',fontsize=17,weight='bold')
  fig.text(.07,.915,'Task 2: Text classification',fontsize=13)
  lines=['HCMUT | Semester I, academic year 2026-2027', '',
   'Course code: [team to complete]    Instructor: [team to complete]',
   'Team members, IDs, email and contribution percentages: [team to complete]',
   'Work distribution and collaboration evidence: [team to complete]', '',
   'OBJECTIVE',
   'Compare a conventional text pipeline, pretrained embeddings and a neural model.', '',
   'EXPERIMENT DESIGN',
   'Fit, validation and test are distinct. The fit set alone builds the TF-IDF vocabulary',
   'and trains models. Validation macro-F1 chooses hyperparameters and early stopping.',
   'Each family is evaluated once on the held-out official test set.',
   'AG News: stratified sample of 10,000 training and 2,000 test records.',
   'Rotten Tomatoes and Banking77: official train and test; train is split 85/15.', '',
   'MODELS AND FEATURES',
   'TF-IDF: stop-word removal on/off; unigrams or unigrams+bigrams; NB alpha=1;',
   'logistic regression and linear SVM C in {1, 4} (20 validation configurations).',
   'BGE-small: frozen pretrained encoder, CLS pooling; logistic regression C in {0.5, 2, 8}.',
   'TextCNN: learned word embeddings and convolutions trained end-to-end from scratch;',
   'Adam learning rate 0.001, dropout 0.5, up to 7 epochs, patience 2.',
   'TF-IDF sparse matrices are stored as HDF5; BGE embeddings are stored as NPY.',
   'The downstream classifiers read these stored features.', '',
   'RELIABILITY AND LIMITATIONS',
   'One random seed and one validation split; no confidence intervals are estimated.',
   'The neural and traditional models have different architectures and training costs.',
   'Padding settings use masks. Dataset selection must be confirmed with the instructor.',
   'Do not use smoke-test metrics as final results. Review class-level errors before submission.', '',
   'PUBLIC SOURCES',
   'https://huggingface.co/datasets/fancyzhx/ag_news',
   'https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes',
   'https://huggingface.co/datasets/mteb/banking77',
   'https://huggingface.co/BAAI/bge-small-en-v1.5']
  fig.text(.07,.875,'\n'.join(lines),fontsize=9,va='top',linespacing=1.5)
  if quick: fig.text(.07,.04,'SMOKE TEST — not final experimental results',color='darkred',fontsize=11,weight='bold')
  pdf.savefig(fig); plt.close(fig)
  for name in names:
   path=reports/name/'run_metadata.json'
   if not path.exists(): continue
   meta=json.loads(path.read_text(encoding='utf8')); d=meta['data']; subset=data[data.dataset==name]
   fig=plt.figure(figsize=(8.27,11.69)); fig.text(.07,.95,f'RESULTS: {name}',fontsize=16,weight='bold')
   fig.text(.07,.921,f"Fit / validation / test: {d['n_fit']} / {d['n_val']} / {d['n_test']}   |   classes: {d['classes']}   |   seed: {SEED}",fontsize=9)
   lines=[f"Fit-set length: mean {d['length_mean']:.1f}, 95th percentile {d['length_p95']:.1f} words; vocabulary {d['vocabulary']}.",
          f"Class counts in fit: minimum {d['min_train_class']}, maximum {d['max_train_class']}.",
          f"Exact fit/test overlaps after filtering: {d.get('fit_test_exact_overlap','not checked')}; removed fit: {d.get('removed_fit_for_overlap','?')}, validation: {d.get('removed_val_for_overlap','?')}.", '',
          'Held-out test results: accuracy, macro precision, macro recall, macro F1']
   for row in subset.itertuples():
    brier=getattr(row,'brier_score',np.nan); ece=getattr(row,'ece',np.nan)
    lines.append(f'{row.family:12s}  {row.accuracy:.4f}   {row.precision_macro:.4f}   {row.recall_macro:.4f}   {row.f1_macro:.4f}   Brier {brier:.4f}   ECE {ece:.4f}')
   neg_path=reports/name/'negation.csv'
   if neg_path.exists():
    lines+=['','Negation vs other accuracy (cue words / n\'t):']
    for r in pd.read_csv(neg_path).itertuples():
     lines.append(f'{r.family:12s}  neg={r.acc_negation:.4f} (n={int(r.n_negation)})  other={r.acc_other:.4f} (n={int(r.n_other)})  gap={r.gap:.4f}')
   lines+=['','Configurations selected on validation:']
   for row in subset.itertuples():
    lines.extend(textwrap.wrap(f'{row.family}: {row.selection}',width=95))
   ranking=subset.sort_values('f1_macro',ascending=False)
   if len(ranking)>1:
    first,second=ranking.iloc[0],ranking.iloc[1]
    lines += ['',f"In this run, {first['family']} has the highest test macro-F1 ({first['f1_macro']:.4f});",
              f"{second['family']} follows at {second['f1_macro']:.4f}. This is descriptive, not model selection."]
   lines+=['Results reflect a single split; inspect per-class CSVs and errors for limitations.',
           'Training-only EDA below: length distribution, class counts and most frequent words.']
   fig.text(.07,.88,'\n'.join(lines),fontsize=9,va='top',linespacing=1.5)
   img=plt.imread(reports/name/'eda.png'); ax=fig.add_axes([.055,.09,.89,.34]); ax.imshow(img); ax.axis('off')
   fig.text(.07,.045,'Validation tables, confusion matrices and example errors accompany this report.',fontsize=8)
   if quick: fig.text(.67,.95,'SMOKE TEST',color='darkred',fontsize=10,weight='bold')
   pdf.savefig(fig); plt.close(fig)

 # Editable companion report; no fabricated identities or experiment values.
 doc=['# Text classification: machine learning assignment', '', 'HCMUT — Semester I, 2026–2027', '', '**Run:** ' + ('SMOKE TEST — not final results' if quick else 'Full configured datasets'), '', '## Team and work distribution', '', '| Name | Student ID | Email | Actual responsibility | Contribution |', '| --- | --- | --- | --- | --- |', '| [to complete] | [to complete] | [to complete] | [to complete] | [%] |', '| [to complete] | [to complete] | [to complete] | [to complete] | [%] |', '', 'Instructor and course code: [to complete]. Percentages must total 100%. Add real collaboration evidence.', '', '## Objective and method', '', 'Compare TF-IDF with traditional classifiers, frozen BGE-small embeddings with logistic regression, and end-to-end TextCNN. A stratified 85/15 split of the original training data is used for fitting and validation. The official test split is held out. AG News is stratified-sampled to 10,000 training and 2,000 test examples; Rotten Tomatoes and Banking77 use their original train/test sets. Rotten Tomatoes official validation is not used.', '', 'The fit set alone is used to fit the TF-IDF vocabulary and classifiers. TF-IDF validation compares stop-word removal on/off, unigrams versus unigrams plus bigrams, Naive Bayes (alpha=1), and logistic regression or linear SVM (C=1 or 4): 20 configurations. BGE logistic regression compares C=0.5, 2 and 8. TextCNN uses 100-dimensional trainable embeddings, convolution widths 3/4/5, dropout 0.5, Adam learning rate 0.001, up to 7 epochs and patience 2. Validation macro-F1 selects TF-IDF configuration, BGE regularization and the TextCNN early-stopping epoch. One chosen configuration per family is evaluated on test. Frozen BGE-small is not end-to-end fine-tuning. Both dynamic and fixed padding use masks and identical maximum lengths.', '', '## Experimental results', '']
 for name in names:
  path=reports/name/'run_metadata.json'
  if not path.exists(): continue
  meta=json.loads(path.read_text(encoding='utf8')); d=meta['data']; sub=data[data.dataset==name]
  doc += [f'### {name}', '', f"Dataset: `{meta['dataset']}`. Fit / validation / test: {d['n_fit']} / {d['n_val']} / {d['n_test']}; {d['classes']} classes. Fit word count: mean {d['length_mean']:.1f}, 95th percentile {d['length_p95']:.1f}; vocabulary {d['vocabulary']}. Fit class counts range from {d['min_train_class']} to {d['max_train_class']}.", '', f'![Training-set EDA]({name}/eda.png)', '', '| Family | Accuracy | Macro P | Macro R | Macro F1 | Brier | ECE | Selected configuration |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
  doc.insert(len(doc)-4, f"Data quality: exact unique-text overlap fit/test before filtering = {d.get('fit_test_exact_overlap_before','not checked')}; removed fit records = {d.get('removed_fit_for_overlap','not checked')}, validation records = {d.get('removed_val_for_overlap','not checked')}. Missing/blank original train = {d.get('train_missing_or_blank','not checked')}, test = {d.get('test_missing_or_blank','not checked')}. Original source and test set are unchanged.")
  for r in sub.itertuples():
   brier=getattr(r,'brier_score',float('nan')); ece=getattr(r,'ece',float('nan'))
   doc.append(f'| {r.family} | {r.accuracy:.4f} | {r.precision_macro:.4f} | {r.recall_macro:.4f} | {r.f1_macro:.4f} | {brier:.4f} | {ece:.4f} | `{r.selection}` |')
  neg_path=reports/name/'negation.csv'
  if neg_path.exists():
   doc += ['', '| Family | Acc (negation) | n | Acc (other) | n | Gap |', '| --- | ---: | ---: | ---: | ---: | ---: |']
   for r in pd.read_csv(neg_path).itertuples():
    doc.append(f'| {r.family} | {r.acc_negation:.4f} | {int(r.n_negation)} | {r.acc_other:.4f} | {int(r.n_other)} | {r.gap:.4f} |')
  doc += ['', f"Validation experiments: [`{name}/validation.csv`]({name}/validation.csv). TextCNN training history, per-class scores, confusion matrices and error examples are stored in the same folder.", '']
  for family in ('tfidf','bge','textcnn'):
   matrix_path=reports/name/(family+'_confusion_matrix.npy'); label_names=meta.get('label_names',[])
   if matrix_path.exists() and label_names:
    matrix=np.load(matrix_path).copy(); np.fill_diagonal(matrix,0); a,b=np.unravel_index(np.argmax(matrix),matrix.shape)
    if matrix[a,b]: doc.append(f"- {family}: the most frequent observed confusion is `{label_names[a]}` predicted as `{label_names[b]}` ({matrix[a,b]} examples). This is descriptive and may not represent other samples.")
  ranked=sub.sort_values('f1_macro',ascending=False)
  if len(ranked)>1:
   lead,runner=ranked.iloc[0],ranked.iloc[1]
   doc += ['', f"In this held-out test, {lead['family']} scored {lead['f1_macro']:.4f} macro-F1 and {runner['family']} scored {runner['f1_macro']:.4f}; the observed gap is {lead['f1_macro']-runner['f1_macro']:.4f}. This is a descriptive single-run comparison, not evidence of a statistically significant difference or a basis for choosing hyperparameters."]
  doc += ['', 'Review the saved examples and per-class metrics before drawing task-specific conclusions.', '']
 doc += ['## Metrics and interpretation', '', 'Accuracy is the proportion of correct predictions. For each class, precision = TP / (TP + FP), recall = TP / (TP + FN), and F1 = 2 precision × recall / (precision + recall). Macro scores are the unweighted mean of per-class scores. Undefined class-level scores use zero. Macro-F1 is useful when class sizes differ because it gives each class equal weight.', '', 'When class probabilities are available, the multiclass Brier score is the mean squared error between the predicted probability vector and the one-hot label; ECE is the top-label expected calibration error over equal-width confidence bins. Negation analysis reports accuracy on examples that match simple English negation cues versus the remainder.', '', 'TF-IDF uses training-only document frequencies and weights a term by its within-document frequency and inverse document frequency. Linear classifiers learn decision boundaries on those sparse vectors. BGE-small supplies frozen pretrained contextual vectors; its logistic-regression classifier is trained on the fit split only. TextCNN learns its word vectors and convolution filters together using supervised cross-entropy and stops based on validation macro-F1.', '', '## Limitations and reproducibility', '', 'The results use a single train/validation split and random seed, without confidence intervals. They do not establish causal explanations or generalization to new domains. Model and dataset versions can change; preserve the execution environment and feature files. Compare test results descriptively and use validation for configuration choices.', '', '## Sources', '', '- https://huggingface.co/datasets/fancyzhx/ag_news', '- https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes', '- https://huggingface.co/datasets/mteb/banking77', '- https://huggingface.co/BAAI/bge-small-en-v1.5', '']
 (reports/'assignment_report.md').write_text('\n'.join(doc),encoding='utf8')
