import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy import sparse
from modules.assignment_pipeline import vectorizer, write_sparse_h5, read_sparse_h5, select_sparse
from modules.assignment_pipeline import load_splits
from unittest.mock import patch

class PipelineTests(unittest.TestCase):
 def test_sparse_saved_roundtrip(self):
  x={k:sparse.csr_matrix([[0.,1.],[2.,0.]],dtype=np.float32) for k in ('fit','val','test')}
  y={k:np.array([0,1]) for k in x}
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'features.h5'; write_sparse_h5(path,x,y); restored=read_sparse_h5(path)
   for k in x: np.testing.assert_array_equal(restored[k].toarray(),x[k].toarray())
 def test_vocab_fit_excludes_validation_and_test(self):
  v=vectorizer(False,1); v.fit(['alpha apple','alpha apple','beta banana','beta banana'])
  v.transform(['unseen validationsecret']); v.transform(['testsecret'])
  self.assertNotIn('validationsecret',v.vocabulary_); self.assertNotIn('testsecret',v.vocabulary_)
 def test_model_selection_ignores_test_labels(self):
  split={'fit':(['red good','red nice','blue bad','blue poor']*4,np.array([0,0,1,1]*4)), 'val':(['red good','blue bad'],np.array([0,1])), 'test':(['red nice','blue poor'],np.array([0,1]))}
  with tempfile.TemporaryDirectory() as d:
   _,a,_,_,_=select_sparse(split,Path(d))
   split['test']=(split['test'][0],np.array([1,0]))
   _,b,_,_,_=select_sparse(split,Path(d))
   for k in ('model','stop_words','max_ngram','C','f1_macro'): self.assertEqual(a[k],b[k])
 def test_test_overlap_removed_from_development_only(self):
  class Labels:
   names=['zero','one']
  class Split:
   features={'label':Labels()}
   column_names=['text','label']
   def __init__(self,texts,labels): self.texts=texts; self.labels=labels
   def __getitem__(self,key): return self.texts if key=='text' else self.labels
  train=Split([f'text {i}' for i in range(40)],[i%2 for i in range(40)])
  test=Split(['text 0','unseen sample'],[0,1])
  with patch('datasets.load_dataset',return_value={'train':train,'test':test}):
   split=load_splits('rotten_tomatoes')
  self.assertEqual(split['test'][0],['text 0','unseen sample'])
  self.assertNotIn('text 0',split['fit'][0]); self.assertNotIn('text 0',split['val'][0])
  self.assertEqual(split['quality']['removed_fit_for_overlap']+split['quality']['removed_val_for_overlap'],1)

if __name__=='__main__': unittest.main()
