"""Run the assignment pipeline; full and quick outputs are isolated."""
import argparse
from pathlib import Path
from modules.assignment_pipeline import REGISTRY, run_dataset

def main():
 p=argparse.ArgumentParser(description='Text classification assignment')
 p.add_argument('--dataset',choices=['all',*REGISTRY],default='all')
 p.add_argument('--quick',action='store_true',help='small subsets for smoke testing only')
 p.add_argument('--padding',choices=['dynamic','fixed'],default='dynamic')
 p.add_argument('--tokenization',choices=['regex','whitespace'],default='regex',help='TF-IDF tokenization; Transformer uses its pretrained tokenizer')
 p.add_argument('--classifiers',nargs='+',choices=['nb','logreg','svm'],default=['nb','logreg','svm'])
 p.add_argument('--embeddings',nargs='+',choices=['tfidf','bge'],default=['tfidf','bge'],help='Use both for assignment submission')
 p.add_argument('--batch-size',type=int,default=32)
 p.add_argument('--epochs',type=int,default=7)
 p.add_argument('--output',type=Path,default=None)
 a=p.parse_args()
 if a.batch_size<1 or a.epochs<1: p.error('batch size and epochs must be positive')
 root=a.output or Path(__file__).resolve().parent/'outputs'/('quick' if a.quick else 'full')
 names=list(REGISTRY) if a.dataset=='all' else [a.dataset]
 for n in names: run_dataset(n,root,a.quick,a.epochs,a.batch_size,a.padding,a.tokenization,a.classifiers,a.embeddings)
 print(f'Finished; results and feature files: {root.resolve()}')

if __name__=='__main__': main()
