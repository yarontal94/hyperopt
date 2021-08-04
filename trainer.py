import os
import time
import dill

import hyperopt as hpo

import numpy as np
import pandas as pd

from sklearn import metrics

import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import matplotlib.cm as cm



class HyperoptTrainer:
    def __init__(self,
                 base_model,
                 train_args,
                 val_args,
                 *,
                 models_path='models',
                 prints=False,
                 seed=42,
                 score_fn=False,
                 model_int_feats=False,
                 **model_kwargs):
        self.base_model = base_model
        self.train_args = train_args
        self.val_args = val_args
        #self.model_name='base'
        self.models_path = models_path
        self.prints = prints
        self.seed = seed
        self.score_fn = score_fn
        self.model_int_feats = model_int_feats
        self.model_kwargs = model_kwargs
        
        self.metadata_columns = ['timestamp', 'time', 'train_score', 'val_score', 'hyperparams']
        self.trials = hpo.Trials()
        self.log = pd.DataFrame(columns=self.metadata_columns)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return dill.load(f)

    def make_model(self, hyperparams):
        for p in self.model_int_feats:
            if p in hyperparams:
                hyperparams[p] = int(hyperparams[p])

        self.model = self.base_model(**hyperparams, **self.model_kwargs)

    def objective(self, space):
        hyperparams = dict(sorted(list(space.items()), key=lambda x: x[0]))
        if self.prints:
            print(hyperparams)

        self.make_model(hyperparams)

        tic = time.time()
        train_score, val_score, best = self.train()
        toc = time.time()

        to_log = {
            'timestamp': time.strftime('%d-%m-%Y %H:%M:%S'),
            'time': (toc - tic)/60,
            'train_score': train_score,
            'val_score':val_score ,
            'hyperparams': hyperparams,
        }
        to_log.update(hyperparams)
        self.log = self.log.append(to_log, ignore_index=True)
        self.save(last=True, best=best)
        return -val_score

    def fmin(self, space, max_evals=500):
        best = hpo.fmin(fn=self.objective,
                        space=space,
                        algo=hpo.tpe.suggest,
                        max_evals=max_evals,
                        trials=self.trials,
                        rstate=np.random.seed(self.seed),
                        verbose=True)

    def get_path(self, best=False):
        if best:
            #return os.path.join(self.models_path, f'model_{self.model_name}_best.pkl')
            return os.path.join(self.models_path, 'model_x_best.pkl')
        else:
            #return os.path.join(self.models_path, f'model_{self.model_name}_last.pkl')
            return os.path.join(self.models_path, 'model_x_last.pkl')
        
    def save(self, last=False, best=False):
        if best:
            with open(self.get_path(best=True), 'wb') as f:
                dill.dump(self, f)
        if last:
            with open(self.get_path(best=False), 'wb') as f:
                dill.dump(self, f)

    def train(self):
        self.model.fit(self.train_args[0], self.train_args[1])

        preds_train = pd.Series(self.model.predict(self.train_args[0]), index=self.train_args[0].index)
        preds_val = pd.Series(self.model.predict(self.val_args[0]), index=self.val_args[0].index)
        
        train_score = self.score_fn(self.train_args[1], preds_train)
        val_score = self.score_fn(self.val_args[1], preds_val)

        best = (len(self.log) == 0 or val_score > self.log['val_score'].max())
        
        return train_score, val_score, best

    def plot(self, feat='val_score', top=True, q=0.01, size=5, markersize=0.5, sigma=8, bins=1000):
        for col in self.log.columns.difference(['timestamp', 'time', 'train_score', 'val_score', 'hyperparam_dict']):
            print('col',col)
#             plt.scatter(self.log[col], self.log[feat])
#             plt.title(col)
#             plt.show()

            fig, axs = plt.subplots(1, 2, figsize=(int(size*2), size))
            print(self.log.dtypes)
            x, y = self.log[col], self.log[feat]
            sign = -1 if top else 1
            quantile = sign*(sign*y).quantile(q=q)
            if top:
                x_best, y_best = self.log[y >= quantile][col], self.log[y >= quantile][feat]
            else:
                x_best, y_best = self.log[y <= quantile][col], self.log[y <= quantile][feat]
            # scatter
            axs[0].plot(x, y, 'k.', markersize=markersize)
            axs[0].plot(x_best, y_best, 'r.', markersize=markersize*10)
            axs[0].set_title("Scatter plot")
            print('x_type',type(x))
            print('y_type',type(y))
            # heatmap
            """
            heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
            heatmap = gaussian_filter(heatmap, sigma=sigma)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            axs[1].imshow(heatmap.T, extent=extent, origin='lower', cmap=cm.jet, aspect='auto')
            axs[1].set_title("Smoothing with  $\sigma$ = %d" % s)
            """
            fig.suptitle(col)
            plt.show()



