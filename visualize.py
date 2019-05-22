import visdom
import time
import numpy as np


class Visualizer(object):
    def __init__(self, env='main', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}
        self.env = env

    def plot_one(self, loss, name, step=1, xlabel='epoch', ylabel='loss'):
        x = self.index.get(name, 1)
        self.vis.line(Y=np.array([loss]), X=np.array([x]),
                      win=str(name),
                      opts=dict(title=name, xlabel=xlabel, ylabel=ylabel),
                      update=None if x == 1 else 'append'
                      )
        self.index[name] = x + step

    def plot_many_stack(self, d, xlabel='epoch',  ylabel='loss'):
        name = list(d.keys())
        name_total = " ".join(name)
        x = self.index.get(name_total, 1)
        val = list(d.values())
        if len(val) == 1:
            y = np.array(val)
        else:
            y = np.array(val).reshape(-1, len(val))
        self.vis.line(Y=y, X=np.ones(y.shape)*x,
                      win=str(name_total),
                      opts=dict(legend=name, title=name_total, xlabel=xlabel, ylabel=ylabel),
                      update=None if x == 1 else 'append'
                      )
        self.index[name_total] = x+1

    def log(self, info, win='log_text'):
        self.log_text += ('[{}] {} <br>'.format(time.strftime('%m/%d_%H:%M:%S'),info))
        self.vis.text(self.log_text, win)


if __name__ == '__main__':
    vis = Visualizer(env='test2')
    since = time.time()
    while True:
        for i in range(10):
            loss1 = np.random.rand()
            vis.plot_one(loss1, 'train loss', 5)
            time.sleep(1)
        loss2 = np.random.rand()
        vis.plot_many_stack({'train':loss1, 'val':loss2})
        time.sleep(0.2)
        # vis.log(time.time()-since)

        if time.time()-since > 30:
            print('over')
            break
