import visdom
import time
import numpy as np


class Visualizer(object):
    def __init__(self, env='main', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}
        self.env = env
        self.log_text = ''

    # def plot_train_val(self, step=5, loss_train=None, loss_val=None):
    #     '''
    #     plot val loss and train loss in one figure
    #     '''
    #     x = self.index.get('train_val', 0)
    #
    #     if x == 0:
    #         loss = loss_train if loss_train is not None else loss_val
    #         win_y = np.column_stack((loss, loss))
    #         win_x = np.column_stack((x, x))
    #         self.win = self.vis.line(Y=win_y, X=win_x, env=self.env)
    #         self.index['train_val'] = x + 1
    #         return
    #
    #     if loss_train is not None:
    #         self.vis.line(Y=np.array([loss_train]), X=np.array([x]),
    #                       win=self.win,
    #                       name='1',
    #                       update='append',
    #                       env=self.env)
    #         self.index['train_val'] = x + step
    #     else:
    #         self.vis.line(Y=np.array([loss_val]), X=np.array([x]),
    #                       win=self.win,
    #                       name='2',
    #                       update='append',
    #                       env=self.env)

    def plot_one(self, loss, name, step=1):
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([loss]), X=np.array([x]),
                      win=str(name),
                      opts=dict(title=name),
                      update=None if x == 0 else 'append'
                      )
        self.index[name] = x + step

    def plot_many_stack(self, d):
        name = list(d.keys())
        name_total = " ".join(name)
        x = self.index.get(name_total, 0)
        val = list(d.values())
        if len(val) == 1:
            y = np.array(val)
        else:
            y = np.array(val).reshape(-1, len(val))
        self.vis.line(Y=y, X=np.ones(y.shape)*x,
                      win=str(name_total),
                      opts=dict(legend=name, title=name_total),
                      update=None if x == 0 else 'append'
                      )
        self.index[name_total] = x+1

    def log(self, info, win='log_text'):
        self.log_text += ('[{}] {} <br>'.format(time.strftime('%m/%d_%H:%M:%S'),info))
        self.vis.text(self.log_text, win)


if __name__ == '__main__':
    vis = Visualizer(env='test')
    since = time.time()
    while True:
        for i in range(10):
            loss1 = np.random.rand()
            vis.plot_one(loss1, 'train', 5)
            time.sleep(1)
        loss2 = np.random.rand()
        vis.plot_many_stack({'train':loss1, 'val':loss2})
        time.sleep(0.2)
        vis.log(time.time()-since)

        if time.time()-since > 30:
            print('over')
            break
