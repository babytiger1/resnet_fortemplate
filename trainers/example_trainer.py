from base.base_train import BaseTrain
#Sfrom tqdm import tqdm
import numpy as np


class ExampleTrainer(BaseTrain):
    def __init__(self, sess, model, data, config,logger):
        super(ExampleTrainer, self).__init__(sess, model, data, config,logger)

    def train_epoch(self):
        loop = range(self.config.num_iter_per_epoch)
        losses = []
        accs = []
        for _ in loop:
            loss, acc = self.train_step()
            print("the {} epoch, and train loss is {} , train acc is {}".format(_,loss,acc))
            losses.append(loss)
            accs.append(acc)
        loss = np.mean(losses)
        acc = np.mean(accs)
        tloss, tacc = self.eval_step()
        print("test loss is {} , test acc is {}".format(tloss, tacc))


        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss': loss,
            'acc': acc,
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        batch_x, batch_y = next(self.data.next_batch())
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
        _, loss, acc = self.sess.run([self.model.train_step, self.model.cross_entropy, self.model.accuracy],
                                     feed_dict=feed_dict)
        return loss, acc

    def eval_step(self):
        test_x, test_y = self.data.eval_data()
        feed_dict = {self.model.x: test_x, self.model.y: test_y, self.model.is_training: False}
        loss, acc = self.sess.run([ self.model.cross_entropy, self.model.accuracy],
                                     feed_dict=feed_dict)
        return loss, acc