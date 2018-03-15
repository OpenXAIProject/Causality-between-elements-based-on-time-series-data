import tensorflow as tf
import random
import pdb
import numpy as np
import time
from datetime import datetime


class MODEL(object):
    def __init__(self,config):

        self.sess   = config['sess']
        self.tasks = config['tasks']
        self.train_x= config['train_x']
        self.train_y= config['train_y']
        self.eval_x = config['eval_x']
        self.eval_y = config['eval_y']


        # MODEL CONFIGURATION
        self.num_features   = config['num_features']
        self.steps          = config['steps']
        self.hidden_units   = config['hidden_units']
        self.embed_size     = config['embed_size']
        self.num_layers     = config['num_layers']
        self.num_tasks      = config['num_tasks']
        
        # TRAINING CONFIGURATION
        self.lr             = config['lr']
        self.batch_size     = config['batch_size']
        
        # TESTING CONFIGURATION
        self.check_iter     = config['check_iter']
        self.save_iter      = config['save_iter']
        self.total_iter     = config['total_iter']
        self.save_dir       = config['save_dir']

        # Regularization coefficient
        self.asym_mu = config['asym_mu']
        self.asym_lambda = config['asym_lambda']
        self.ld_l2 = config['ld_l2']

        self.model_tag = config['model_tag']

        self.x = [tf.placeholder(shape=[None, self.steps, self.num_features], dtype=tf.float32, name='data'+str(i)) for i in range(self.num_tasks)]
        self.y = [tf.placeholder(shape=[None,1], dtype=tf.float32, name='labels'+str(i)) for i in range(self.num_tasks)]

        self.keep_prob = tf.placeholder('float')


    def attention_op(self, str_id, reversed_v_outputs, cell):

        with tf.variable_scope(str_id) as scope:
            if str_id=="alpha":
                att_W = self.att_W_alpha
                att_B = self.att_B_alpha
            else:
                att_W = self.att_W_beta 
                att_B = self.att_B_beta

            outputs, state = tf.nn.dynamic_rnn(cell,
                                            reversed_v_outputs,
                                            dtype=tf.float32)
            scope.reuse_variables()


        #Create attention
        att_v = [] 
        
        for i in range(self.steps):
            al = tf.matmul(outputs[:, i, :], att_W) + att_B
            att_v.append(al)

        if str_id == 'alpha':
            att_attention = tf.reshape(tf.nn.softmax(tf.concat(att_v, 1)), [-1, self.steps, 1])
        else:
            tanh_box = [tf.nn.tanh(att_v[tanh_idx]) for tanh_idx in range(len(att_v))]
            att_attention = tf.reshape(tf.concat(tanh_box, 1), [-1, self.steps, self.hidden_units])

        return att_attention
  
    def build_model(self,use_lstm=True):
        print("It's a good time to build a RETAIN model.")

        # Create regularization graph
        with tf.variable_scope('B'):
            self.b_o_t = [[] for i in range(self.num_tasks)]
            self.b_i_t = [[] for i in range(self.num_tasks)]
            for i in range(self.num_tasks):
                for j in range(self.num_tasks):
                    if j == i:
                        x = tf.constant(value=0.0)
                    else:
                        x = tf.Variable(0.0,str(i)+str(j))

                    self.b_i_t[j].append(x)
                    self.b_o_t[i].append(x)

            for i in range(self.num_tasks):
                self.b_o_t[i] = tf.stack(self.b_o_t[i])
                self.b_i_t[i] = tf.stack(self.b_i_t[i])

        self.losses = []
        self.preds = []
        self.total_auc = []
        self.total_acc = []


        #Make cells.
        def single_cell():
            return tf.contrib.rnn.GRUCell(self.hidden_units)
        if use_lstm:
            def single_cell():
                lstm_cell = tf.contrib.rnn.LSTMCell(self.hidden_units)
                return tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=self.keep_prob)
        
        with tf.variable_scope('shared_parameters' ) as scope:
        
            self.V = tf.get_variable('v_weight', shape=[self.num_features, self.hidden_units], dtype=tf.float32)
            self.att_W_alpha = tf.get_variable('att_weight_alpha', shape=[self.hidden_units, 1])
            self.att_B_alpha = tf.get_variable('att_bias_alpha', shape=[1])
            self.att_W_beta = tf.get_variable('att_weight_beta', shape=[self.hidden_units, self.hidden_units])
            self.att_B_beta = tf.get_variable('att_bias_beta', shape=[self.hidden_units])
        with tf.variable_scope('alpha',reuse=True):
            cell_alpha = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(self.num_layers)])
        with tf.variable_scope('beta',reuse=True):
            cell_beta = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(self.num_layers)])

        for task_id in range(self.num_tasks):
            v_emb = []
            for s in range(self.steps):
                embbed = tf.matmul(self.x[task_id][:, s, :], self.V)
                v_emb.append(embbed)

                embedded = tf.reshape(tf.concat(v_emb, 1), [-1, self.steps, self.hidden_units])

            #Reverse embedded_v
            reversed_v_outputs = tf.reverse(embedded, [1])

            # Get attention each tasks
            alpha_attention = self.attention_op('alpha', reversed_v_outputs, cell_alpha)
            alpha_attention = tf.reverse(alpha_attention, [1])
            beta_attention = self.attention_op('beta', reversed_v_outputs, cell_beta)
            beta_attention = tf.reverse(beta_attention, [1])

            # Attention_sum 
            c_i = tf.reduce_sum(alpha_attention * (beta_attention * embedded), 1)
            
            with tf.variable_scope('task' + str(task_id)):
                out_weight = tf.get_variable('weight', shape=[self.hidden_units,1])
                out_bias   = tf.get_variable('bias', shape=[1])

                logits = tf.matmul(c_i, out_weight) + out_bias
                preds = tf.nn.sigmoid(logits)

                # auc = tf.contrib.metrics.streaming_auc(predictions=preds,labels=self.y[task_id])
                auc = tf.metrics.auc(labels=self.y[task_id],predictions=preds)
                acc = 100*tf.contrib.metrics.accuracy(labels=tf.cast(self.y[task_id],tf.bool),predictions=(preds>=0.5))
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.y[task_id]) )

            self.losses.append(loss)
            self.preds.append(preds)
            self.total_auc.append(auc[1])
            self.total_acc.append(acc)

        # main_obj
        W_parameters = []
        w_t = []
        for task_id in range(self.num_tasks):
            w_i = tf.concat([tf.reshape(var,[-1]) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='task'+str(task_id)+'/')],0)
            w_t.append(w_i)
        W = tf.reshape(tf.concat(w_t,axis=0),[self.num_tasks,-1])

        obj_bt = []
        stack_loss = tf.stack(self.losses)
        self.main_loss = 0
        for t in range(self.num_tasks):
            p1 = (1.0 + self.asym_mu*tf.norm(self.b_o_t[t],ord=1))*self.losses[t]*1/np.sqrt((len(self.train_x[t])))
            temp = W * tf.reshape(self.b_i_t[t],[-1,1])
            L2 = tf.norm(w_t[t]-tf.reduce_sum(temp, axis=0),ord=2)
            p2 = self.asym_lambda * L2 * L2
            self.main_loss += p1 + p2

        self.main_optimize = tf.train.AdamOptimizer(self.lr).minimize(self.main_loss)
        print("You're done with graph building.")

    def get_batch(self):
        batch_x = []
        batch_y = []
        for i in range(self.num_tasks):
            index = np.random.choice(len(self.train_x[i]),self.batch_size,replace=False)
            batch_x.append(np.array([self.train_x[i][idx] for idx in index]))
            batch_y.append(np.array([self.train_y[i][idx] for idx in index]))
        return batch_x, batch_y


    def run_epoch(self, input_x, target_y, is_train=True):

        feed_dict = {self.x[i]:input_x[i] for i in range(self.num_tasks)}
        dict_y = {self.y[i]:target_y[i] for i in range(self.num_tasks)}
        feed_dict.update(dict_y)
        self.sess.run(tf.local_variables_initializer())
        if is_train:
            feed_dict.update({self.keep_prob:0.5})
            _, main_loss, losses, total_auc, total_acc = self.sess.run([self.main_optimize, self.main_loss, self.losses, self.total_auc, self.total_acc], feed_dict=feed_dict)
        else:
            feed_dict.update({self.keep_prob:1.0})
            main_loss, losses, total_auc, total_acc = self.sess.run([self.main_loss, self.losses, self.total_auc, self.total_acc], feed_dict=feed_dict)

        return main_loss, losses, total_auc, total_acc


    def run(self):
        self.sess.run(tf.global_variables_initializer())
        
        saver = tf.train.Saver()
        # saver.restore(self.sess, "_.ckpt")

        eval_loss_min = [float('inf')] * self.num_tasks
        eval_auc_min = [float('inf')] * self.num_tasks
        eval_Accuracy_min = [float('inf')] * self.num_tasks
        step_min = [0] * self.num_tasks

        sum_step_min = 0 
        eval_sum_loss_min = float('inf')
        eval_loss_at_sum_min = ['']*self.num_tasks
        eval_auc_at_sum_min = ['']*self.num_tasks
        eval_Accuracy_at_sum_min = ['']*self.num_tasks
        f = open('B_matrix_loss_sqrtn.txt','w')
        f.close()

        for i in range(self.total_iter):
            batch_x, batch_y = self.get_batch()
            train_sum_loss, train_loss, train_auc, train_Accuracy = self.run_epoch(batch_x, batch_y, is_train=True)

            if (i+1) % self.check_iter == 0:
                
                eval_sum_loss, eval_loss, eval_auc, eval_Accuracy = self.run_epoch(self.eval_x, self.eval_y, is_train=False)
  

                print("==============================================================================")
                print('Time : ' + str(datetime.now()) + '    ')
                print self.model_tag
                print 'Mu =',self.asym_mu, 'Lambda =',self.asym_lambda, 'Total loss:', train_sum_loss
                print("============================================================")
                print ("Step:%6d,      Train sum loss: %.3f" % (i+1, train_sum_loss))
                print ("Step:%6d,      Eval sum loss: %.3f" % (i+1, eval_sum_loss))
                if eval_sum_loss < eval_sum_loss_min:
                    sum_step_min = i+1
                    eval_sum_loss_min = eval_sum_loss
                    for task_id in range(self.num_tasks):
                        eval_loss_at_sum_min[task_id] = eval_loss[task_id]
                        eval_auc_at_sum_min[task_id] = eval_auc[task_id]
                        eval_Accuracy_at_sum_min[task_id] = eval_Accuracy[task_id]
                print("============================================================")
                for task_id in range(self.num_tasks):
                    print("-----------------------------------------------------------------------------")
                    if eval_loss[task_id] < eval_loss_min[task_id]:
                        eval_loss_min[task_id] = eval_loss[task_id]
                        eval_auc_min[task_id] = eval_auc[task_id]
                        eval_Accuracy_min[task_id] = eval_Accuracy[task_id]
                        step_min[task_id] = i+1

                        print("MIN_eval_loss " + self.tasks[task_id] + " is updated, lr: %f" %(self.lr))
                    else:
                        print("Task " + self.tasks[task_id] + ":")        

                    print("Step:%6d,      Train loss: %.3f, Train AUC: %.3f , Train Accuracy: %.3f" \
                            % (i+1, train_loss[task_id], train_auc[task_id], train_Accuracy[task_id]))

                    print("Step:%6d,       Eval loss: %.3f,  Eval Auc: %.3f ,  Eval Accuracy: %.3f" \
                            % (i+1, eval_loss[task_id], eval_auc[task_id], eval_Accuracy[task_id])) 
               
            if (i+1) % self.save_iter == 0:
                print("==============================================================================")
                print("======================           RESULT        ===============================")

                print ("AT Step:%6d,      Eval sum loss min: %.3f" % (sum_step_min, eval_sum_loss_min))
                for task_id in range(self.num_tasks):
                    print("--------------------------------------------------------------------------")
                    print("Task " + self.tasks[task_id] + ":")
                    print("Step:%7d,      Eval loss: %.3f, Eval AUC: %.3f, Eval Accuracy: %.3f" \
                            % (sum_step_min, eval_loss_at_sum_min[task_id], \
                                    eval_auc_at_sum_min[task_id], eval_Accuracy_at_sum_min[task_id]))

                for task_id in range(self.num_tasks):
                    print("==============================================================================")
                    print("Task " + self.tasks[task_id] + ":")
                    print("Step:%7d,      Eval loss: %.3f, Eval AUC: %.3f, Eval Accuracy: %.3f" \
                            % (step_min[task_id], eval_loss_min[task_id], \
                                    eval_auc_min[task_id], eval_Accuracy_min[task_id]))
                print("==============================================================================")


            # Stupid way to print out some information
            if (i+1) % 500 == 0:
                f = open('B_matrix_loss_sqrtn.txt','a')
                f.write('Step '+str(i+1)+'\n')
                for t in range(self.num_tasks):
                    d = ''
                    b_o_t = self.sess.run(self.b_o_t[t])
                    for x in b_o_t:
                        d = d+ str(x) + ','
                    d = d + '\n'
                    f.write(d)
                f.write('Eval Main Loss = '+str(eval_sum_loss)+'\n')
                for t in range(self.num_tasks):
                    f.write('Eval Loss '+str(eval_loss[t])+'= '+str(eval_sum_loss)+'\n')
                f.write('\n')
                f.close()
             