"""
    Define ModelRunner class to run the model.
"""
import pandas as pd 
import torch
import time
from datetime import datetime
import os
os.environ['PYTORCH_PYTHON_LOG_LEVEL'] = 'error'
import torch.nn as nn
from Models.optimize import Optimize
from Models.Time import time, get_interval
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ModelRunner():
    def __init__(self, paras, data, model):
        """
        paras: 模型参数
        data_gen: 数据
        model: 需要运行的模型
        """
        self.paras = paras
        self.data = data
        self.model = model
        self.best_rmse = None
        self.best_rse = None
        self.best_mae = None
        self.running_times = []
        self.train_losses = []
        self.predictlast=[]
        self.train_mae=[]
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    """
    Train the model
    """
    def train(self, epoch):
        self.model.train()
        total_loss = 0
        n_samples = 0
        Batch = 1
        
        for X, Y, NPPT, PCA in self.data.get_batches(self.data.train_set[0], # 训练数据 x
                                          self.data.train_set[1], # 训练数据 y
                                          self.data.train_set[2], # 训练数据 NPPT
                                          self.data.train_set[3], # 训练数据 PCA
                                          self.paras.batch_size, 
                                          True):
            batch_start = time.time()
            
            X, Y, NPPT = X.float(), Y.float(), NPPT.float()

            self.model.zero_grad()
            # 计算每个epoch的总batchs
            total_batchsize = self.data.train_set[0].shape[0]
            batch_size = self.paras.batch_size
            total_batchs = (total_batchsize//batch_size
                             if (total_batchsize%batch_size == 0) 
                             else (total_batchsize//batch_size + 1))
            print('-'*100)
            print('\nEpoch:  %s/%s'%(epoch, self.paras.epochs))
            print('Batch: %s/%s'%(Batch, total_batchs))
            Batch += 1
            print('该批次训练数据维度：%s, %s, %s'%(X.size()[0], X.size()[1], X.size()[2]))
            output= self.model(X, NPPT, PCA)

            torch.autograd.set_detect_anomaly(True)

            loss = self.loss(Y, output)
            model_loss = loss.clone()
            # 将损失记录下来
            self.train_mae.append(model_loss)
            
            # L1正则化
            l1_norm = sum(torch.norm(p, 1) for p in self.model.parameters())
            
            # L2正则化
            l2_norm = sum(( (torch.norm(p,2))**2).sum() for p in self.model.parameters())

            # 根据公式计算最终的组合损失
            #loss_combined = torch.exp(-log_var_a) * loss1 + log_var_a + torch.exp(-log_var_b) * loss2 + log_var_b
            loss_combined = ( model_loss + self.paras.l1_lambda * l1_norm + self.paras.l2_lambda * l2_norm).float()
            
            print('temp_loss :  %s\n' %(loss_combined.item()))
                      
            loss_combined.backward()  # 计算梯度

            self.optim.step()  # 梯度裁剪和参数更新由自定义优化器处理
            total_loss += loss_combined.item()
            n_samples += output.size(0) * output.size(1) * self.data.datadim

            batch_interval = time.time() - batch_start
            batch_time = f"{int(batch_interval // 3600)}小时{int((batch_interval % 3600) // 60)}分钟{int(batch_interval % 60)}秒"
            print('Batch Time Interval:  %s' %(batch_time))

            total_interval = time.time() - self.paras.start_time
            total_time = f"{int(total_interval // 3600)}小时{int((total_interval % 3600) // 60)}分钟{int(total_interval % 60)}秒"
            print('Run Time:  %s\n' %(total_time))
            
        return total_loss / n_samples

    # ---------------------------------------------------------------------------------------------------------------------------------------------
    """
    Valid the model while training
    """
    def evaluate(self, mode='train'):
        """
        Arguments:
            mode   - (string) 'train' or 'valid'
        """
        self.model.eval()
        total_loss = 0
        total_loss_l1 = 0
        n_samples = 0
        predict = None
        test = None

        if (mode == 'valid') & (len(self.data.valid_set[0])!=0):
            tmp_X = self.data.valid_set[0]
            tmp_Y = self.data.valid_set[1]
            tmp_NPPT = self.data.valid_set[2]
            tmp_PCA = self.data.valid_set[3]
        else:
            tmp_X = self.data.train_set[0]
            tmp_Y = self.data.train_set[1]
            tmp_NPPT = self.data.train_set[2]
            tmp_PCA = self.data.train_set[3]

        for X, Y, NPPT, PCA in self.data.get_batches(tmp_X, tmp_Y, tmp_NPPT, tmp_PCA, self.paras.batch_size, False):
            output= self.model(X, NPPT, PCA)
            L1_loss = self.evaluateL1(output, Y).item()
            L2_loss = self.evaluateL2(output, Y).item()
            if predict is None:
                predict = output
                test = Y
            else:
                predict = torch.cat((predict, output))
                test = torch.cat((test, Y))
            total_loss_l1 = total_loss_l1 + L1_loss
            total_loss = total_loss + L2_loss
            n_samples = n_samples + (output.size(0) * self.data.datadim)

        mse = total_loss / n_samples
        rse=0
        mae = total_loss_l1 / n_samples
        return mse, rse, mae
    # ---------------------------------------------------------------------------------------------------------------------------------------------
    def evaluate1(self):

        self.model.eval()
        X = self.data.test_set[0]
        NPPT = self.data.test_set[2]
        PCA = self.data.test_set[3]

        X=torch.tensor(X, dtype=torch.float32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X=X.to(device)
        predict= self.model(X, NPPT, PCA, )
        print('output:',predict.shape)
        return predict[0,:,:], self.data.test_set[1][0,:,:], self.data.test_time[1][0]

    def run(self):
        use_cuda = self.paras.gpu is not None
        if use_cuda:
            if type(self.paras.gpu) == list:
                self.model = nn.DataParallel(self.model, device_ids=self.paras.gpu)
            else:
                torch.cuda.set_device(self.paras.gpu)
        torch.manual_seed(self.paras.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed(self.paras.seed)
        if use_cuda: self.model.cuda()

        self.nParams = sum([p.nelement() for p in self.model.parameters()])

        if self.paras.L1Loss:
            self.loss = nn.L1Loss(reduction='sum')
        else:
            self.loss = nn.MSELoss(reduction='sum')
        self.evaluateL1 = nn.L1Loss(reduction='sum')
        self.evaluateL2 = nn.MSELoss(reduction='sum')
        if use_cuda:
            self.evaluateL1 = self.evaluateL1.cuda()
            self.evaluateL2 = self.evaluateL2.cuda()

        self.optim = Optimize(self.model.parameters(), 
                              self.paras.optim, 
                              self.paras.lr, 
                              self.paras.clip)

        best_train_mse = float("inf")
        best_train_rse = float("inf")
        best_train_mae = float("inf")

        tmp_losses = []
        try:
            for epoch in range(1, self.paras.epochs+1):
                epoch_start_time = time.time()
                train_loss = self.train(epoch)
                self.running_times.append(time.time() - epoch_start_time)
                tmp_losses.append(train_loss)
                tra_mse, tra_rse, tra_mae = self.evaluate(mode = 'valid')
                if tra_mse < best_train_mse:
                    best_train_mse = tra_mse
                    best_train_rse = tra_rse
                    best_train_mae = tra_mae

                self.optim.updateLearningRate(tra_mse, epoch)
        except KeyboardInterrupt:
            pass
        # 获得预测数据和真实数据
        self.predictlast, ground_truth, time_header=self.evaluate1()
        self.train_losses.append(tmp_losses)
        print('predict:',self.predictlast.shape)
        # 将数据转化为list
        predictlast = self.predictlast.cpu().detach().numpy().tolist()
        ground_truth = ground_truth.cpu().detach().numpy().tolist()
        # 将数据转化为 数据维度 × taskspan 的数据
        result_p = pd.DataFrame(predictlast, index = time_header).T
        result_t = pd.DataFrame(ground_truth, index = time_header).T
        # 数据保存
        result_p.to_csv('%sPredict.txt'%self.paras.output_path, sep = '\t', index = False)
        result_t.to_csv('%sGround Truth.txt'%self.paras.output_path, sep = '\t', index = False)
        final_best_mse = best_train_mse
        final_best_rse = best_train_rse
        final_best_mae = best_train_mae

        self.best_rmse = np.sqrt(final_best_mse)
        self.best_rse = final_best_rse
        self.best_mae = final_best_mae

    # ---------------------------------------------------------------------------------------------------------------------------------------------
    """
    其他函数
    """
    def getMetrics(self):
        """
        输出训练信息
        """
        print('-' * 100)
        print()
        print('* 参数总量: %d' % self.nParams)
        for k in self.paras.__dict__.keys():
            print(k, ': ', self.paras.__dict__[k])
        running_times = np.array(self.running_times)
        print("time: sum {:8.7f} | mean {:8.7f}".format(np.sum(running_times), np.mean(running_times)))
        print("rmse: {:8.7f}".format(self.best_rmse))
        print("rse: {:8.7f}".format(self.best_rse))
        print("mae: {:8.7f}".format(self.best_mae))
        print()

    def update_path(self):
        """
        更新输出文件路径
        """
        # 文件名从 Run_1 开始
        i = 1
        path = r'%sRun_%s/'%(self.paras.output_path, i)
        # 判断 Run_1 文件是否存在，不存在则新建
        if os.path.exists(path) == False:
            os.makedirs(path)
        else:
            while (os.path.exists(path)):
                # 获得文件中的文件数目
                files_num = os.listdir(path)
                # 文件夹为空，则不需要新建文件夹
                if len(files_num) == 0:
                    break
                i += 1
                path = r'%sRun_%s/'%(self.paras.output_path, i)
            # 如果路径没有文件夹，则新建文件夹
            if os.path.exists(path) == False:
                os.makedirs(path)              
        # 获取当前输出路径
        self.paras.output_path = path
    
    def output_params(self):
        with open('%sResult.txt'%self.paras.output_path, 'a', encoding = 'utf-8') as file:   #写入数据
            file.writelines('%s,%s'%(self.data.min_value, self.data.max_value))
            file.writelines('\n')
            file.writelines('-' * 100)
            file.writelines('\n')
            file.writelines('\n* 参数总量: %d\n' % self.nParams)
        for k in self.paras.__dict__.keys():
            if k in ['start_time', 'end_time']:
                if k == 'start_time':
                    with open('%sResult.txt'%self.paras.output_path, 'a', encoding = 'utf-8') as file:   #写入数据
                        file.writelines('%s: %s\n' % (k, datetime.fromtimestamp(self.paras.__dict__[k]).strftime('%Y-%m-%d %H:%M:%S')))
                if k == 'end_time':
                    end_time = time.time()
                    with open('%sResult.txt'%self.paras.output_path, 'a', encoding = 'utf-8') as file:   #写入数据
                        file.writelines('%s: %s\n'%(k,  datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')))
                        hours, minutes, seconds = get_interval(end_time-float(self.paras.__dict__['start_time']))
                        formatted_time = f"{hours}小时 {minutes}分钟 {seconds}秒"
                        file.writelines('%s: %s\n'%('time interval',  formatted_time))
            else:
                with open('%sResult.txt'%self.paras.output_path, 'a', encoding = 'utf-8') as file:   #写入数据
                    file.writelines('%s: %s\n'%(k,  self.paras.__dict__[k]))
        running_times = np.array(self.running_times)
        with open('%sResult.txt'%self.paras.output_path, 'a', encoding = 'utf-8') as file:   #写入数据
            file.writelines('\n')
            file.writelines('-' * 100)
            file.writelines('\n')
            file.writelines("time: sum {:8.7f} | mean {:8.7f}\n".format(np.sum(running_times), np.mean(running_times)))
            file.writelines("rmse: {:8.7f}\n".format(self.best_rmse))
            file.writelines("mae: {:8.7f}\n".format(self.best_mae))
