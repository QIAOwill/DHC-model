import pandas as pd
import numpy as np
import torch

class DataGenerator():
    def __init__(self, 
                 time_data:pd.DataFrame, 
                 NPPT_data:pd.DataFrame,  
                 pca_data:pd.DataFrame,
                 data_ratio:float, 
                 TimeWindow:int, 
                 cuda:bool,
                 task_span,
                 data_range):
        """
        time_data: 以时间为行, OD对为列的时间序列数据
        data_ratio: 训练数据的占比
        TimeWindow: X中的时间窗长度(天)
        """
        self.time_data = time_data.iloc[:, 1:].iloc[:,data_range[0]:data_range[1]]
        self.NPPT_data = NPPT_data.iloc[:,data_range[0]:data_range[1]]
        self.pca_data = pca_data

        self.date = time_data['Time'].values.tolist()
        self.data_ratio = data_ratio
        self.TimeWindow = TimeWindow
        self.task_span = task_span
        self.cuda = cuda
        self.train_time = [[], []] # 获取训练数据的时间刻度
        self.valid_time = [[], []] # 获取验证数据的时间刻度
        self.test_time = [[], []] # 获取测试数据的时间刻度
        self.get_train_data() # 添加训练和测试数据集
        self.input_T = self.train_set[0].size()[1] # 获取每个批量数据的时间维度
        self.datadim = self.train_set[0].size()[2] # 获取数据的变量维度

        self.max_value = self.time_data.max().max()
        self.min_value = self.time_data.min().min()

        print('训练数据维度为: %s, %s, %s'%(self.train_set[0].size()[0], self.input_T, self.datadim))
        print('验证数据维度为: %s, %s, %s'%(self.valid_set[0].size()[0], self.input_T, self.datadim))

    def get_train_data(self):
        """
        根据时间窗对原时间序列数据转换为训练数据, 根据数据比例, 建立训练集和测试集
        """
        # 训练数据的转换
        X_list = [] # 训练数据
        y_list = [] # 测试数据
        PCA_list = [] # 属性数据的PCA结果
        NPPT_list = [] # NeuralProphet的训练归一化数据 
        time_index = [[],[]]
        
        time = 0
        # time: 代表当前时间;       self.TimeWindow: 代表每次截取的时间窗长度;      self.task_span: 代表需要预测的天数;
        while time + self.TimeWindow + self.task_span <= len(self.time_data):
            # 截取当前时刻的时间窗作为训练数据
            X_list.append(self.time_data.iloc[time:time+self.TimeWindow, :].values.tolist())
            PCA_list.append(self.pca_data.iloc[time:time+self.TimeWindow, :].values.tolist())
            # 获得该时间窗的日期
            time_index[0].append([date for date in self.date[time : time+self.TimeWindow]])

            # 截取紧跟时间窗后的 self.task_span 天作为模型预测的 groundtruth 数据
            y_list.append(self.time_data.iloc[(time + self.TimeWindow):
                                              (time + self.TimeWindow + self.task_span),:].values.tolist())
            # 截取 self.NPPT_data 中紧跟时间窗后的 self.task_span 天的数据
            NPPT_list.append(self.NPPT_data.iloc[(time + self.TimeWindow):
                                              (time + self.TimeWindow + self.task_span),:].values.tolist())
            # 截取 groundtruth 数据的对应日期
            time_index[1].append([date for date in self.date[(time + self.TimeWindow):
                                              (time + self.TimeWindow + self.task_span)]])
            time += 1
            #time = time + self.TimeWindow + self.task_span

        # 获得训练数据和验证数据总大小 (总数据-1)
        train_valid = len(X_list) - 1
        # 按比例将训练数据和验证数据进行分割
        train_size = int(train_valid * self.data_ratio)

        # ------------------------------------------------------------截取训练集-------------------------------------------------------
        train_X = torch.from_numpy(np.array(X_list[ : train_size]))
        train_pca = torch.from_numpy(np.array(PCA_list[ : train_size]))
        self.train_time[0].append(time_index[0][ : train_size])
        train_y = torch.from_numpy(np.array(y_list[ : train_size]))
        # 与 train_y 对应的NNPT数据
        train_NPPT = torch.from_numpy(np.array(NPPT_list[ : train_size]))
        self.train_time[1].append(time_index[1][ : train_size])
        
        # ------------------------------------------------------------截取验证集-------------------------------------------------------
        valid_X = torch.from_numpy(np.array(X_list[train_size : -1]))
        valid_pca = torch.from_numpy(np.array(PCA_list[train_size : -1]))
        self.valid_time[0].append(time_index[0][train_size : -1])
        valid_y = torch.from_numpy(np.array(y_list[train_size : -1]))
        # 与 valid_y 对应的NNPT数据
        valid_NPPT = torch.from_numpy(np.array(NPPT_list[train_size : -1]))
        self.valid_time[1].append(time_index[1][train_size : -1])

        # ------------------------------------------------------------截取测试集 (选取最后一组数据作为测试集)-------------------------------------------------------
        test_X = torch.from_numpy(np.array(X_list[-1 : ]))
        test_pca = torch.from_numpy(np.array(PCA_list[-1 : ]))
        self.test_time[0].append(time_index[0][-1 : ])
        test_y = torch.from_numpy(np.array(y_list[-1 : ]))
        # 与 test_y 对应的NNPT数据
        test_NPPT = torch.from_numpy(np.array(NPPT_list[-1 : ]))
        self.test_time[1].append(time_index[1][-1 : ])
        
        # ------------------------------------------------------------整合训练集和测试集-------------------------------------------------------
        self.train_set = [train_X, train_y, train_NPPT, train_pca]
        self.valid_set = [valid_X, valid_y, valid_NPPT, valid_pca]
        self.test_set = [test_X, test_y, test_NPPT, test_pca]
    
    def get_batches(self, 
                    train_X:torch.Tensor, 
                    train_y:torch.Tensor, 
                    train_NPPT:torch.Tensor,
                    train_pca:torch.Tensor,  
                    batch_size:int, 
                    shuffle = False):
        """
        返回批量数据

        train_X: 数据集的输入(input)
        train_y: 数据集的标签(ground truth)
        train_NPPT: NeuralProphet的训练归一化数据
        batch_size: 批量大小
        shuffle: 是否将数据集打乱顺序
        """
        length = len(train_X)
        if shuffle:
            index = torch.randperm(length) # 获取随机排序的index
        else:
            index = torch.LongTensor(range(length)) # index按顺序排序

        start_index = 0
        while (start_index < length):
            end_index = min(length, start_index + batch_size)
            part_index = index[start_index:end_index] # 获得本次batch的索引
            batch_X = train_X[part_index] # 按索引获得数据
            batch_y = train_y[part_index] # 按索引获得数据
            batch_NPPT = train_NPPT[part_index] # 按索引获得数据
            batch_pca = train_pca[part_index]

            # 判断是否移动到GPU上计算
            if (self.cuda): 
                batch_X = batch_X.cuda()
                batch_y = batch_y.cuda()
                batch_NPPT = batch_NPPT.cuda()
                batch_pca = batch_pca.cuda()
                
            yield batch_X, batch_y, batch_NPPT, batch_pca # 每次循环到yield函数会停一下
            start_index += batch_size

class GetData(DataGenerator):
    def __init__(self, data_path:str, NPPT_path:str, PCA_path:str, 
                 data_ratio, TimeWindow, task_span, cuda, data_range):
        """
        data_path: 时间序列数据
        TimeWindow: 每个训练数据的时间长度(天)
        """
        # 数据读取
        time_data = pd.read_csv(data_path, sep = '\t', header = 0)
        NPPT_data = pd.read_csv(NPPT_path, sep = '\t', header = 0)
        pca_data = pd.read_csv(PCA_path, sep = '\t', header = 0)
        # 清空列名
        NPPT_data.columns = pd.RangeIndex(len(NPPT_data.columns))

        
        # 调用DataGenerator类生成训练数据
        super(GetData, self).__init__(time_data = time_data,
                                      NPPT_data = NPPT_data, 
                                      pca_data = pca_data, 
                                      data_ratio = data_ratio, 
                                      TimeWindow = TimeWindow,
                                      task_span = task_span,
                                      cuda = cuda,
                                      data_range = data_range,
                                      )