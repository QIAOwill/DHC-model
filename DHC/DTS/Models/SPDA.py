import numpy as np
import pandas as pd
from mmcv.cnn import ConvModule
# pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.HAmodel import HA_model

# 判断是否有GPU信息
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SPDANet(nn.Module):
    def __init__(self, paras, data):
        """
        paras: (Class Parameter) 模型需要的超参数
        data: (Class GetData)数据
        """
        super(SPDANet, self).__init__() # 导入父函数的属性
        self.attri_path = paras.attribute_path
        self.data_range = paras.data_range
        self.use_cuda = paras.cuda

        self.before_fc = paras.before_fc
        self.fina_fc = paras.fina_fc

        if paras.h_w == ():
            self.h_w = (int(paras.TimeWindow/7),7)
        else:
            self.h_w = paras.h_w

        self.batch_size = data.train_set[0].size()[0]# 获取批量大小
        self.input_T = data.train_set[0].size()[1] # 获取每个批量数据的时间维度
        self.data_dim = data.train_set[0].size()[2] # 获取数据的变量维度
        self.hidR = paras.hidRNN # 隐藏状态的维度
        self.kernel_size = paras.kernel_size  # SDK的 K 值大小
        self.highway_window = paras.highway_window
        self.task_span = paras.task_span
        self.main_taskpoint = self.task_span//2
        self.dropout_layer = nn.Dropout(p = paras.dropout)
        self.convs = nn.ModuleList([])
        self.bns = nn.ModuleList([])

        self.stride = 1
        self.groups = 1
        reduction_ratio = paras.reduction_ratio # 通过该参数调节卷积参数量
        self.group_channels = self.data_dim // self.groups # 得到每组的维度数

        # 两个一维卷积层，生成权重参数矩阵
        self.conv1 = ConvModule(
            in_channels = self.data_dim,
            out_channels = self.data_dim // reduction_ratio, # # 通过reduction_ratio调节卷积参数量
            kernel_size = 1, # 卷积核大小为 1×1
            conv_cfg = None, # 填充、步长均为0
            norm_cfg = dict(type = 'BN'), # 采用批量归一化 (Batch Normalization)
            act_cfg = dict(type = 'ReLU')) # 激活函数为ReLU
        
        self.conv2 = ConvModule(
            in_channels = self.data_dim // reduction_ratio,
            out_channels = self.kernel_size**2 * self.groups,  #----------？？？？？？？？？--------------#
            kernel_size = 1, # 卷积核大小为 1×1
            stride = 1, # 步长1
            conv_cfg = None,
            norm_cfg = None,
            act_cfg = None) # 无激活函数，线性
        
        # 如果步长大于1，添加一个平均池化层
        if self.stride > 1:
            self.avgpool = nn.AvgPool2d(self.stride, self.stride)
        
        # 将原有数据进行展开，展开大小为 9 (3*3)
        self.unfold = nn.Unfold( 
                                kernel_size = self.kernel_size, 
                                dilation = 1, # 卷积核的间隔(即：非空洞卷积)
                                padding = (self.kernel_size-1)//2, 
                                stride = self.stride)
        
        self.shared_lstm = nn.LSTM(self.input_T, self.hidR)
        self.target_lstm = nn.LSTM(self.input_T, self.hidR)
        self.coARIMA = nn.ModuleList([])
        self.linears = nn.ModuleList([]) # 线性模型列表
        self.highways = nn.ModuleList([]) # 线性模型列表
        # 线性层用于将LSTM的输出转换为最终的输出。
        # self.fc将LSTM的隐藏状态（大小为 self.hidR）映射到数据维度（self.data_dim）
        for i in range(self.task_span,):
            self.linears.append(nn.Linear(self.hidR, self.data_dim))
            if (self.highway_window > 0):
                self.highways.append(nn.Linear(self.highway_window * (i+1), 1))
                # 将 self.highways[i] 返回的张量转换为 float 类型
                self.highways[i] = self.highways[i].float()
        
        # 全连接网络部分    
#-----------------------------------------------------------------------------------
        # 循环得到全连接层
        layers = [] # 线性层列表
        input_dim = self.data_dim + data.train_set[3].size()[2]
        for output_dim in self.before_fc:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(paras.dropout))
            input_dim = output_dim
        layers.append(nn.Linear(input_dim, self.data_dim))
        # 建立全连接层的 Sequential
        self.before_fc_net = nn.Sequential(*layers)
#-----------------------------------------------------------------------------------
        # 循环得到全连接层
        layers = [] # 线性层列表
        input_dim = self.data_dim * 2
        for output_dim in self.fina_fc:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(paras.dropout))
            input_dim = output_dim
        layers.append(nn.Linear(input_dim, self.data_dim))
        # 建立全连接层的 Sequential
        self.fina_fc_net = nn.Sequential(*layers)
        
    def SDK(self, x:torch.Tensor):
        """
        Spationtemporal Dynamic Kernel
        """
        # 得到SDK所需权重
        x = x.permute(0, 3, 1, 2)
        x = x.float() # x.shape = 2*200*5*6
        weight = self.conv2(self.conv1(x))
        batch, channel, h, w = weight.shape # 2*9*5*6
        # 将权重变成 (batch, 1, kernelsize*kernelsize, h, w)
        weight = weight.unsqueeze(1)
        # 将输入reshape
        out = self.unfold(x).view(batch, self.group_channels, self.kernel_size**2, h, w)
        # 求和，reshape回batch, h, w 形式
        out = (weight * out).sum(dim = 2).view(batch, h*w, self.data_dim)
        
        return out

    def forward(self, 
                X:torch.Tensor,
                NPPT:torch.Tensor,
                PCA:torch.Tensor,
                ) -> torch.Tensor:
        """
        x: 训练数据(tensor)

        return: 预测结果(tensor)
        """
        #X = X.to(device)
        #PCA = PCA.to(device)
        #mix_data = torch.cat([X, PCA], dim = 2)
        #mix_data = mix_data.reshape(-1, X.shape[2]+PCA.shape[2]).float()
        x = X#self.before_fc_net(mix_data).reshape(X.shape[0], X.shape[1], X.shape[2])

        # 读取属性数据
        attri_df = pd.read_csv(self.attri_path, sep = '\t', header = 0)
        attri_data = torch.tensor(attri_df.iloc[self.data_range[0]:self.data_range[1],:].values).to(device).to(torch.float32)
        #hidden = []
        SDK_output = []
        # 进行预测任务：task_span为需要预测的未来时间段长度
        for k in range(self.task_span):
            # 将时间维拆分为一个二维张量
            if k == 0:
                output_i = x.reshape(x.shape[0],
                                    self.h_w[0],self.h_w[1],
                                    x.shape[2])
            else:
                output_i = output_i.reshape(x.shape[0],
                                            self.h_w[0],self.h_w[1],
                                            x.shape[2])
            # 动态卷积
            output_i = self.SDK(output_i)

            # 激活函数，negative_slope 是负斜率参数，控制负值的斜率
            output_i = F.leaky_relu(output_i, negative_slope = 0.01)
            SDK_output.append(output_i)
            
            if k ==  self.main_taskpoint:
                # 获得的输出是 (批次 * 隐藏层数) 大小的张量
                HA_Model = HA_model(output_i, attri_data, self.input_T, self.hidR, self.hidR)
                hidden = HA_Model.train_forward()

            output_i = self.dropout_layer(output_i)

        shared_lstm_results = [] # 
        target_R = None # main_taskpoint的
        target_h = None # 
        target_c = None # 
        self.shared_lstm.flatten_parameters()
        for k in range(self.task_span):
            # 时间维度的LSTM
            SDK_i = SDK_output[k].permute(2,0,1).contiguous()
            # 对时间维度进行lstm
            _, (hidden_state, cell_state) = self.shared_lstm(SDK_i)
            # 找到main_taskpoint对应的时刻
            if k ==  self.main_taskpoint:
                # 获取main_taskpoint时刻下的SDK卷积结果
                target_R = SDK_i
                # 将 HA 中的LSTM结果与main_taskpoint时刻的LSTM结果相加
                target_h = hidden_state + hidden
                #target_h = hidden_state
                # 获取main_taskpoint时刻的细胞状态
                target_c = cell_state
            # 储存本时刻的隐藏状态
            hidden_state = self.dropout_layer(torch.squeeze(hidden_state, 0))
            shared_lstm_results.append(hidden_state)

        self.target_lstm.flatten_parameters()
        target_R = target_R.float()
        target_h = target_h.float()
        target_c = target_c.float()
        # 计算main_taskpoint时刻的隐藏状态
        _, (target_result, _) = self.target_lstm(target_R, (target_h, target_c))
        target_result = self.dropout_layer(torch.squeeze(target_result, 0))

#----------------------------------线性部分----------------------------------------
        # 获得具有协整性的索引和数据
        result = None
        for i in range(self.task_span):
            if i == self.main_taskpoint:
                linear_result = self.linears[i](target_result)
            else:
                linear_result = self.linears[i](shared_lstm_results[i])
            # 为 (批量×数据维度) 添加时间维度
            linear_result = torch.unsqueeze(linear_result, 1)
            if result is not None:
                linear_result = linear_result.to(result.device)
                # 将linear_result和res在时间维度拼接起来
                result = torch.cat((result, linear_result), 1)
            else:
                result = linear_result

        #highway
        if (self.highway_window > 0):
            highway = None
            for i in range(self.task_span):
                # 多余维度用0补全
                if i+1 > self.input_T:
                    shape = list(result.shape)
                    shape[1] = 1
                    zero_tensor = torch.zeros(shape).to(device)
                    highway = torch.cat((highway, zero_tensor), 1)
                    continue
                # 取最后self.highway_window * (i+1)个时间步的数据
                z = x[:, -(self.highway_window * (i+1)):, :]
                z = z.permute(0,2,1).contiguous().view(-1, self.highway_window * (i+1))
                z = self.highways[i](z.float()) # 全连接网络，变为 (批次*数据维度)
                z = z.view(-1, self.data_dim)
                z = torch.unsqueeze(z, 1)
                if highway == None:
                    # 在时间维度拼接起来
                    highway = z
                else:
                    highway = torch.cat((highway, z), 1)

            result = result + highway

#----------------------------------融合NeuralProphet部分----------------------------------------
        NPPT = NPPT.to(result.device)
        #mix_result = result * NPPT
        
        #final_result = self.fina_fc_net(mix_result.reshape(-1, self.data_dim).float())
        #final_result = final_result.reshape(result.shape)
        mix_result = torch.cat([result, NPPT], dim = 2)
        mix_result = mix_result.reshape(-1, self.data_dim * 2).float()
        
        final_result = self.fina_fc_net(mix_result)
        
        return final_result.reshape(result.shape)



        