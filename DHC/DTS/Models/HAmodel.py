import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F

class SA(nn.Module):
    def __init__(self, time_dim, data_dim, SA_num_hidden, attri_data:torch.tensor):
        """
        SA_num_hidden: 循环神经网络的隐藏层数
        data_dim: 变量维度
        time_dim : 获取每个批量数据的时间维度
        attri_data: 属性数据
        """
        super(SA, self).__init__()
        self.SA_num_hidden = SA_num_hidden # 循环神经网络的隐藏层数
        self.data_dim = data_dim # 变量维度
        self.time_dim = time_dim  # 获取每个批量数据的时间维度
        self.attri_data = attri_data # 属性数据

        # 时间注意机制:SA是LSTM
        self.SA_lstm = nn.LSTM(
            input_size = self.data_dim,
            hidden_size = self.SA_num_hidden,
            num_layers = 1
        )

        # 通过确定性注意模型构建输入注意机制
        self.SA_attn = nn.Sequential(
            nn.Linear(2*self.SA_num_hidden + self.time_dim, SA_num_hidden),
            nn.Tanh(),
            nn.Linear(SA_num_hidden, 1)
        )

        # 对属性数据进行处理
        self.Attribute = nn.Sequential(
            nn.Linear(self.attri_data.shape[1], self.attri_data.shape[1]//2),
            nn.LeakyReLU(0.5),
            nn.Linear(self.attri_data.shape[1]//2, 1)
        )

    def forward(self, X:torch.tensor):
        """
        X: 输入的训练数据
        """
        # 通过属性数据归一化，对变量采用注意力机制进行加权
        alpha_attri = F.softmax(self.Attribute(self.attri_data), dim = 0)
        X_attri = torch.mul(alpha_attri.view(1, 1, X.shape[2]), X)
        
        # X.data.new它能够创建一个与 X 具有相同类型的新张量，但是具有不同形状的特性。
        # 创建大小为：(批量大小 * 时间维度 * 变量维度)的全零张量
        X_tilde = Variable(X.data.new(X.size(0), self.time_dim , self.data_dim).zero_())
        # 初始化：创建大小为：(1 * 批量大小 * 隐藏层大小)的全零张量
        hidden_t = Variable(X.data.new(1, X.size(0), self.SA_num_hidden).zero_()) # t时刻的隐藏层向量
        cell_t = Variable(X.data.new(1, X.size(0), self.SA_num_hidden).zero_()) # t时刻的细胞向量
        
        for t in range(self.time_dim):
            # repeat三个参数代表三个维度重复的次数：(hidden_t.size(0) * self.data_dim, hidden_t.size(1)*1, hidden_t.size(2)*1)
            # permute(1, 0, 2)：交换维度一和维度二
            # 最终变为：批量大小 * 变量维度 * 隐藏层大小
            x = torch.cat((hidden_t.repeat(self.data_dim, 1, 1).permute(1, 0, 2), 
                            cell_t.repeat(self.data_dim, 1, 1).permute(1, 0, 2),
                            # 变为：批量大小 * 变量维度 * 时间维度
                            X_attri.permute(0,2,1)), dim=2)

            # 把批量大小 * 变量维度拉平，时间维坍缩为 1，得到每个批量下变量的权重
            x = x.to(next(self.parameters()).dtype)
            x = self.SA_attn(x.view(-1, self.SA_num_hidden * 2 + self.time_dim ))
            # 通过softmax对批次下的每个变量的权重进行归一化，计算加权系数 alpha
            alpha_s = F.softmax(x.view(-1, self.data_dim), dim=1)
            # 分别对每个时间下的(批量大小*变量维度)进行逐元素加权计算
            X_tilde[:, t, :] = torch.mul(alpha_s, X_attri[:, t, :])

            # 更新 隐藏状态 和 细胞状态   
            if t != self.time_dim - 1:
                # flatten_parameters会将 RNN 层的权重矩阵展平成连续的一维数组
                # 可以更高效地传输到 GPU，并且在计算的时候也更加高效
                self.SA_lstm.flatten_parameters()
                # SA LSTM：返回输出序列 and 最终的隐藏状态和细胞状态。
                _, (hidden_t, cell_t) = self.SA_lstm(X_attri[:, t, :].unsqueeze(0), (hidden_t, cell_t))
            
        return X_tilde

class TA(nn.Module):
    def __init__(self, time_dim, TA_num_hidden, SA_num_hidden, data_dim):
        """
        time_dim : 获取每个批量数据的时间维度
        TA_num_hidden: TA中循环神经网络的隐藏层数
        SA_num_hidden: SA中循环神经网络的隐藏层数
        data_dim: 变量维度
        """
        super(TA, self).__init__()
        self.TA_num_hidden = TA_num_hidden # TA部分的隐藏层大小
        self.SA_num_hidden = SA_num_hidden # SA部分的隐藏层大小
        self.time_dim = time_dim  # 获取每个批量数据的时间维度
        self.data_dim = data_dim # 获得数据的变量个数

        # 注意力层
        self.attn_layer = nn.Sequential(
            nn.Linear(TA_num_hidden + self.data_dim, SA_num_hidden),
            nn.Tanh(),
            nn.Linear(SA_num_hidden, 1))
        
        # 循环神经网络
        self.TA_lstm = nn.LSTM(
            input_size = self.data_dim,
            hidden_size = self.TA_num_hidden,
            num_layers = 1)

        #self.fc = nn.Linear(SA_num_hidden + 1, 1)

        #self.fc_final = nn.Linear(TA_num_hidden + SA_num_hidden, 1)

        #self.fc.weight.data.normal_()

    def get_alpha(self, hidden_set):
        """
        hidden_set: 时间*批量*隐藏层
        """
        ht_prior = hidden_set[:-1]  # 取时间维度除了最后一个时间步之外的所有数据
        ht_finally = hidden_set[-1].view(1, -1)  # 取最后一个时间步的数据，并转换为形状为 [1, -1]

        # 创建一个新的相似度张量，用于保存每个时间步的余弦相似度
        similarity = Variable(ht_prior.data.new(len(ht_prior), 1).zero_())

        # 计算每个时间步的余弦相似度
        for i in range(ht_prior.shape[0]):
            similarity[i] = F.cosine_similarity(ht_prior[i].view(1, -1), ht_finally, dim=1)  # 计算余弦相似度并保存到相似度张量中

        # 使用softmax进行归一化，dim=0 表示在每个时间步上进行归一化
        alpha_t = F.softmax(similarity, dim=0)

        return alpha_t

    def forward(self, X_tilde):
        """
        X: 输入的训练数据
        X_tilde: TA的加权数据
        """
        # 初始化：创建大小为：(1 * 批量大小 * 隐藏层大小)的全零张量
        hidden_t = Variable(X_tilde.data.new(1, X_tilde.size(0), self.TA_num_hidden).zero_()) # t时刻的隐藏层向量
        cell_t = Variable(X_tilde.data.new(1, X_tilde.size(0), self.TA_num_hidden).zero_()) # t时刻的细胞向量
        hidden_set = []

        for t in range(self.time_dim):

            hidden_set.append(hidden_t)
            
            # 更新 隐藏状态 和 细胞状态
            if t != self.time_dim - 1:
                # TA LSTM
                self.TA_lstm.flatten_parameters()
                # 返回输出序列 and 最终的隐藏状态和细胞状态。
                _, (hidden_t, cell_t) = self.TA_lstm(X_tilde[:, t, :].unsqueeze(0), (hidden_t, cell_t))

        # 将tensor的list转化为tensor
        hidden_tensor = torch.cat(hidden_set, dim=0)
        alpha_t = self.get_alpha(hidden_tensor)
        alpha_t = alpha_t.view(len(hidden_tensor)-1, 1, 1)
        # ht_final： (1 * 批量大小 * 隐藏层大小)
        ht_final = torch.sum(alpha_t * hidden_tensor[:-1], dim=0).unsqueeze(0)

        return ht_final

class HA_model(nn.Module):
    def __init__(self, X, attri_data, time_dim, SA_num_hidden, TA_num_hidden):
        """initialization."""
        super(HA_model, self).__init__()
        self.SA_num_hidden = SA_num_hidden
        self.TA_num_hidden = TA_num_hidden
        self.X = X
        self.learning_rate=0.001
        self.data_dim = X.shape[2]
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.SA = SA(data_dim = self.data_dim,
                     SA_num_hidden = SA_num_hidden,
                     time_dim = time_dim,
                     attri_data = attri_data).to(self.device)
        self.TA = TA(data_dim = self.data_dim,
                     SA_num_hidden = SA_num_hidden,
                     TA_num_hidden = TA_num_hidden,
                     time_dim = time_dim).to(self.device)
        self.SA_optimizer = optim.Adam(params = filter(lambda p: p.requires_grad,
                                                          self.SA.parameters()),
                                            lr = self.learning_rate)
        self.TA_optimizer = optim.Adam(params = filter(lambda p: p.requires_grad,
                                                          self.TA.parameters()),
                                            lr = self.learning_rate)

    def train_forward(self):
        """Forward pass."""
        X_tilde = self.SA(self.X)
        hidden = self.TA(X_tilde)
        self.SA_optimizer.step()
        self.TA_optimizer.step()
        return hidden

