from DataProcessing.Install_Packages import install_package
packages_to_check = ["statsmodels", "numba", "pandas", "torchsummary"]
install_package(packages_to_check)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from DataProcessing.Data_Generator import GetData
from DataProcessing.Paras import Args, list_of_param_dicts
from DataProcessing.Install_Packages import install_package
from DataProcessing.Others import generate_range, get_data_source
from Models.SPDA import SPDANet
from Models.Time import time, output_duration
from Models.model_runner import ModelRunner
import torch
import gc
import os
# 减少内存碎片
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.backends.cudnn.enabled = True

param_dict = dict(                  
        # 对内存影响较大
        kernel_size = [7], # 动态卷积核大小
        batch_size =[8],
        hidRNN = [30], # 隐藏状态的维度   
        TimeWindow = [7], # 每个批量的时间维度
        h_w = [(7,1)],  # 为空时默认为: (TimeWindow/7 ,7)
        
        before_fc = [[128, 256, 128]], # 最初的全连接网络的隐藏维度
        fina_fc = [[128, 256, 128]], # 最后的全连接网络的隐藏维度

        dropout = [0.2],
        clip = [5.,],  # 梯度裁剪: 梯度的范数将被限制在 clip 以内
        L1Loss = [True,], # L1Loss or MSELoss      
        reduction_ratio = [5], # 通过该参数调节卷积参数量

        l1_lambda = [0, 0.1, 0.01, 0.001], # L1正则化参数
        l2_lambda = [0.001, 0.01, 0.1, 0], # L2正则化参数
        lr = [0.1, 0.01, 0.001, ],  
        highway_window = [0, 1], # 多步长自回归过程, 最大只能等于 max(TimeWindow, task_span) // task_span (整除倍数)
    
        seed = [54321], #设置随机种子
        gpu = [0],
        cuda = [True],
        optim = ['adam'],
        epochs = [200],
        task_span = [8],
        data_ratio = [0.1],
        start_time = [time.time()], # 记录起始时间    
        end_time = [''],
    )

if __name__ == '__main__':
    data_set = ['data_mini', 'total_data']
    Machine_list = ['QIAO-PC', 'Web-Sever']
    data_range_list = generate_range(50, 31)

    #获取算法开始时间
    start_time = time.time()

    for i in range(len(data_range_list)):
        #param_dict = get_data_source(data_set[0], Machine_list[0])
        param_dict = get_data_source(data_set[1], Machine_list[1], 
                                    data_range_list[i], param_dict)

        params = list_of_param_dicts(param_dict)
        for param in params:

            cur_paras = Args(param)
            cur_paras.start_time = time.time() # 每次循环重置时间
            data_gen = GetData(data_path = cur_paras.data_path,
                            NPPT_path = cur_paras.NPPT_path, 
                            PCA_path = cur_paras.PCA_path,
                            
                                data_ratio = cur_paras.data_ratio, 
                                TimeWindow = cur_paras.TimeWindow,
                                task_span = cur_paras.task_span,
                                cuda = cur_paras.cuda,
                                data_range = cur_paras.data_range
                            )
            runner = ModelRunner(cur_paras, data_gen, None)        
            runner.model = SPDANet(cur_paras, data_gen)
            runner.update_path()
            runner.run() 
            runner.getMetrics()
            runner.output_params()
            del runner
            gc.collect()

    # 获取算法结束时间
    end_time = time.time()
    # 计算并输出运行时间
    _, _, _ = output_duration(start_time, end_time, print_time = True)

