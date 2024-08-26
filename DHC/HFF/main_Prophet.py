import sys
sys.path.append(r'./Algorithm/')
from Model_Runner import run_NPXGB, run_Prophet
from Time import time, output_duration
import pandas as pd


def main(run_name, precict_day = 7, citys_range = None,
        data_path = './Result/data.txt',
        predict_path = './Result/predict.txt',
        od_path = 'change your data path'):
    """
    对多变量的时间序列进行循环预测;

    参数:
        run_name (str): 函数名;
        precict_day (int): 预测天数, 默认为: 7;
        citys_range (int): 执行预测的序列数范围;

    return: 原始数据, 预测结果 (tuple[DataFrame, DataFrame])

    """
    od_df = pd.read_csv(od_path, sep = '\t', header = 0)
    #od_df = od_df.iloc[:,:-10]
    # 获取算法开始时间
    start_time = time.time()
    # 算法运行
    if run_name == 'NPXGB':
        data_df, precict_df = run_NPXGB(od_df, 
                                        citys_range = citys_range, 
                                        start_date = '2023-01-01', 
                                        precict_day = precict_day, 
                                        front_impact = 0,
                                        rear_impact = 0,
                                        accelerator = None,
                                        #accelerator = 'gpu',
                                )
    if run_name == 'Prophet':
        data_df, precict_df = run_Prophet(od_df, 
                                        citys_range = citys_range, 
                                        start_date = '2023-01-01', 
                                        precict_day = precict_day, 
                                        front_impact = 0,
                                        rear_impact = 0,
                                )
    # 获取算法结束时间
    end_time = time.time()
    
    # 计算并输出运行时间
    _, _, _ = output_duration(start_time, end_time, print_time = True)

    # 保存运行结果
    data_df.to_csv(data_path, sep = '\t', index=False)
    precict_df.to_csv(predict_path, sep = '\t', index=False)


if __name__ == "__main__":

    namelist = ['NPXGB', 'Prophet']
    precict_day = 8
    # 选择训练数据维度
    citys_range = None

    main(namelist[1], precict_day = precict_day, citys_range = citys_range,
         od_path='./OD Series.txt')