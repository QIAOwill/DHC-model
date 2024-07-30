import pandas as pd
from NeuralProphet import NP_forecast
from Prophet_model import Prophet_forecast
from XGB_Regression import XGB_boost
from Stretch import stretch_data
from IPython.display import clear_output

def run_NPXGB(od_df:pd.DataFrame, 
        citys_range = None, 
        start_date = '2023-01-01', 
        precict_day = 7, 
        front_impact = 0,
        rear_impact = 0,
        accelerator = None):
    
    """
    对多变量的时间序列进行循环预测;

    参数: 
        od_df (DataFrame): OD序列数据;
        citys_range (int) : 执行预测的序列数范围;
        start_date (str) 起始时间, 默认为: '2023-01-01'; 
        precict_day (int) 预测天数, 默认为: 7;
        front_impact (int) 节假日前影响期, 默认为: 0; 
        rear_impact (int) 节假日后影响期, 默认为: 0;
        accelerator (str) : 加速器, 默认为None;

    return: 原始数据, 预测结果 (tuple[DataFrame, DataFrame])
    
    """
    # 数据处理
    data_df = od_df.T.reset_index(drop=True) # 以城市对为列, 时间为行;
    # 设置列名
    data_df.columns = data_df.iloc[0] + ',' + data_df.iloc[1]
    data_df = data_df.drop([0, 1]).reset_index(drop=True)
    # 设置需要预测的城市对数量
    if citys_range == None:
        citys_range = [0, len(od_df)]
    # 筛选出需要的原始数据
    data_df = data_df.iloc[:,citys_range[0]:citys_range[1]]

    # 初始化预测结果
    precict_df = pd.DataFrame()

    # 对每个城市对进行预测
    citys = 0
    num = 1 # 预测进度
    while citys < (citys_range[1]-citys_range[0]):
        clear_output(wait=True)  # 清除之前的输出
        print('\n\n', '-'*100, '\n')
        print(' '*40, f"预测进度: {num}/{citys_range[1]-citys_range[0]}")
        print('\n', '-'*100, '\n\n')
        y = data_df.iloc[:, citys:citys+1]
        # 数据拉伸
        y = stretch_data(
                            origin_df = y,
                            start = 119,
                            day = 5,
                            insert_num = 4,
                    )
        # 调用模型进行预测
        forecast = NP_forecast(
                                y.iloc[:-precict_day],
                                start_date = start_date, 
                                precict_day = precict_day,
                                front_impact = front_impact, 
                                rear_impact = rear_impact,
                                accelerator = accelerator,
                            )
        # 将各类特征采用 XGB 方法进行拟合
        predict_i, _ = XGB_boost(forecast, precict_day)
        
        # 积累当前计算结果
        precict_df = pd.concat([precict_df, pd.DataFrame(predict_i),], axis = 1)
        citys += 1
        num += 1

    # 设置城市对的列名及时间
    precict_df.columns = data_df.columns

    return data_df, precict_df

def run_Prophet(od_df:pd.DataFrame, 
        citys_range = None, 
        start_date = '2023-01-01', 
        precict_day = 7, 
        front_impact = 0,
        rear_impact = 0,):
    
    """
    对多变量的时间序列进行循环预测;

    参数: 
        od_df (DataFrame): OD序列数据;
        citys_range (int) : 执行预测的序列数范围;
        start_date (str) 起始时间, 默认为: '2023-01-01'; 
        precict_day (int) 预测天数, 默认为: 7;
        front_impact (int) 节假日前影响期, 默认为: 0; 
        rear_impact (int) 节假日后影响期, 默认为: 0;

    return: 原始数据, 预测结果 (tuple[DataFrame, DataFrame])
    
    """
    # 数据处理
    data_df = od_df.T.reset_index(drop=True) # 以城市对为列, 时间为行;
    # 设置列名
    data_df.columns = data_df.iloc[0] + ',' + data_df.iloc[1]
    data_df = data_df.drop([0, 1]).reset_index(drop=True)
    # 设置需要预测的城市对数量
    if citys_range == None:
        citys_range = [0, len(od_df)]
    # 筛选出需要的原始数据
    data_df = data_df.iloc[:,citys_range[0]:citys_range[1]]

    # 初始化预测结果
    precict_df = pd.DataFrame()

    # 对每个城市对进行预测
    citys = 0
    num = 1 # 预测进度
    while citys < (citys_range[1]-citys_range[0]):
        clear_output(wait=True)  # 清除之前的输出
        print('\n\n', '-'*100, '\n')
        print(' '*40, f"预测进度: {num}/{citys_range[1]-citys_range[0]}")
        print('\n', '-'*100, '\n\n')

        origin_df = data_df.iloc[:, citys:citys+1]

        # 数据拉伸
        y = stretch_data(
                            origin_df = origin_df,
                            start = 119,
                            day = 5,
                            insert_num = 4,
                    )
        # 调用模型进行预测
        forecast = Prophet_forecast(
                                    data_df = y,
                                    start_date = start_date, 
                                    precict_day = precict_day,
                                    front_impact = front_impact, 
                                    rear_impact = rear_impact,
                                )
        # 将真实值和预测值加起来
        true_data = origin_df[origin_df.columns[0]].iloc[:-precict_day].values.tolist()
        pre_data = forecast['yhat'].values.tolist()        
        
        predict_i = true_data + pre_data

        # 积累当前计算结果
        precict_df = pd.concat([precict_df, pd.DataFrame(predict_i),], axis = 1)
        citys += 1
        num += 1

    # 设置城市对的列名及时间
    precict_df.columns = data_df.columns

    return data_df, precict_df