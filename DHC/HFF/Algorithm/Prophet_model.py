import pandas as pd
import numpy as np
np.random.seed(54321)
from Holiday_df import make_columns, getHolidays_df
from prophet import Prophet

def Prophet_forecast(data_df: pd.DataFrame, 
                    start_date = '2023-01-01', 
                    precict_day = 7,
                    front_impact = 0, 
                    rear_impact = 0,):
    """
    根据输入的单变量时间序列, 采用Prophet模型进行预测
    
    参数: 
        data_df (DataFrame): 单变量时间序列数据;
        start_date (str) 起始时间, 默认为: '2023-01-01'; 
        precict_day (int) 预测天数, 默认为: 7;
        front_impact (int) 节假日前影响期, 默认为: 0; 
        rear_impact (int) 节假日后影响期, 默认为: 0;

    return: 预测模型, 预测结果 (tuple[NeuralProphet, DataFrame])
    """
    # 初始化 Prophet 模型
    model = Prophet(
                    growth='linear',
                    seasonality_mode='additive',
                    interval_width=0.95,
                    daily_seasonality=True,
                    weekly_seasonality=False,
                    yearly_seasonality=False,
                    changepoint_prior_scale=0.9,
                    #holidays=holiday_set,
                )

    # 设置标准的列名
    column_df = data_df.copy()
    column_df = column_df.reset_index(drop = True)
    column_df.columns = ['y']
    column_df['ds'] = pd.date_range(start = start_date, periods = len(column_df))

    # 获取节假日数据
    holidays_df = make_columns(getHolidays_df(front_impact = front_impact, 
                                              rear_impact = rear_impact ))
    # 将预测数据与节假日数据结合
    data = pd.merge(column_df, holidays_df, how = 'left', on = 'ds')
    data.fillna(0, inplace=True)

    # 切分数据集
    train, test = data.iloc[:-precict_day,:], data.iloc[-precict_day:,:]
    #  添加额外的事项
    for col in holidays_df.columns:
        if col != 'ds':
            model.add_regressor(col)

    model.fit(train)

    # 测试集提取preheat特征
    predictions_test = model.predict(test.drop('y', axis=1))
    
    return predictions_test