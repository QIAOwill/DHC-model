import pandas as pd
from Holiday_df import make_columns, getHolidays_df
from neuralprophet import NeuralProphet

def make_events(holidays_df):
    """
    根据holiday_df, 创建事件;
    """
    events_dict = {}
    # 对每列进行循环
    for event in holidays_df:
        # 跳过"ds"列
        if event == 'ds':
            continue
        # 筛选日期
        date_list = holidays_df[holidays_df[f"{event}"]==1]['ds'].values.tolist()
        # 创建事件字典
        event_df = pd.DataFrame({"event": f"{event}",
                                 "ds": pd.to_datetime(date_list),})
        # 数据储存
        events_dict[f"{event}"] = event_df
        
    return events_dict

def add_event_dates(df, event_df, event_str):
    """ 添加事件列(0-1) """
    df['%s'%event_str] = df['ds'].isin(event_df['ds']).astype(int)
    return df

def NP_forecast(data_df: pd.DataFrame, 
                start_date = '2023-01-01', 
                precict_day = 7,
                front_impact = 0, 
                rear_impact = 0,
                accelerator = None,):
    """
    根据输入的单变量时间序列, 采用Prophet模型进行预测
    
    参数: 
        data_df (DataFrame): 单变量时间序列数据;
        start_date (str) 起始时间, 默认为: '2023-01-01'; 
        precict_day (int) 预测天数, 默认为: 7;
        front_impact (int) 节假日前影响期, 默认为: 0; 
        rear_impact (int) 节假日后影响期, 默认为: 0;
        accelerator (str) : 加速器, 默认为None;

    return: 预测模型, 预测结果 (tuple[NeuralProphet, DataFrame])
    """

    # 初始化 NeuralProphet 模型
    model = NeuralProphet(
        #global_normalization = True,
        #learning_rate = 0.001,
        #n_forecasts=7,
        #n_lags=n_lags,
        #ar_layers = [32,64,128,32],
        #ar_reg = 0.01,
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True,
        normalize = 'minmax',
        accelerator = accelerator,
        #learning_rate = 0.00001,
        #epochs = 8,
        #batch_size = 200,
    )

    # 设置标准的列名
    column_df = data_df.copy()
    column_df = column_df.reset_index(drop = True)
    column_df.columns = ['y']
    column_df['ds'] = pd.date_range(start = start_date, periods = len(column_df))

    holidays_df = make_columns(getHolidays_df(front_impact = front_impact, 
                                              rear_impact = rear_impact ))
    # 获取事件数据
    events_dict = make_events(holidays_df)

    # 为模型添加事件
    for event in events_dict:
        model.add_events(f"{event}")
        column_df = add_event_dates(column_df, events_dict[f"{event}"], f"{event}")

    # 拟合模型
    model.fit(column_df, freq='D')
    df_future = model.make_future_dataframe(
                                            column_df, 
                                            n_historic_predictions=True, 
                                            periods = precict_day)
    
    # 添加事件模型
    for event in events_dict:
        df_future = add_event_dates(df_future, events_dict[f"{event}"], f"{event}")
    
    # 模型预测
    forecast = model.predict(df_future)
    
    return forecast
