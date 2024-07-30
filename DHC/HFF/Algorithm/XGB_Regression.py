from xgboost import XGBRegressor

def XGB_boost(  
                forecast,
                precict_day,
                drop_set = ['y',
                            'yhat1', ],
                            #'trend', 
                            #'season_weekly', 
                            #'season_daily', ], 
                y = 'y',
            ):
    """
    
    对NeuralProphet的预测结果进行 XGB-boost回归拟合;

    参数: 
        forecast (DataFrame): NeuralProphet的预测结果;
        precict_day (int): 预测数据大小;
        drop_set (list for str): 需要从forecast中删除的列名, 构建训练数据; 
        y (str): 训练的目标值;

    return (np.array): 预测结果 & 真实值

    """
    data = forecast.copy()
    data.set_index('ds', inplace=True)
    test_size = precict_day

    # 构造XGBR数据集
    X = data.drop(drop_set, axis=1)
    y = data[y]

    # 切分数据集
    X_train, X_test = X.iloc[:-test_size,:], X.iloc[-test_size:,:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

    # 模型拟合
    model_xgbr = XGBRegressor()  # 建立XGBR对象
    model_xgbr.fit(X_train, y_train)

    # 预测结果
    predict = model_xgbr.predict(X)

    return predict, y_test