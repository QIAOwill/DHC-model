import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
pd.options.mode.chained_assignment = None
from IPython.display import clear_output

def calculate_features(df:pd.DataFrame, before_days):
    """
    输入一个只有一列的时间序列，输出该时间序列的各个特征
    参数：
        df (pd.DataFrame): 只有一列的时间序列;
        before_days (int): 滞后指标范围;
    
    return: 包含各个特征的DataFrame;
    """
    feature_df = df.copy()
    col = feature_df.columns[0]
    # 计算前三天的平均值
    feature_df['mean_last_%s_days'%before_days] = feature_df[col].rolling(window = before_days).mean().shift(1)

    # 计算趋势指标（线性回归的斜率）
    def calc_trend(series):
        if len(series) < 3:
            return np.nan
        y = series.values
        x = np.array(range(1, 4))
        A = np.vstack([x, np.ones(len(x))]).T
        slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        return slope

    feature_df['trend_last_%s_days'%before_days] = feature_df[col].rolling(window = before_days).apply(calc_trend, raw=False).shift(1)

    # 计算差异指标
    feature_df['diff_3rd_1st_day'] = feature_df[col].diff(2).shift(1)

    # 计算日环比增长率
    for i in range(before_days-1):
        feature_df['day_over_day_growth_%s'%(i+1)] = feature_df[col].pct_change(i+1).shift(1)

    # 计算标准差
    feature_df['std_last_%s_days'%before_days] = feature_df[col].rolling(window = before_days).std().shift(1)

    # 计算时滞嵌入特征（前1天和前2天的流量）
    for i in range(1,before_days+1):
        feature_df['lag_%s'%i] = feature_df[col].shift(i)

    # 计算差分特征
    for i in range(1,before_days+1):
        feature_df['diff_%s'%i] = feature_df[col].diff(i)
    
    # 丢弃流量列，只保留特征列
    return feature_df.iloc[:,1:]

def PCA_features(feature_df:pd.DataFrame):
    
    features = feature_df.copy()
    # 替换无限值为NaN
    features.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 删除包含NaN值的行
    features = features.fillna(0)

    # 进行PCA
    pca = PCA(n_components=0.95)  # 保留95%的方差
    principal_components = pca.fit_transform(features)

    # 创建主成分DataFrame
    pca_df = pd.DataFrame(data=principal_components, 
                        index=[i for i in range(len(feature_df))],
                        columns=[f'PC{i+1}' for i in range(principal_components.shape[1])])
    
    # 将主成分添加回原始数据中
    #pca_df = pd.concat([feature_df[feature_df.columns[0]], pca_df], axis=1)
    #pca_df = pca_df.fillna(0)
    
    return pca_df

def add_feature(time_data:pd.DataFrame,
                lags = 3):
    
    feature_list = []
    for i in range(len(time_data.columns)):
        if i%5 == 0:
            clear_output(wait=True)  # 清除之前的输出
        print('-'*100,'\n进度： %s / %s\n'%(i+1, len(time_data.columns)))
        column_df = time_data.iloc[:,i:i+1]
        feature_df = calculate_features(column_df, before_days = lags)
        pca_df = PCA_features(feature_df)
        feature_list.append(pca_df)

    feature_data = pd.concat(feature_list, axis=1)
    feature_data.columns = [i for i in range(len(feature_data.columns))]
    
    return feature_data

if __name__ == '__main__':
    time_data = pd.read_csv(r'D:\SEU-all\Self Works\Prediction\Data\Time Data\Time Data.txt', sep = '\t', header = 0)
    feature_data = add_feature(time_data.iloc[:, 1:])
    feature_data.to_csv(r'D:\SEU-all\Self Works\Prediction\Data\PCA.txt', sep = '\t', index=False)
