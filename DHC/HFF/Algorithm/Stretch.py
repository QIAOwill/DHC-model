import pandas as pd

def stretch_data(
        origin_df: pd.DataFrame, 
        start = 118, 
        day = 6, 
        insert_num = 3):
    """
    将输入的单列DataFrame在五一假期的流量数据进行拉伸, 实现对国庆进行标定的目的

    参数: 
        origin_df (pd.DataFrame): 单列datafram数据; 
        start (int): 起始数据行(4月29日), 默认为: 118; 
        day (int): 需要拉伸的范围, 默认为: 6; 
        insert_num (int): 差值数, 默认为: 3;

    return: 新数据覆盖后(拷贝后覆盖)的DataFrame.
    """
    df = origin_df.copy()
    # 提取指定范围的原始数据列并转换为列表
    data = df.iloc[start : start + day][df.columns[0]].values.tolist()

    # 结果列表，用于存储拉伸后的数据
    result = []
    result.append(data[0])

    i = 0
    while (i + 1) < len(data):
        if (i+1) <= insert_num:
            # 计算斜率
            ratio = (data[i + 1] - data[i]) / 1
            # 计算插值数据并添加到结果列表
            insert_data = data[i] + 0.5* ratio
            result.append(int(insert_data))
        result.append(int(data[i + 1]))
        i += 1

    # 确定要写回结果的行范围
    start_index = start
    end_index = min(start + len(result), len(df))

    df.iloc[start_index:end_index, 0] = result

    return df