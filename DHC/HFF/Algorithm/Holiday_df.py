from Parameters import get_holidays
import pandas as pd
from datetime import datetime, timedelta

def get_preDate(date, day):
    """
    计算给定日期的前 day 天

    date (str): 当前日期(yyyy-mm-dd);
    day (int): 往前追溯的天数;

    return (list): 包含前 day 天的字符串;
    """
    previous_set = [] # 存放前 day 天的日期
    for i in range(day, 0, -1):
        # 计算当前日期前第 i 天的日期
        previous_date = date - timedelta(days=i)
        # 将日期对象转换为字符串,并添加到列表中
        previous_set.append(previous_date.strftime('%Y-%m-%d'))
    
    return previous_set

def get_afterDate(date, day):
    """
    计算给定日期的后 day 天

    date (str): 当前日期(yyyy-mm-dd);
    day (int): 往后追溯的天数;

    return (list): 包含后 day 天的字符串;
    """
    after_set = [] # 存放后 day 天的日期
    for i in range(1, day + 1):
        # 计算当前日期后第 i 天的日期
        after_date = date + timedelta(days=i)
        # 将日期对象转换为字符串,并添加到列表中
        after_set.append(after_date.strftime('%Y-%m-%d'))
    
    return after_set

def get_datelist(holiday_dict, front_impact, rear_impact, year = 2023):
    """
    对每个节假日字典进行处理, 得到影响力日期List;

    holiday_dict (dict): 包含'month', 'day', 'lower_window', 'upper_window', 'spcial_work_day';
    front_impact (int): 假期前影响力;
    rear_impact (int): 假期后影响天数;
    year (int): 年份, 默认为2023年

    return : total_date(list): 所有日期字符串(str)的列表;
              date_order(list): 每天所属日期的第几天(int)列表;

    """
    date = '%s-%s-%s'%(year, holiday_dict['month'], holiday_dict['day'])
    # 将字符串转换为 datetime 对象
    date = datetime.strptime(date, '%Y-%m-%d')
    # 获取前后天数的 list
    previous_set = get_preDate(date, holiday_dict['lower_window'])
    after_set = get_afterDate(date, holiday_dict['upper_window'])

    # 计算该节假日的所有影响天数
    total_date = previous_set + [date.strftime('%Y-%m-%d')] + after_set

    # 计算假期的天数
    holiday_num = (len(total_date)-front_impact-rear_impact)
    
    # 将假期前的影响天数标记为 1, 假期后的影响天数标记为 2, 假期中的天数标记为"总天数 * 10 + 第几天"
    date_order = [1 for _ in range(front_impact)]
    date_order += [(holiday_num*10+i) for i in range(1,1+holiday_num)]
    date_order += [2 for _ in range(rear_impact)]

    # 加入因节假日而调休的周末日期,并将序号设为0
    total_date +=  holiday_dict['spcial_work_day']
    date_order += [0 for _ in range(len(holiday_dict['spcial_work_day']))]
    
    return total_date, date_order

def getHolidays_df(front_impact, rear_impact):
    """
    获取节假日的信息,包括日期和序号;

    front_impact (int): 每个节日前的影响天数;
    rear_impact (int): 每个节日后的影响天数;

    return : order (DataFrame): 一列“ds”, 一列日期序号;
    """
    # 获取假期字典
    holidays = get_holidays(front_impact = front_impact, rear_impact = rear_impact)
    holiday_dates = [] # 节假日日期列表
    date_order_set = [] # 日期序号列表
    
    # 对每个节日进行循环
    for day in holidays:
        total_date, date_order = get_datelist(day, front_impact, rear_impact)
        holiday_dates += total_date
        date_order_set += date_order

    # 得到 order_df
    order_df = pd.DataFrame({"ds": pd.to_datetime(holiday_dates),
                                "date_order": date_order_set})
    
    return order_df.reset_index(drop = True)

def make_columns(origin_df):
    """
    将每个事件列为一列

    order_df: 天数序号;

    return: 将每个事件列为一列的datafram;
    """
    holidays_df = origin_df.copy()
    # 获得所有标签
    value_list = holidays_df['date_order'].drop_duplicates().values.tolist()
    
    for order in value_list:
        if order == 0:
            # 标记调休的周六周天时间
            holidays_df['spcial_work_day'] = (holidays_df['date_order'] == order).astype(int)
            # 标记 “holiday” 事件, 即真正假期时间 + 前后影响时间
            holidays_df['holiday'] = (holidays_df['date_order'] != order).astype(int)
        else:
            # 标记前后假期时间和假期时间
            holidays_df['day_%s'%order] = (holidays_df['date_order'] == order).astype(int)
    
    del holidays_df['date_order']
    holidays_df = holidays_df.groupby('ds').sum().reset_index()
    
    return holidays_df