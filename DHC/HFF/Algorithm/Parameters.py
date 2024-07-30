# Prophet参数设置
def get_holidays(front_impact = 1, rear_impact = 1):
    """
    获得节假日信息
    front_impact: 假期的前影响
    rear_impact: 假期的后影响
    """
    # 节假日设置
    holidays = [
        {
            'name': "New Year's Day", 
            'month': 1, 
            'day': 1, 
            'lower_window': 1 + front_impact, 
            'upper_window': 1 + rear_impact, 
            'spcial_work_day': []
            },
        {
            'name': "Spring Festival", 
            'month': 1, 
            'day': 22, 
            'lower_window': 1 + front_impact, 
            'upper_window': 5 + rear_impact, 
            'spcial_work_day': ['2023-01-28', '2023-01-29',]
            },
        #{
        #    'name': "Tomb-Sweeping Day", 
        #    'month': 4, 
        #    'day': 5, 
        #    'lower_window': 0 + front_impact, 
        #    'upper_window': 0 + rear_impact, 
        #    'spcial_work_day': []
        #    },
        {
            'name': "Labour Day", 
            'month': 5, 
            'day': 1, 
            'lower_window': 5 + front_impact, 
            'upper_window': 7 + rear_impact, 
            'spcial_work_day': ['2023-04-23', '2023-05-06',]
            },
        {
            'name': "Dragon Boat Festival", 
            'month': 6, 
            'day': 22, 
            'lower_window': 0 + front_impact, 
            'upper_window': 2 + rear_impact, 
            'spcial_work_day': ['2023-06-25']
            },
        #{
        #    'name': "Mid-Autumn Festival", 
        #    'month': 9, 
        #    'day': 29, 
        #    'lower_window': 0 + front_impact, 
        #    'upper_window': 0 + rear_impact, 
        #    'spcial_work_day': []
        #    },
        {
            'name': "National Day", 
            'month': 10, 
            'day': 1, 
            'lower_window': 5 + front_impact, 
            'upper_window': 7 + rear_impact, 
            'spcial_work_day': ['2023-10-07', '2023-10-08']
            },
        ]
    return holidays
