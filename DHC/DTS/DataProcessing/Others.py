import os

def generate_range(gap, count):
    """
    输入间隔(gap), 和个数(count), 产生对应的list
    eg. gap = 300, count = 4; 输出 range_list = [[0, 300], [300, 600], [600, 900], [900, 1200]]
    """
    range_list = []
    for i in range(count):
        range_list.append([i * gap, (i + 1) * gap])
    return range_list

def get_data_source(data_type, Machine, data_range, param_dict:dict):

    """通过设备类型和数据大小, 设置合适的文件路径"""

    if data_type == 'data_mini':
        string = '(min)'
    elif data_type == 'total_data':
        string = ''

    if Machine == 'QIAO-PC':
        head_path = './'
        os.system('cls')

    elif Machine == 'Web-Sever':
        head_path = '/data/coding/DHC/DTS/'
        os.system('clear')

    data_path = head_path + 'Data/Time Data%s.txt'%string
    attribute_path = head_path + 'Data/data_Attribute%s.txt'%string
    output_path = head_path + 'Result/'
    NPPT_path = head_path + 'Data/predict(MinMax)%s.txt'%string
    PCA_path = head_path + 'Data/PCA.txt'

    param_dict['data_path'] = data_path,
    param_dict['attribute_path'] = attribute_path,
    param_dict['output_path'] = output_path,
    param_dict['NPPT_path'] = NPPT_path,
    param_dict['PCA_path'] = PCA_path,
    # 选择使用的数据范围
    param_dict['data_range'] = [data_range]
    
    return param_dict