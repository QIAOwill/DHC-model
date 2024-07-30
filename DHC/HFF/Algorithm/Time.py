import time

def get_interval(time_seconds):
    """
    将秒数转换为小时、分钟和秒
    返回: (hours, minutes, seconds)
    """
    hours = int(time_seconds // 3600)
    minutes = int((time_seconds % 3600) // 60)
    seconds = int(time_seconds % 60)

    return hours, minutes, seconds

def get_normal_time(timestamp):
    """
    将time.time()获得的时间转化为 '年-月-日 时-分-秒' 的形式
    """
    tuple_time = time.localtime(timestamp)
    normal_time = time.strftime("%Y-%m-%d %H:%M:%S", tuple_time)
    return normal_time

def output_duration(start_time, end_time, print_time = False):
    """
    计算算法运行时间, 输出格式为'xx小时xx分钟xx秒'

    参数:
        start_time (time): 算法运行起始时间;
        end_time (time): 算法运行结束时间;
        print_time (bool): 控制是否print结果;

    return: 规范化的起始、终止时间以及算法运行总时长('xx小时xx分钟xx秒') 
    """
    # 计算运行时间（单位：秒）
    elapsed_time_seconds = end_time - start_time

    hours, minutes, seconds = get_interval(elapsed_time_seconds)

    # 格式化时间字符串
    formatted_time = f"{hours}小时{minutes}分钟{seconds}秒"
    normal_start_time = get_normal_time(start_time)
    normal_end_time = get_normal_time(end_time)

    if print_time == True:
        # 运行时间输出
        print('\n\n', '-'*100, '\n')
        print(' '*35, f"* 算法开始时间为: {normal_start_time}")
        print(' '*35, f"* 算法终止时间为: {normal_end_time}\n")
        print(' '*35, f"算法运行总时长为: {formatted_time}")
        print('\n', '-'*100, '\n\n')

    return normal_start_time, normal_end_time, formatted_time

