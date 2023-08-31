import pandas as pd
import numpy as np

def dt_integration(data_file_lst):
    merged_df = pd.DataFrame()
    for file_path in data_file_lst:
        temp_df = pd.read_csv(file_path)
        observation_label = data_file_lst[0].split('/')[-2]
        if file_path.split('/')[-1].startswith('WELL'):
            instance_type = 'WELL'
        elif file_path.split('/')[-1].startswith('SIMULATED'):
            instance_type = 'SIMULATED'
        elif file_path.split('/')[-1].startswith('DRAWN'):
            instance_type = 'DRAWN'
        temp_df['event_type'] = observation_label
        temp_df['instance_type'] = instance_type
        temp_df['id_label'] = file_path[:-4].split('/')[-1] + '_' + observation_label
        merged_df = pd.concat([merged_df, temp_df])
    return merged_df

def missing_data_proportion(df):
    '''
    1. input (df) : 라벨 단위 observation_df
    2. output (df) : 라벨 단위 instance_df
    '''
    index_lst = []
    missing_lst = []
    columns = df.columns
    instance_names = df['id_label'].unique()

    for i, instance in enumerate(instance_names):
        instance_df = df[df['id_label'] == instance]
        
        # proportion of how many missing observations are included(%)
        missing_num_arr = np.round(instance_df.isna().sum().values / len(instance_df) * 100, decimals=3)
        
        index_lst.append(instance)
        missing_lst.append(missing_num_arr)

    result = pd.DataFrame(data=missing_lst, index = index_lst, columns = columns)
    return result