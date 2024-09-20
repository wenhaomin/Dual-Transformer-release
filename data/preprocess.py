# -*- coding: utf-8 -*-


#-------------------------------------------------------------------------------------------------------------------------#
#for linux import package
import sys
import os
import platform
file = 'D:\Study\Lab\project\CMU project\haystac\temp_sync\D-Transformer' if platform.system() == 'Windows'  else '/home/haominwe/code/D-Transformer/'
sys.path.append(file)
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk(file) for name in dirs])
#-------------------------------------------------------------------------------------------------------------------------#

import numpy as np
import pandas as pd
import pickle as pk

from utils.util import ws

from scipy import stats

# Define a function to get the first mode
def mode(x):
    return stats.mode(x)[0]

def pre_process(fin, fout, is_test = False):
    print('Raw input file:', fin)

    # read stay data from npy
    # stay = np.load(fin + '/stay.npy', allow_pickle=True)
    # df = pd.DataFrame(stay, columns=['uid', 'lat', 'lng', 'start_time', 'stay_time']) # stay time (unit: minutes)
    """
    data example:
    'uid',      'lat' ,      'lng'    ,    'start_time'            ,       'stay_time'
      1  ,    1.355167,     103.709616,   2023-03-01 00:00:00+00:00,       832
    """

    # read stay data from .csv, for trial1
    # df = pd.read_csv(fin + '/data_stays_poi.csv')
    # df = df.rename(columns={'agent_id':'uid','latitude':'lat','longitude':'lng',
    #                    'start_datetime':'start_time', 'stay_time_minutes':'stay_time', 'POItype':'poi'})

    # read stay data from .csv, for trial2
    # df = pd.read_csv(fin + '/data_stays_poi.csv')
    if is_test:
        df = pd.read_csv(fin + '/stay.csv', nrows=100000)
    else:
        df = pd.read_csv(fin + '/stay.csv')


    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    print('Number of Records:', len(df))

    print('Expand basic information...')
    date, start_hour, start_time_minute = [], [], []
    for x in df['start_time'].tolist():
        x = pd.to_datetime(x)
        date.append(x.date().__str__())
        start_hour.append(x.hour)
        start_time_minute.append(x.hour * 60 + x.minute)

    df['date'] = date
    df['start_hour'] = start_hour
    df['start_time_minute'] = start_time_minute

    df['poi'] = df['poi'].replace('home', 'residential')

    ### Tag assignment: 
    print('Assign Tag: Initial Assignment...')
    df['start_time'] = pd.to_datetime(df['start_time'])
    firstday = df['start_time'].min()
    df['day'] = (df['start_time'] - firstday).dt.days
    
    # Step 1: Filter the training data
    traindata = df[df['day']<=27]

    # Step 2: Sort 'train_data' by 'uid' and then group by 'uid' and 'lat', 'lng' and sum 'stay_time'
    traindata.sort_values('uid', inplace=True)
    grouped = traindata.groupby(['uid', 'lat', 'lng']).agg({'stay_time': 'sum', 'uid': 'count'}).rename(columns={'uid': 'visit_count'})

    # Step 3: Assign 'home', 'top2', 'top3' tags
    grouped.sort_values(['uid', 'stay_time'], ascending=[True, False], inplace=True)
    grouped['rank'] = grouped.groupby('uid')['stay_time'].rank(method='first', ascending=False)
    grouped['tag'] = pd.cut(grouped['rank'], bins=[0, 1, 2, 3, np.inf], labels=['home', 'top2', 'top3', ''])
    grouped['tag'] = grouped['tag'].astype('category')

    # Add 'routine' and 'explore' to the categories
    grouped['tag'] = grouped['tag'].cat.add_categories(['routine', 'explore'])

    # Step 4: Assign 'routine' and 'explore' tags
    grouped.loc[(grouped['tag'] == '') & (grouped['visit_count'] > 1), 'tag'] = 'routine'
    grouped.loc[(grouped['tag'] == '') & (grouped['visit_count'] == 1), 'tag'] = 'explore'

    # Step 5: Merge the tags back into the original DataFrame
    df = df.merge(grouped.reset_index()[['uid', 'lat', 'lng', 'tag']], on=['uid', 'lat', 'lng'], how='left')

    # Step 6: Assign 'explore' to locations in the test data that do not exist in the training data
    df['tag'].fillna('explore', inplace=True)
    df['tag'] = df['tag'].cat.remove_unused_categories()

    print('Assign Tag: top2 Modification Work or top2...')
    # After initial assignment, top2 tag modification: work or top2
    # Condition 1: if avg_duration on weekday >= 5 hours, then it is work
    traindata['departure_time'] = traindata['start_time'] + pd.to_timedelta(traindata['stay_time'], unit='m')
    work_locations = traindata[traindata['tag'] == 'top2']
    total_stay_duration = work_locations.groupby(['uid', 'lat', 'lng'])['stay_time'].sum().reset_index()
    work_locations_condition1 = total_stay_duration[total_stay_duration['stay_time'] / 20 > 5 * 60] # 20 = 4 * 5 weekdays in training period.might need to be adjusted for new datasets

    # Condition 2: reference arrival/departure time = mode(arrival/departure time during training period); 
    # if arrival/departure time is within one hour difference from reference arrival/departure time for at least 3 days a week, it is work; 
    # otherwise, it is top2
    work_locations['start_time_hour'] = work_locations['start_time'].dt.hour
    work_locations['departure_time_hour'] = work_locations['departure_time'].dt.hour
    # reference time = the most usual time that an agent goes and leaves from his top2 location
    ref_times = work_locations.groupby(['uid', 'lat', 'lng']).agg({'start_time_hour': mode, 'departure_time_hour': mode}).reset_index()
    work_locations = pd.merge(work_locations, ref_times, on=['uid', 'lat', 'lng'], suffixes=('', '_refer'))
    work_locations['start_time_diff'] = (work_locations['start_time_hour'] - work_locations['start_time_hour_refer']).abs() <= 1
    work_locations['departure_time_diff'] = (work_locations['departure_time_hour'] - work_locations['departure_time_hour_refer']).abs() <= 1
    # 12 = 4 * 3 weekdays during one month training period. might need to be adjusted for new datasets
    work_locations_condition2 = work_locations.groupby(['uid', 'lat', 'lng']).filter(lambda x: (x['start_time_diff'].sum() >= 12) & (x['departure_time_diff'].sum() >= 12))
    work_locations_condition2 = work_locations_condition2.drop_duplicates(subset=['uid', 'lat','lng'], keep='first')

    work_locations_final = pd.concat([work_locations_condition1, work_locations_condition2])
    work_locations_final = work_locations_final.drop_duplicates(subset=['uid', 'lat','lng'], keep='first')

    merged = pd.merge(df, work_locations_final[['uid', 'lat', 'lng']], on=['uid', 'lat', 'lng'], how='left', indicator=True)
    # For the locations in work_locations (satisfying one of the two conditions), change the tag as 'work' in stays df
    df.loc[(merged['_merge'] == 'both') & (df['tag'] == 'top2'), 'tag'] = 'work'

    # For the remaining locations, change the tag to 'top2' in stays df
    df.loc[(merged['_merge'] == 'left_only') & (df['tag'] == 'top2'), 'tag'] = 'top2'
#-------------------------------------------------------------------------------------------------------------------------#


    # rank data by user, date, user, and time
    df = df.sort_values(by=['date', 'uid', 'start_time'])


    if is_test:
        df = df[:6000]

    save_path = fout + '/stay_test.csv' if is_test else fout + '/stay.csv'
    df.to_csv(save_path, index = False)

    # save the statistics information
    stay_stat_path = fin + "/stay_describle.csv"
    df.describe().to_csv(stay_stat_path)

    return df



if __name__ == "__main__":
    if 0:
        fin = ws + '/data/raw/trial2/'
        fout =  ws + '/data/raw/trial2/'
        df = pre_process(fin=fin, fout=fout, is_test=True)

    # # poi_df = pk.load(open(ws + '/data/raw/trial1/singapore_poi_df.pk', 'rb'))
    # poi_df = pd.read_pickle(ws + '/data/raw/trial1/singapore_poi_df.pk')
    # print(poi_df[:5])

    if 0: #  for trial2 data, only take the first 1w user's data
        fin = ws + '/data/raw/trial2/' + 'stay.csv'
        fout = ws + '/data/raw/trial2/' + 'stay_1w.csv'
        df = pd.read_csv(fin)
        u_lst = list(range(0, 10001))
        df = df[df['uid'].isin(u_lst)]
        df.to_csv(fout, index = False)

    if 1:
        import pickle
        fin = ws + '/data/raw/trial2/all_gts_lasim_agent_day_list2.pkl'
        data = pickle.load(open(fin, 'rb'))
        a = 0


