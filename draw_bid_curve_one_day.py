'''
TODO: add annotation 
'''

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json

def draw_bid_curve_for_one_day(data_one_day):
    df = pd.DataFrame(data_one_day)
    RESOURCEBID_SEQ = df["RESOURCEBID_SEQ"].iloc[0]
    DAY = pd.to_datetime(df['SCH_BID_TIMEINTERVALSTART']).dt.date.iloc[0]
    df['SCH_BID_TIMEINTERVALSTART'] = pd.to_datetime(df['SCH_BID_TIMEINTERVALSTART']).dt.hour
    df['SCH_BID_TIMEINTERVALSTOP'] = pd.to_datetime(df['SCH_BID_TIMEINTERVALSTOP']).dt.hour
    # print(df['SCH_BID_TIMEINTERVALSTOP'])
    
    df.sort_values(by=['SCH_BID_TIMEINTERVALSTART', 'SCH_BID_TIMEINTERVALSTOP', 'SCH_BID_XAXISDATA'], inplace=True)

    fig = go.Figure()

    # create the stepwise curve
    unique_pairs = set(zip(df['SCH_BID_TIMEINTERVALSTART'], df['SCH_BID_TIMEINTERVALSTOP']))
    unique_pairs_list = list(unique_pairs)
    unique_pairs_list.sort(key=lambda x: (x[0], x[1]))
    for time_start, time_end in unique_pairs_list:
        for hour in range(time_start, time_end):
            subset = df[df['SCH_BID_TIMEINTERVALSTART'] == time_start]
            x_data = []
            y_data = []
            z_data = []

            for i in range(len(subset) - 1):
                x_data.extend([hour, hour])
                y_data.extend([subset.iloc[i]['SCH_BID_XAXISDATA'], subset.iloc[i + 1]['SCH_BID_XAXISDATA']])
                z_data.extend([subset.iloc[i]['SCH_BID_Y1AXISDATA'], subset.iloc[i]['SCH_BID_Y1AXISDATA']])

            # Add the last point
            x_data.append(hour)
            y_data.append(subset.iloc[-1]['SCH_BID_XAXISDATA'])
            z_data.append(subset.iloc[-1]['SCH_BID_Y1AXISDATA'])

            # manually extend the last step in the graph
            x_data.append(hour)
            y_data.append(subset.iloc[-1]['SCH_BID_XAXISDATA']+1)
            z_data.append(subset.iloc[-1]['SCH_BID_Y1AXISDATA'])
            
            fig.add_trace(go.Scatter3d(x=x_data, y=y_data, z=z_data, mode='lines', name=f"Hour {hour}"))

    
    fig.update_layout(
        title=f"bidding curve for resource id = {RESOURCEBID_SEQ} in a day {DAY}",
        scene=dict(
            xaxis_title='Hour',
            yaxis_title='MW',
            zaxis_title='$',
            camera=dict(
            eye=dict(x=1.25, y=1.25, z=1.25),  # Adjust x, y, z to change the initial view
            up=dict(x=0, y=0, z=1)  # Sets the upward direction (usually Z-axis)
            )
        )
    )

    fig.show()
    
def main():
    # reading csv file 
    df = pd.read_csv("train_combined.csv", low_memory=False)

    #config
    with open('config_draw.json', 'r') as config_file:
        param = json.load(config_file)
    
    COLUMNS_TO_EXTRACT = param['COLUMNS_TO_EXTRACT']
    RESOURCEBID_SEQ = param['RESOURCEBID_SEQ']
    START_SCH_BID_DATE = param['START_SCH_BID_DATE']
    END_SCH_BID_DATE = param['END_SCH_BID_DATE']
    
    END_SCH_BID_DATE = START_SCH_BID_DATE #read one day for now

    #filter rows and columns
    df = df[COLUMNS_TO_EXTRACT]
    df = df[df['RESOURCEBID_SEQ']== RESOURCEBID_SEQ]
    df["SCH_BID_TIMEINTERVALSTART"] = pd.to_datetime(df["SCH_BID_TIMEINTERVALSTART"])
    df["SCH_BID_TIMEINTERVALSTOP"] = pd.to_datetime(df["SCH_BID_TIMEINTERVALSTOP"])

    start_date = pd.to_datetime(START_SCH_BID_DATE).date()
    end_date = pd.to_datetime(END_SCH_BID_DATE).date()
    filtered_df = df[(df['SCH_BID_TIMEINTERVALSTART'].dt.date >= start_date) & 
                    (df['SCH_BID_TIMEINTERVALSTART'].dt.date <= end_date)]

    # print(len(filtered_df))
    # print(filtered_df.head())
    draw_bid_curve_for_one_day(filtered_df)

if __name__ == '__main__':
    main()

