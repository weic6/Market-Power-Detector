"""
This script draws 3D interative bidding curve for generation, for one given resource, within a day range
!!!For now, it does not consider self-schedule bid in drawing.
"""

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import pandas as pd


def draw_bid_curves_for_multiple_days(data_multiple_days):
    df = pd.DataFrame(data_multiple_days)
    RESOURCEBID_SEQ = df["RESOURCEBID_SEQ"].iloc[0]

    df["SCH_BID_TIMEINTERVALSTART"] = pd.to_datetime(df["SCH_BID_TIMEINTERVALSTART"])
    df["SCH_BID_TIMEINTERVALSTOP"] = pd.to_datetime(df["SCH_BID_TIMEINTERVALSTOP"])

    df.sort_values(
        by=[
            "SCH_BID_TIMEINTERVALSTART",
            "SCH_BID_TIMEINTERVALSTOP",
            "SCH_BID_XAXISDATA",
        ],
        inplace=True,
    )

    fig = go.Figure()
    unique_days = sorted(df["SCH_BID_TIMEINTERVALSTART"].dt.date.unique())
    # print("unique_days: ",unique_days)

    # colors of bid curve for each day
    colors = [
        "blue",
        "green",
        "red",
        "cyan",
        "magenta",
        "yellow",
        "black",
        "orange",
        "purple",
        "brown",
    ]

    START_DAY = unique_days[0]
    END_DAY = unique_days[-1]

    print(START_DAY, END_DAY)

    day_hour_start = 0
    for i, day in enumerate(unique_days):
        day_hour_start += 24 * i
        # print("day_hour_start: ", day_hour_start)
        day_df = df[df["SCH_BID_TIMEINTERVALSTART"].dt.date == day].copy()

        day_df["SCH_BID_TIMEINTERVALSTART"] = pd.to_datetime(
            day_df["SCH_BID_TIMEINTERVALSTART"]
        ).dt.hour
        day_df["SCH_BID_TIMEINTERVALSTOP"] = pd.to_datetime(
            day_df["SCH_BID_TIMEINTERVALSTOP"]
        ).dt.hour

        day_df.sort_values(
            by=[
                "SCH_BID_TIMEINTERVALSTART",
                "SCH_BID_TIMEINTERVALSTOP",
                "SCH_BID_XAXISDATA",
            ],
            inplace=True,
        )

        # create the stepwise curve for each day
        unique_pairs = set(
            zip(day_df["SCH_BID_TIMEINTERVALSTART"], day_df["SCH_BID_TIMEINTERVALSTOP"])
        )
        unique_pairs_list = list(unique_pairs)
        unique_pairs_list.sort(key=lambda x: (x[0], x[1]))

        color = colors[i % len(colors)]

        max_y = 180  # to extend the last step
        for time_start, time_end in unique_pairs_list:
            if (
                time_end == 0
            ):  # if time is from 23:00 to 0:00, change it to 23:00 to 24:00
                time_end = 24
            for hour in range(time_start, time_end):
                subset = day_df[day_df["SCH_BID_TIMEINTERVALSTART"] == time_start]
                # print(subset['SCH_BID_XAXISDATA'])
                max_y = max(
                    max(subset["SCH_BID_XAXISDATA"]), max_y
                )  # record max MW to draw beautifully
                # print(max_y)

        for time_start, time_end in unique_pairs_list:
            if (
                time_end == 0
            ):  # if time is from 23:00 to 0:00, change it to 23:00 to 24:00
                time_end = 24
            for hour in range(time_start, time_end):
                cumulative_hour = hour + day_hour_start
                subset = day_df[day_df["SCH_BID_TIMEINTERVALSTART"] == time_start]
                x_data = []  # tme
                y_data = []  # MW
                z_data = []  # $

                for i in range(len(subset) - 1):
                    x_data.extend([cumulative_hour, cumulative_hour])
                    y_data.extend(
                        [
                            subset.iloc[i]["SCH_BID_XAXISDATA"],
                            subset.iloc[i + 1]["SCH_BID_XAXISDATA"],
                        ]
                    )
                    z_data.extend(
                        [
                            subset.iloc[i]["SCH_BID_Y1AXISDATA"],
                            subset.iloc[i]["SCH_BID_Y1AXISDATA"],
                        ]
                    )

                # Add the last point
                x_data.append(cumulative_hour)
                y_data.append(subset.iloc[-1]["SCH_BID_XAXISDATA"])
                z_data.append(subset.iloc[-1]["SCH_BID_Y1AXISDATA"])

                # manually extend the last step in the graph
                x_data.append(cumulative_hour)
                y_data.append(max_y)
                z_data.append(subset.iloc[-1]["SCH_BID_Y1AXISDATA"])

                fig.add_trace(
                    go.Scatter3d(
                        x=x_data,
                        y=y_data,
                        z=z_data,
                        mode="lines",
                        line=dict(color=color),
                        name=f"Date {day} Hour {hour}",
                    )
                )

    # Update the layout once for the entire figure
    fig.update_layout(
        title=f"bidding curve for resource id = {RESOURCEBID_SEQ} from {START_DAY} to {END_DAY}",
        scene=dict(
            # xaxis = dict(nticks=30, range=[0, 10*24*len(unique_days)],),
            # xaxis = dict( nticks=len(unique_days), range=[0, 100*5*len(unique_days)],),
            xaxis_title="Cumulative Hours",
            yaxis_title="MW (Megawatts)",
            zaxis_title="Price ($)",
            camera=dict(
                eye=dict(
                    x=1.5, y=-1.5, z=1
                ),  # Adjust x, y, z to change the initial view
                up=dict(x=0, y=0, z=1),  # Sets the upward direction (usually Z-axis)
            ),
        ),
    )

    fig.show()


# def draw_bid_curve_for_one_day(data_one_day):
#     df = pd.DataFrame(data_one_day)
#     RESOURCEBID_SEQ = df["RESOURCEBID_SEQ"].iloc[0]
#     DAY = pd.to_datetime(df['SCH_BID_TIMEINTERVALSTART']).dt.date.iloc[0]
#     df['SCH_BID_TIMEINTERVALSTART'] = pd.to_datetime(df['SCH_BID_TIMEINTERVALSTART']).dt.hour
#     df['SCH_BID_TIMEINTERVALSTOP'] = pd.to_datetime(df['SCH_BID_TIMEINTERVALSTOP']).dt.hour
#     # print(df['SCH_BID_TIMEINTERVALSTOP'])

#     df.sort_values(by=['SCH_BID_TIMEINTERVALSTART', 'SCH_BID_TIMEINTERVALSTOP', 'SCH_BID_XAXISDATA'], inplace=True)

#     fig = go.Figure()

#     # create the stepwise curve
#     unique_pairs = set(zip(df['SCH_BID_TIMEINTERVALSTART'], df['SCH_BID_TIMEINTERVALSTOP']))
#     unique_pairs_list = list(unique_pairs)
#     unique_pairs_list.sort(key=lambda x: (x[0], x[1]))

#     max_y = 0
#     for time_start, time_end in unique_pairs_list:
#         if time_end == 0: # if time is from 23:00 to 0:00, change it to 23:00 to 24:00
#             time_end = 24
#         for hour in range(time_start, time_end):
#             subset = df[df['SCH_BID_TIMEINTERVALSTART'] == time_start]
#             # print(subset['SCH_BID_XAXISDATA'])
#             max_y = max(max(subset['SCH_BID_XAXISDATA']), max_y) # record max MW to draw beautifully
#             print(max_y)

#     for time_start, time_end in unique_pairs_list:
#         if time_end == 0: # if time is from 23:00 to 0:00, change it to 23:00 to 24:00
#             time_end = 24
#         for hour in range(time_start, time_end):
#             subset = df[df['SCH_BID_TIMEINTERVALSTART'] == time_start]
#             x_data = [] #tme
#             y_data = [] #MW
#             z_data = [] #$

#             for i in range(len(subset) - 1):
#                 x_data.extend([hour, hour])
#                 y_data.extend([subset.iloc[i]['SCH_BID_XAXISDATA'], subset.iloc[i + 1]['SCH_BID_XAXISDATA']])
#                 z_data.extend([subset.iloc[i]['SCH_BID_Y1AXISDATA'], subset.iloc[i]['SCH_BID_Y1AXISDATA']])

#             # Add the last point
#             x_data.append(hour)
#             y_data.append(subset.iloc[-1]['SCH_BID_XAXISDATA'])
#             z_data.append(subset.iloc[-1]['SCH_BID_Y1AXISDATA'])

#             # manually extend the last step in the graph
#             x_data.append(hour)
#             y_data.append(max_y)
#             z_data.append(subset.iloc[-1]['SCH_BID_Y1AXISDATA'])

#             fig.add_trace(go.Scatter3d(x=x_data, y=y_data, z=z_data, mode='lines', name=f"Hour {hour}"))


#     fig.update_layout(
#         title=f"bidding curve for resource id = {RESOURCEBID_SEQ} in a day {DAY}",
#         scene=dict(
#             xaxis_title='Hour',
#             yaxis_title='MW',
#             zaxis_title='$',
#             camera=dict(
#             eye=dict(x=1.5, y=-1.5, z=1),  # Adjust x, y, z to change the initial view
#             up=dict(x=0, y=0, z=1)  # Sets the upward direction (usually Z-axis)
#             )
#         )
#     )

#     fig.show()


def main():
    # reading csv file
    combined_csv_file = "train_combined.csv"  # "valid_combined.csv"
    df = pd.read_csv(combined_csv_file, low_memory=False)

    # config
    with open("config_draw.json", "r") as config_file:
        param = json.load(config_file)

    COLUMNS_TO_EXTRACT = param[
        "COLUMNS_TO_EXTRACT"
    ]  #!!! do not extract self-schedule columns for now
    RESOURCEBID_SEQ = param["RESOURCEBID_SEQ"]
    START_SCH_BID_DATE = param["START_SCH_BID_DATE"]
    END_SCH_BID_DATE = param["END_SCH_BID_DATE"]

    # filter rows and columns
    df = df[COLUMNS_TO_EXTRACT]
    df = df[df["RESOURCEBID_SEQ"] == RESOURCEBID_SEQ]
    df["SCH_BID_TIMEINTERVALSTART"] = pd.to_datetime(df["SCH_BID_TIMEINTERVALSTART"])
    df["SCH_BID_TIMEINTERVALSTOP"] = pd.to_datetime(df["SCH_BID_TIMEINTERVALSTOP"])

    start_date = pd.to_datetime(START_SCH_BID_DATE).date()
    end_date = pd.to_datetime(END_SCH_BID_DATE).date()
    filtered_df = df[
        (df["SCH_BID_TIMEINTERVALSTART"].dt.date >= start_date)
        & (df["SCH_BID_TIMEINTERVALSTART"].dt.date <= end_date)
    ]

    # print(len(filtered_df))
    # print(filtered_df)
    # draw_bid_curve_for_one_day(filtered_df)
    draw_bid_curves_for_multiple_days(filtered_df)


if __name__ == "__main__":
    main()
