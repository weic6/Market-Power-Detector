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
import plotly.express as px


def draw_bid_curves_for_multiple_days_2d(
    data_multiple_days, hr_specified=None, cluster_labels=None
):
    """
    Draw 2D bidding curves for multiple days.
    If hr_specified is not None, then draw the curve for the specified hour only.
    """
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
    unique_days = sorted(
        df["SCH_BID_TIMEINTERVALSTART"].dt.date.unique()
    )  # eg unique_days = [datetime.date(2023, 7, 23), ....,datetime.date(2023, 9, 24)]

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

    day_hour_start = 0
    for i, day in enumerate(unique_days):
        day_hour_start += 24 * i
        # print("day_hour_start: ", day_hour_start)
        day_df = df[df["SCH_BID_TIMEINTERVALSTART"].dt.date == day].copy()

        day_df["SCH_BID_TIMEINTERVALSTART_HR"] = pd.to_datetime(
            day_df["SCH_BID_TIMEINTERVALSTART"]
        ).dt.hour
        day_df["SCH_BID_TIMEINTERVALSTOP_HR"] = pd.to_datetime(
            day_df["SCH_BID_TIMEINTERVALSTOP"]
        ).dt.hour

        day_df.sort_values(
            by=[
                "SCH_BID_TIMEINTERVALSTART_HR",
                "SCH_BID_TIMEINTERVALSTOP_HR",
                "SCH_BID_XAXISDATA",
            ],
            inplace=True,
        )
        # create the stepwise curve for each day
        unique_pairs = set(
            zip(
                day_df["SCH_BID_TIMEINTERVALSTART_HR"],
                day_df["SCH_BID_TIMEINTERVALSTOP_HR"],
            )
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
                subset = day_df[day_df["SCH_BID_TIMEINTERVALSTART_HR"] == time_start]
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
                if hr_specified is not None:
                    if hour != hr_specified:
                        continue
                    if cluster_labels is not None:
                        time_stamp = pd.to_datetime(f"{day} {time_start}:00:00")
                        # print(f"time_stamp is: {time_stamp}")
                        # if time_stamp in cluster_labels.index:
                        #     label = cluster_labels.loc[time_stamp, "cluster_label"]
                        #     color = colors[label % len(colors)]
                        # else:
                        #     result = f"Timestamp {time_stamp} not found in the DataFrame index."
                        assert time_stamp in cluster_labels.index
                        label = cluster_labels.loc[time_stamp, "cluster_label"]
                        color = colors[label % len(colors)]
                subset = day_df[day_df["SCH_BID_TIMEINTERVALSTART_HR"] == time_start]
                x_data = []  # MW
                y_data = []  # $

                for i in range(len(subset) - 1):
                    x_data.extend(
                        [
                            subset.iloc[i]["SCH_BID_XAXISDATA"],
                            subset.iloc[i + 1]["SCH_BID_XAXISDATA"],
                        ]
                    )
                    y_data.extend(
                        [
                            subset.iloc[i]["SCH_BID_Y1AXISDATA"],
                            subset.iloc[i]["SCH_BID_Y1AXISDATA"],
                        ]
                    )

                # Add the last point
                x_data.append(subset.iloc[-1]["SCH_BID_XAXISDATA"])
                y_data.append(subset.iloc[-1]["SCH_BID_Y1AXISDATA"])

                # # manually extend the last step in the graph
                # x_data.append(cumulative_hour)
                # y_data.append(max_y)
                # z_data.append(subset.iloc[-1]["SCH_BID_Y1AXISDATA"])

                fig.add_trace(
                    go.Scatter(
                        x=x_data,
                        y=y_data,
                        mode="lines",
                        line=dict(color=color),
                        name=f"Date {day} Hour {hour}",
                    )
                )

    # Update the layout once for the entire figure
    fig_title = f"bidding curve for resource id = {RESOURCEBID_SEQ} from {START_DAY} to {END_DAY}"
    if hr_specified is not None:
        fig_title += f", at hour {hr_specified}"
    fig.update_layout(
        title=fig_title,
        scene=dict(
            xaxis_title="MW (Megawatts)",
            yaxis_title="Price ($)",
        ),
    )

    fig.show()


def draw_bid_curves_for_multiple_days(data_multiple_days, hr_specified=None):
    """
    Draw 3D bidding curves for multiple days.
    if hr_specified is not None, then draw the curve for the specified hour only
    """
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

        day_df["SCH_BID_TIMEINTERVALSTART_HR"] = pd.to_datetime(
            day_df["SCH_BID_TIMEINTERVALSTART"]
        ).dt.hour
        day_df["SCH_BID_TIMEINTERVALSTOP_HR"] = pd.to_datetime(
            day_df["SCH_BID_TIMEINTERVALSTOP"]
        ).dt.hour

        day_df.sort_values(
            by=[
                "SCH_BID_TIMEINTERVALSTART_HR",
                "SCH_BID_TIMEINTERVALSTOP_HR",
                "SCH_BID_XAXISDATA",
            ],
            inplace=True,
        )

        # create the stepwise curve for each day
        unique_pairs = set(
            zip(
                day_df["SCH_BID_TIMEINTERVALSTART_HR"],
                day_df["SCH_BID_TIMEINTERVALSTOP_HR"],
            )
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
                subset = day_df[day_df["SCH_BID_TIMEINTERVALSTART_HR"] == time_start]
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

                if hr_specified is not None:
                    if hour != hr_specified:
                        continue
                cumulative_hour = hour + day_hour_start
                subset = day_df[day_df["SCH_BID_TIMEINTERVALSTART_HR"] == time_start]
                x_data = []  # time
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
    fig_title = f"bidding curve for resource id = {RESOURCEBID_SEQ} from {START_DAY} to {END_DAY}"
    if hr_specified is not None:
        fig_title += f", at hour {hr_specified}"
    fig.update_layout(
        title=fig_title,
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
    # draw_bid_curves_for_multiple_days(filtered_df, 18)
    draw_bid_curves_for_multiple_days_2d(filtered_df, 18)


if __name__ == "__main__":
    main()
