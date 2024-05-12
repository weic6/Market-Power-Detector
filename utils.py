import os
import zipfile
from datetime import datetime, timedelta
import pytz  # set time zone
from tqdm import tqdm  # add progress bar
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances
import seaborn as sns
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def unzip_files(input_dir, output_dir):
    """
    unzip files inside input_dir to output_dir.
    """

    # list all the ZIP files
    zip_files = [file for file in os.listdir(input_dir) if file.endswith(".zip")]

    with tqdm(total=len(zip_files)) as p_bar:  # init the progress bar
        for file in zip_files:
            filepath = os.path.join(input_dir, file)
            p_bar.set_description(f"Extracting: {file}")
            with zipfile.ZipFile(filepath, "r") as zip_ref:

                zip_ref.extractall(output_dir)  # extract
            p_bar.update(1)  # update the progress bar
    print(f"Successfully extract files into {output_dir}")


def generate_dates(start_date_str, end_date_str):
    """
    Generates a list of date strings for each day between a specified and valid start and end date, inclusive.

    Parameters:
    start_date_str (str): The start date in 'YYYYMMDD' format.
    end_date_str (str): The end date in 'YYYYMMDD' format.

    Returns:
    date_strings: A list of date strings, each in 'YYYYMMDD' format, for each day from the start date to the end date, inclusive.
    """

    # Convert strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y%m%d")
    end_date = datetime.strptime(end_date_str, "%Y%m%d")

    # Generate dates
    current_date = start_date
    date_strings = []
    while current_date <= end_date:
        date_strings.append(current_date.strftime("%Y%m%d"))
        current_date += timedelta(days=1)

    return date_strings


def get_next_date(date_str):
    current_date = datetime.strptime(date_str, "%Y%m%d")
    next_date = current_date + timedelta(days=1)
    next_date_str = next_date.strftime("%Y%m%d")
    return next_date_str


def is_in_dst(date_str):  # return true if date_str is in DST timeframe
    """
    Determine if a given date in California (Pacific Time Zone) is within Daylight Saving Time (DST).

    Parameters:
    date_str (str): A string representing a date in the format "YYYYMMDD".

    Returns:
    True if the date is within DST, False otherwise.
    """
    date_obj = datetime.strptime(date_str, "%Y%m%d")
    california_tz = pytz.timezone("America/Los_Angeles")  # set time zone as California
    date_in_california = california_tz.localize(
        date_obj
    )  # convert the datetime object to California timezone
    return date_in_california.dst() != timedelta(0)


def printStats(df, shape=True, type=True, stats=True, topFewRow=True):
    """
    print the following statistics of the dataframes:
    shape, type of each column, statistics of each column, top few rows.

    Parameters:
    df: pandas dataframe
    shape: bool, default=True
        print the shape of the dataframe
    type: bool, default=False
        print the type of each column in the dataframe
    stats: bool, default=False
        print the statistics of each column in the dataframe
    topFewRow: bool, default=False
        print the top few rows of the dataframe
    Returns:
    None
    """

    if shape:
        print("=====================================================")
        print(f"size of data is: {df.shape}")

    if type:
        print("=====================================================")
        print(f"type of each column is:\n{df.dtypes}")

    if stats:
        print("=====================================================")
        print(f"Statistics in data:\n {df.describe()}")

    if topFewRow:
        print("=====================================================")
        print(f"Top few rows in data:\n{df.head()}")


def old_plot_dist_mat(df):
    """
    plot the distance matrix of the dataframe
    """

    dist_mat = pairwise_distances(df, metric="euclidean")

    # Heatmap of the distance matrix
    # ax = sns.heatmap(dist_mat, square=True, cmap='viridis')
    # ax = sns.heatmap(dist_mat, annot=True, cmap='viridis', xticklabels=df.index.date, yticklabels=df.index.date, square=True)
    # ax = sns.heatmap(dist_mat, cmap='viridis', xticklabels=df.index.date, yticklabels=df.index.date, square=True)
    step = 15
    date_labels = [
        str(date.date()) if idx % step == 0 else "" for idx, date in enumerate(df.index)
    ]
    # print(date_labels)

    ax = sns.heatmap(
        dist_mat,
        cmap="viridis",
        xticklabels=date_labels,  # Label every 'step' dates
        yticklabels=date_labels,  # Label every 'step' dates
        square=True,
    )

    plt.title("Distance Matrix Heatmap")
    ax.set(xlabel="date", ylabel="date")
    ax.xaxis.tick_top()

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)  # rotate x-axis labels
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.show()


def plot_dist_mat(df, width=900, height=900):
    """
    plot the distance matrix of the dataframe
    """

    # Calculate the pairwise distance matrix
    dist_mat = pairwise_distances(df, metric="euclidean")

    dates = pd.to_datetime(df.index.date)
    formatted_dates = dates.strftime("%Y-%m")
    unique_month_years = formatted_dates.unique()
    tickvals = [
        formatted_dates.tolist().index(month_year) for month_year in unique_month_years
    ]
    ticktext = unique_month_years.tolist()

    dates = df.index.date
    heatmap = go.Heatmap(z=dist_mat, x=dates, y=dates, colorscale="Viridis")

    # layout = go.Layout(
    #     title="Distance Matrix Heatmap",
    #     xaxis=dict(title="date", tickangle=90),
    #     yaxis=dict(title="date"),
    #     width=width,
    #     height=height,
    # )

    layout = go.Layout(
        title="Distance Matrix Heatmap",
        xaxis=dict(
            title="date",
            tickangle=270,
            tickvals=tickvals,
            ticktext=ticktext,
            type="category",  # Specify axis type as 'category' to avoid automatic ticks
        ),
        yaxis=dict(
            title="date",
            tickvals=tickvals,
            ticktext=ticktext,
            type="category",  # Specify axis type as 'category' to avoid automatic ticks
        ),
        width=width,
        height=height,
    )

    fig = go.Figure(data=[heatmap], layout=layout)
    fig.show()


def draw_distribution(df):
    """
    draw the distribution of each feature in the DataFrame using Plotly,
    where each dot represents the frequency of a unique value.
    """
    from plotly.subplots import make_subplots

    num_features = len(df.columns)
    cols = 2
    rows = (num_features + 1) // cols

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=df.columns,
        horizontal_spacing=0.08,
        vertical_spacing=0.08,
    )

    subplot_index = 0

    for column in df.columns:
        row_idx = (subplot_index) // cols + 1
        col_idx = (subplot_index) % cols + 1

        value_counts = df[column].value_counts().sort_index()

        fig.add_trace(
            go.Scatter(
                x=value_counts.index,
                y=value_counts.values,
                mode="markers",
                marker=dict(
                    # color=np.random.choice(['red', 'blue', 'green', 'purple', 'orange', 'cyan']),
                    size=10,
                    line=dict(width=2, color="DarkSlateGrey"),
                ),
                name=column,  # legend name
            ),
            row=row_idx,
            col=col_idx,
        )

        subplot_index += 1

    fig.update_layout(
        height=300 * rows,
        width=800,
        title_text="Frequency Distribution of Features",
        showlegend=False,
        # template='plotly_white'
    )

    fig.show()


def scale_feat(df, scaler=None):
    """
    scale the features such that they have mean 0 and variance 1
    """

    if scaler is None:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()

    features_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(features_scaled, index=df.index, columns=df.columns)
    return df_scaled, scaler
