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
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
import sys
import plotly.graph_objects as go
import plotly.express as px


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


def filter_rows(df, HOUR=None, RESOURCEBID_SEQ=None):
    """
    filter data to specific hour and and specific resourcebid_seq
    """
    df_new = df.copy()
    df_new["hr_start"] = df_new["SCH_BID_TIMEINTERVALSTART"].dt.hour
    df_new["hr_stop"] = df_new["SCH_BID_TIMEINTERVALSTOP"].dt.hour
    df_new.loc[
        df_new["hr_stop"] == 0,
        "hr_stop",
    ] = 24

    if RESOURCEBID_SEQ is not None:
        df_new = df_new[df_new["RESOURCEBID_SEQ"] == RESOURCEBID_SEQ].copy()

    if HOUR is not None:
        df_new = df_new[
            (df_new["hr_start"] <= HOUR) & (df_new["hr_stop"] > HOUR)
        ].copy()

    df_new.sort_values(
        by=[
            "SCH_BID_TIMEINTERVALSTART",
            "SCH_BID_TIMEINTERVALSTOP",
            "SCH_BID_XAXISDATA",
        ],
        inplace=True,
    )

    return df_new


def extract_feat_from_bid(df_data):
    """
    extract features from the bidding data
    """
    matrices = {}
    grouped = df_data.groupby("SCH_BID_TIMEINTERVALSTART")

    for start_time, group in grouped:
        # For each group, select SCH_BID_XAXISDATA and SCH_BID_Y1AXISDATA and convert to a list of lists
        matrix = group[["SCH_BID_XAXISDATA", "SCH_BID_Y1AXISDATA"]].values.tolist()
        matrices[start_time] = matrix

    # print(matrices)

    # for start_time, matrix in matrices.items():
    #     print(f"Start Time: {start_time}, Matrix:\n{matrix}\n")

    features = {}
    for start_time, matrix in matrices.items():
        # extract all MW and bidding prices for the current date
        megawatt = [row[0] for row in matrix]
        prices = [row[1] for row in matrix]
        # print(prices)
        # Calculate max, min, and range of prices
        max_MW = max(megawatt)
        max_price = max(prices)
        min_price = min(prices)
        avg_price = np.mean(prices)
        price_range = max_price - min_price
        # Save the features for the current date
        features[start_time] = {
            "max_MW": max_MW,
            "max_price": max_price,
            "min_price": min_price,
            # "avg_price": avg_price,
            # "price_range": price_range,
            "num_steps": len(prices),
        }

    # for date, feature in features.items():
    #     print(f'Date: {date}, Features: {feature}')

    # print(f'features:\n{features}')
    df_train_feat = pd.DataFrame.from_dict(features, orient="index")
    return df_train_feat


def read_data(file):
    """
    read data from csv file (and convert the time columns to datetime format)

    Parameters:
    - file (str): the path to the csv file

    Returns:
    - df (pandas DataFrame): the dataframe of the csv file
    """
    df = pd.read_csv(file, low_memory=False)
    df["SCH_BID_TIMEINTERVALSTART"] = pd.to_datetime(df["SCH_BID_TIMEINTERVALSTART"])
    df["SCH_BID_TIMEINTERVALSTOP"] = pd.to_datetime(df["SCH_BID_TIMEINTERVALSTOP"])
    return df


def find_inter_intra_err_and_best_model(
    df_train_feat_scaled, min_clusters, max_clusters
):
    """
    find the inter-cluster, intra-cluster errors for different number of clusters, and find the best model where distance between intra and inter cluster is minimal

    Returns:
    - intra_cluster_error (list): Sum of Squared Distances of each sample to their closest cluster centroid for each cluster number from min_clusters to max_clusters
    - inter_cluster_error (list): Sum of Squared Distances Between Centroids for each cluster number from min_clusters to max_clusters
    - best_model_dict (dict): the best model with kmeans model, n_cluster, and minimal distance between intra and inter cluster.
    """
    import warnings
    from sklearn.exceptions import ConvergenceWarning

    intra_cluster_error = []
    inter_cluster_error = []

    best_model_dict = {
        "n_clusters": 0,
        "min_err_diff_intra_inter": float("inf"),
        "kmeans": None,
    }

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", category=ConvergenceWarning
        )  # ignore the convergence warning
        for n_clusters in range(min_clusters, max_clusters):
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(
                df_train_feat_scaled
            )
            intra_cluster_error.append(
                kmeans.inertia_
            )  # the sum of squared distances of samples to their closest cluster center

            centroid = kmeans.cluster_centers_
            # print("centroid:", centroid)
            pairwise_dists = pairwise_distances(centroid, metric="euclidean")
            squared_pairwise_dists = pairwise_dists**2
            sum_squared_distances = np.sum(
                np.triu(squared_pairwise_dists, 1)
            )  # sum the upper triangle part excluding the diagonal
            inter_cluster_error.append(sum_squared_distances)

            # find best model and its relevant parameters
            if len(intra_cluster_error) > 0 and len(inter_cluster_error) > 0:
                err_diff = abs(intra_cluster_error[-1] - inter_cluster_error[-1])
                if err_diff < best_model_dict["min_err_diff_intra_inter"]:
                    best_model_dict["min_err_diff_intra_inter"] = err_diff
                    best_model_dict["n_clusters"] = n_clusters
                    best_model_dict["kmeans"] = kmeans

    return intra_cluster_error, inter_cluster_error, best_model_dict


def find_best_model(df_train_feat_scaled, min_clusters, max_clusters):
    """
    find the best kmeans model with the number of clusters within [min_clusters, max_clusters]

    Returns:
    - best_model: the best kmeans model
    - n_clusters: the number of clusters for the best model
    """

    intra_cluster_error, inter_cluster_error, best_model_dict = (
        find_inter_intra_err_and_best_model(
            df_train_feat_scaled, min_clusters, max_clusters
        )
    )
    return best_model_dict["kmeans"], best_model_dict["n_clusters"]


def find_trace_inter_intra_err(
    df_train_feat_scaled, min_clusters, max_clusters, **kwargs
):
    """
    return traces for the inter-cluster (Sum of Squared Distances Between Centroids) and intra-cluster errors
    (Sum of Squared Distances of each sample to their closest cluster centroid) for different number
    of clusters, and a green cross marking the best model's number of clusters.
    """

    # Access additional parameters using kwargs with defaults
    showlegend = kwargs.get(
        "showlegend", True
    )  # Show legend by default unless specified

    intra_cluster_error, inter_cluster_error, best_model_dict = (
        find_inter_intra_err_and_best_model(
            df_train_feat_scaled, min_clusters, max_clusters
        )
    )

    traces = []
    # intra-cluster error trace
    traces.append(
        go.Scatter(
            name="Intra-cluster error",
            x=list(range(min_clusters, max_clusters)),
            y=intra_cluster_error,
            line=dict(color="red"),
            mode="lines+markers",
            showlegend=showlegend,
        )
    )

    # inter-cluster error trace
    traces.append(
        go.Scatter(
            name="Inter-cluster error",
            x=list(range(min_clusters, max_clusters)),
            y=inter_cluster_error,
            line=dict(color="blue"),
            mode="lines+markers",
            showlegend=showlegend,
        )
    )

    # Mark the best model
    if best_model_dict["n_clusters"] > 0:

        # # mark with a vertical line
        # y_min = min(
        #     min(intra_cluster_error), min(inter_cluster_error)
        # )
        # y_max = max(max(intra_cluster_error), max(inter_cluster_error))
        # traces.append(
        #     go.Scatter(
        #         x=[best_model_dict["n_clusters"], best_model_dict["n_clusters"]],
        #         y=[y_min, y_max],
        #         mode="lines",
        #         line=dict(color="green", width=2, dash="dash"),
        #         showlegend=False,
        #     )
        # )

        # mark with a green cross
        traces.append(
            go.Scatter(
                name="best model",
                x=[best_model_dict["n_clusters"]],
                y=[intra_cluster_error[best_model_dict["n_clusters"] - min_clusters]],
                mode="markers",
                marker=dict(symbol="cross", color="green", size=15),
                showlegend=showlegend,
            )
        )

    return traces


def plot_inter_intra_err(
    df_train_feat_scaled, min_clusters, max_clusters, width=600, height=400
):
    """
    Plot the inter-cluster (Sum of Squared Distances Between Centroids) and intra-cluster errors
    (Sum of Squared Distances of each sample to their closest cluster centroid) for different number
    of clusters, and annotate the plot with the best model's number of clusters.
    """
    traces = find_trace_inter_intra_err(
        df_train_feat_scaled, min_clusters, max_clusters
    )

    fig = go.Figure()
    for trace in traces:
        fig.add_trace(trace)

    fig.update_layout(
        title="Error vs Number of Clusters",
        xaxis_title="Number of Clusters",
        yaxis_title="Distance",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        width=width,
        height=height,
    )

    fig.show()


def plot_3D_3pc_w_labels(df_train_feat_scaled_pca):
    """
    plot 3D scatter plot of the first 3 PCA components with cluster labels
    """
    if "cluster_label" not in df_train_feat_scaled_pca.columns:
        print("Please assign cluster labels to the dataframe first.")
        return

    # PC1, PC2, PC3, cluster_label columns are required
    if len(df_train_feat_scaled_pca.columns) != 3 + 1:
        print("This function works only when the number of PCA components is 3.")
        return

    fig = go.Figure()

    labels = df_train_feat_scaled_pca["cluster_label"].unique()
    colors = px.colors.qualitative.Plotly

    for i, label in enumerate(labels):
        # filter row with current label
        df_label = df_train_feat_scaled_pca[
            df_train_feat_scaled_pca["cluster_label"] == label
        ]

        fig.add_trace(
            go.Scatter3d(
                x=df_label["PC1"],
                y=df_label["PC2"],
                z=df_label["PC3"],
                mode="markers",
                marker=dict(size=5, color=colors[i % len(colors)], opacity=0.8),
                name=f"Cluster {label}",
            )
        )

    fig.update_layout(
        title="3D Scatter Plot of PCA Components",
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
            camera=dict(eye=dict(x=2, y=-1.5, z=1)),
        ),
        legend_title_text="Cluster Labels",
        margin=dict(l=0, r=0, b=0, t=30),
    )

    fig.show()


def plot_3D_3pc(df_train_feat_scaled_pca):
    """
    plot 3D scatter plot of the first 3 PCA components
    """

    if len(df_train_feat_scaled_pca.columns) != 3:
        print("This function works only when the number of PCA components is 3.")
        return

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=df_train_feat_scaled_pca["PC1"],
                y=df_train_feat_scaled_pca["PC2"],
                z=df_train_feat_scaled_pca["PC3"],
                mode="markers",
                marker=dict(size=5, opacity=0.8),
            )
        ]
    )

    fig.update_layout(
        title="3D Scatter Plot of PCA Components",
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        margin=dict(l=0, r=0, b=0, t=30),
    )

    fig.show()


def select_number_of_pca_components(explained_var_ratio, goal_var=0.95):
    """
    select the number of PCA components based on desired explained variance threshold
    """
    total_variance = 0.0
    for i, explained_variance in enumerate(explained_var_ratio):
        total_variance += explained_variance
        if total_variance >= goal_var:
            return i + 1  # return the count of components used
    return len(explained_var_ratio)  # when the loop finishes without reaching the goal


def plot_explain_variance(pca):

    explained_variance = pca.explained_variance_ratio_
    plt.figure(figsize=(10, 5))
    plt.bar(
        range(1, len(explained_variance) + 1),
        explained_variance,
        alpha=0.5,
        align="center",
        label="Individual explained variance",
    )
    plt.step(
        range(1, len(explained_variance) + 1),
        np.cumsum(explained_variance),
        where="mid",
        label="Cumulative explained variance",
    )
    plt.ylabel("Explained variance ratio")
    plt.xlabel("Principal components")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def convert_numpy_pca_to_df(np_train_feat_scaled_pca, df_train_feat_scaled):
    """
    convert the numpy PCA result into a dataframe

    Parameters:
    np_train_feat_scaled_pca (numpy array): the PCA result
    df_train_feat_scaled (dataframe): the scaled feature dataframe

    Returns:
    df_train_feat_scaled_pca (dataframe): the PCA components
    """

    df_train_feat_scaled_pca = pd.DataFrame(
        data=np_train_feat_scaled_pca,
        index=df_train_feat_scaled.index,  # use the original index
        columns=[f"PC{i+1}" for i in range(np_train_feat_scaled_pca.shape[1])],
    )
    return df_train_feat_scaled_pca


def extract_pca_components(df_train_feat_scaled, goal_var=0.95, isPrint=False):
    """
    extract PCA components from the scaled feature dataframe

    Parameters:
    df_train_feat_scaled (dataframe): the scaled feature dataframe
    goal_var (float): the desired explained variance ratio
    isPrint (boolean): whether to print the PCA result

    Returns:
    df_train_feat_scaled_pca (dataframe): the PCA components
    pca.components_ (array): the PCA components
    explained_variance (array): the explained variance ratio
    """
    from sklearn.decomposition import PCA

    # apply PCA and retain all components
    pca_all = PCA()
    pca_all.fit(df_train_feat_scaled)
    explained_variance_all = pca_all.explained_variance_ratio_
    n_components = select_number_of_pca_components(explained_variance_all, goal_var)

    # reapply PCA with the chosen number of components
    pca = PCA(n_components=n_components)
    np_train_feat_scaled_pca = pca.fit_transform(df_train_feat_scaled)
    explained_variance = pca.explained_variance_ratio_

    # convert the PCA result into a dataframe
    df_train_feat_scaled_pca = convert_numpy_pca_to_df(
        np_train_feat_scaled_pca, df_train_feat_scaled
    )

    # print the PCA result
    if isPrint == True:
        print(
            f"Number of components to keep for {100*goal_var}% explained variance: {n_components}"
        )
        print("Original shape: ", df_train_feat_scaled.shape)
        print("Transformed shape: ", np_train_feat_scaled_pca.shape)
        print("PCA components: \n", pca.components_)
        print("PCA explained variance: \n", explained_variance)
        print(
            "Cumulative explained variance by component: \n",
            explained_variance.cumsum(),
        )

    return df_train_feat_scaled_pca, pca.components_, explained_variance, pca


def find_cluster_threshold(
    model_kmeans_train, df_train_feat_scaled_pca, n_cluster, percentile=95
):
    """
    find the percentile th distance to the centroid for each cluster.

    Parameters:
    model_kmeans_train: the trained kmeans model
    df_train_feat_scaled_pca: the PCA components of the scaled train feature dataframe
    n_cluster (int): the number of clusters for the trained kmeans model
    percentile (int): the percentile threshold

    Returns:
    cluster_thresholds (dict): the percentile threshold for each cluster
    """

    clusters = model_kmeans_train.labels_
    centroids = model_kmeans_train.cluster_centers_
    distances = pairwise_distances(
        df_train_feat_scaled_pca, centroids, metric="euclidean"
    )
    distance_to_centroid = np.min(distances, axis=1)
    percentile_thresholds = []
    for i in range(n_cluster):
        cluster_distances = distance_to_centroid[clusters == i]
        percentile_distance = np.percentile(cluster_distances, percentile)
        percentile_thresholds.append(percentile_distance)

    cluster_thresholds = {
        i: threshold for i, threshold in enumerate(percentile_thresholds)
    }

    return cluster_thresholds


def predict_valid_label(
    model_kmeans_train, df_valid_feat_scaled_pca, cluster_thresholds
):
    centroids = model_kmeans_train.cluster_centers_
    valid_distances = pairwise_distances(
        df_valid_feat_scaled_pca, centroids, metric="euclidean"
    )
    valid_min_distances = np.min(valid_distances, axis=1)
    valid_min_indices = np.argmin(valid_distances, axis=1)
    # compare each validation point's distance to its nearest centroid's threshold
    valid_labels = []
    for dist, idx in zip(valid_min_distances, valid_min_indices):
        if dist <= cluster_thresholds[idx]:
            valid_labels.append(idx)
        else:
            valid_labels.append(-1)  # -1 denoting 'not belonging to any cluster'

    # print("df_valid_feat_scaled_pca.shape: ", df_valid_feat_scaled_pca.shape)
    # print(f"=====\nvalid_distances:\n shape={valid_distances.shape}\n{valid_distances}")
    # print(f"=====\nvalid_min_distances:\n shape={valid_min_distances.shape}\n{valid_min_distances}")
    # print(f"=====\valid_min_indices:\n shape={valid_min_indices.shape}\n{valid_min_indices}")
    return valid_labels


# def main():
#     train_file = "train_combined_pub_0313_0905.csv"  # train_file = "train_combined.csv"
#     df = read_data(train_file)
#     # print(df)

#     columns_to_drop = [
#         "MAXEOHSTATEOFCHARGE",
#         "PRODUCTBID_DESC",
#         "PRODUCTBID_MRID",
#         "MARKETPRODUCT_DESC",
#         "SCH_BID_Y2AXISDATA",
#         "MINEOHSTATEOFCHARGE",
#         "MAXEOHSTATEOFCHARGE",
#         "STARTTIME",
#         "STOPTIME",
#         "RESOURCE_TYPE",
#         "TIMEINTERVALSTART",
#         "TIMEINTERVALEND",
#         "MARKETPRODUCTTYPE",
#         "SCH_BID_TIMEINTERVALSTART_GMT",
#         "SCH_BID_TIMEINTERVALSTOP_GMT",
#         "SCH_BID_CURVETYPE",
#     ]

#     HOUR = 12
#     RESOURCEBID_SEQ = 100651  # None #100651
#     df_droped = df.drop(columns=columns_to_drop, axis="columns")
#     printStats(df_droped)
#     df_filtered = filter_rows(df_droped, HOUR, RESOURCEBID_SEQ)

#     # draw bid curve
#     from draw_bid_curve import draw_bid_curves_for_multiple_days_2d

#     draw_bid_curves_for_multiple_days_2d(df_filtered, HOUR)

#     # extract features and scale them
#     df_train_feat = extract_feat_from_bid(df_filtered)
#     print("\nStatistics before scaling:")
#     printStats(df_train_feat, type=False)

#     df_train_feat_scaled = scale_feat(df_train_feat)
#     print("\nStatistics after scaling:")
#     printStats(df_train_feat_scaled, type=False)

#     # # plot distribution
#     # draw_distribution(df_train_feat)

#     # plot distance matrix
#     plot_dist_mat(df_train_feat_scaled)

#     # plot inter-cluster and intra-cluster errors
#     plot_inter_intra_err(df_train_feat_scaled, min_clusters=2, max_clusters=20)

#     # pick the number of clusters and train the model
#     n_clusters = 3

#     from sklearn.cluster import KMeans

#     kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(df_train_feat_scaled)
#     centroid = kmeans.cluster_centers_
#     # assign the cluster labels to each bid
#     df_train_feat_scaled["cluster_label"] = kmeans.labels_
#     # df_train_feat_scaled.head()

#     cluster_labels = df_train_feat_scaled[["cluster_label"]].copy()
#     # subplot 4: draw bid curves with cluster labels
#     from draw_bid_curve import draw_bid_curves_for_multiple_days_2d

#     draw_bid_curves_for_multiple_days_2d(df_filtered, HOUR, cluster_labels)


# if __name__ == "__main__":
#     sys.exit(main())
