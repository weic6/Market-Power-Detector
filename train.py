import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
import sys
from utils import printStats, plot_dist_mat, draw_distribution, scale_feat


def test():
    # Import the necessary libraries

    # Load the dataset
    train_data = pd.read_csv("train_combined.csv")

    valid_data = pd.read_csv("valid_combined.csv")

    # Preprocess the data (if needed)
    # ...

    # Perform feature scaling (if needed)
    # ...

    # Calculate the distance matrix from input features
    distance_matrix = distance_matrix(data, data)

    # Initialize the unsupervised learning model
    model = KMeans(n_clusters=3)  # Specify the number of clusters
    # Fit the model to the distance matrix
    model.fit(distance_matrix)
    # Get the predicted labels for the data points
    labels = model.labels_
    # Perform further analysis or visualization with the obtained labels

    # ...

    # Evaluate the performance of the model (if applicable)
    # ...

    # Save the model (if needed)
    # ...

    pass


def filter_rows(df, HOUR=None, RESOURCEBID_SEQ=None):
    """
    filter data to specific hour and and specific resourcebid_seq
    """
    df["hr_start"] = df["SCH_BID_TIMEINTERVALSTART"].dt.hour
    df["hr_stop"] = df["SCH_BID_TIMEINTERVALSTOP"].dt.hour
    df.loc[
        df["hr_stop"] == 0,
        "hr_stop",
    ] = 24

    if RESOURCEBID_SEQ is not None:
        df = df[df["RESOURCEBID_SEQ"] == RESOURCEBID_SEQ].copy()

    if HOUR is not None:
        df = df[(df["hr_start"] <= HOUR) & (df["hr_stop"] > HOUR)].copy()

    df.sort_values(
        by=[
            "SCH_BID_TIMEINTERVALSTART",
            "SCH_BID_TIMEINTERVALSTOP",
            "SCH_BID_XAXISDATA",
        ],
        inplace=True,
    )

    return df


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
        # Extract all bidding prices for the current date
        prices = [row[1] for row in matrix]
        # print(prices)
        # Calculate max, min, and range of prices
        max_price = max(prices)
        min_price = min(prices)
        avg_price = np.mean(prices)
        price_range = max_price - min_price
        # Save the features for the current date
        features[start_time] = {
            "max_price": max_price,
            "min_price": min_price,
            "avg_price": avg_price,
            "price_range": price_range,
            "num_steps": len(prices),
        }

    # for date, feature in features.items():
    #     print(f'Date: {date}, Features: {feature}')

    # print(f'features:\n{features}')
    df_train_feat = pd.DataFrame.from_dict(features, orient="index")
    return df_train_feat


def read_data(file):
    """
    read data from csv file

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
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances
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

    import plotly.graph_objects as go

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


def plot_inter_intra_err(df_train_feat_scaled, min_clusters, max_clusters):
    """
    Plot the inter-cluster (Sum of Squared Distances Between Centroids) and intra-cluster errors
    (Sum of Squared Distances of each sample to their closest cluster centroid) for different number
    of clusters, and annotate the plot with the best model's number of clusters.
    """
    import plotly.graph_objects as go

    traces = find_trace_inter_intra_err(
        df_train_feat_scaled, min_clusters, max_clusters
    )

    fig = go.Figure()
    for trace in traces:
        fig.add_trace(trace)

    fig.update_layout(
        title="Error vs Number of Clusters",
        xaxis_title="Number of Clusters",
        yaxis_title="Error",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.show()


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
