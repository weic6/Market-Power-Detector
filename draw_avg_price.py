import sys
import os
from glob import glob
import pandas as pd
from utils import unzip_files
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
import plotly.graph_objects as go

# sys.path.append(os.path.dirname(__file__))


def draw_avg_price_for_multiple_days(data_dir):
    lmp = glob(os.path.join(data_dir, "*LMP*.csv"))[0]
    mce = glob(os.path.join(data_dir, "*MCE*.csv"))[0]
    mcc = glob(os.path.join(data_dir, "*MCC*.csv"))[0]
    mcl = glob(os.path.join(data_dir, "*MCL*.csv"))[0]

    df_mce = pd.read_csv(mce)
    df_mcl = pd.read_csv(mcl)
    df_mcc = pd.read_csv(mcc)
    df_lmp = pd.read_csv(lmp)

    df_combined = pd.concat([df_mce, df_mcl, df_mcc, df_lmp])

    # Step 3: Convert INTERVALSTARTTIME_GMT to datetime
    df_combined["INTERVALSTARTTIME_GMT"] = pd.to_datetime(
        df_combined["INTERVALSTARTTIME_GMT"]
    )

    # print(df_combined.head(10))
    # Plotting
    fig = go.Figure()
    marker_types = {"MCE": "circle", "MCL": "circle", "MCC": "circle", "LMP": "x"}
    colors = {"MCE": "blue", "MCL": "green", "MCC": "red", "LMP": "brown"}

    # Loop through each LMP_TYPE and add a scatter plot for each
    for lmp_type in df_combined["LMP_TYPE"].unique():
        df_filtered = df_combined[df_combined["LMP_TYPE"] == lmp_type]

        for day in df_filtered["OPR_DT"].unique():
            df_day = df_filtered[df_filtered["OPR_DT"] == day].copy()
            df_day.sort_values(by=["INTERVALSTARTTIME_GMT"], inplace=True)
            df_day["hour_idx"] = df_day.groupby("OPR_DT").cumcount()
            # print(df_day.head(30))

            fig.add_trace(
                go.Scatter3d(
                    x=df_day["hour_idx"],
                    y=df_day["OPR_DT"],
                    z=df_day["MW"],
                    mode="markers",
                    marker_symbol=marker_types[lmp_type],
                    marker_color=colors[lmp_type],
                    marker_size=2,
                    name=lmp_type,
                )
            )
    # Update layout for better readability
    fig.update_layout(
        title="Price (z-axis) vs. Date (y-axis) vs. Hour idx (x-axis)",
        scene=dict(
            xaxis_title="Hour idx",
            # xaxis_tickformat="%Y-%m-%d %H:%M",
            yaxis_title="Date",
            zaxis_title="Price ($/MW)",
            camera=dict(
                eye=dict(
                    x=3, y=-1.5, z=1.5
                ),  # Adjust x, y, z to change the initial view
                up=dict(x=0, y=0, z=1),  # Sets the upward direction (usually Z-axis)
            ),
        ),
        legend_title="LMP_TYPE",
        hovermode="x unified",
    )

    fig.show()


def draw_avg_price_for_one_day(data_dir):
    # Step 1: Read the CSV files
    lmp = glob(os.path.join(data_dir, "*LMP*.csv"))[0]
    mce = glob(os.path.join(data_dir, "*MCE*.csv"))[0]
    mcc = glob(os.path.join(data_dir, "*MCC*.csv"))[0]
    mcl = glob(os.path.join(data_dir, "*MCL*.csv"))[0]

    df_mce = pd.read_csv(mce)
    df_mcl = pd.read_csv(mcl)
    df_mcc = pd.read_csv(mcc)
    df_lmp = pd.read_csv(lmp)

    # Step 2: Combine the DataFrames
    df_combined = pd.concat([df_mce, df_mcl, df_mcc, df_lmp])

    # Step 3: Convert INTERVALSTARTTIME_GMT to datetime
    df_combined["INTERVALSTARTTIME_GMT"] = pd.to_datetime(
        df_combined["INTERVALSTARTTIME_GMT"]
    )

    # Plotting
    fig, ax = plt.subplots()

    # Define marker types for different LMP_TYPES
    # marker_types = {"MCE": "o", "MCL": "s", "MCC": "^", "LMP": "x"}
    marker_types = {"MCE": ".", "MCL": ".", "MCC": ".", "LMP": "x"}
    colors = {"MCE": "blue", "MCL": "green", "MCC": "red", "LMP": "brown"}
    # Group by LMP_TYPE and plot each group with different markers
    for lmp_type, group in df_combined.groupby("LMP_TYPE"):
        ax.scatter(
            group["INTERVALSTARTTIME_GMT"],
            group["MW"],
            label=lmp_type,
            color=colors[lmp_type],
            marker=marker_types[lmp_type],
            # marker=".",
        )

    # Formatting the plot
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    plt.xticks(rotation=45)
    plt.xlabel("Interval Start Time GMT")
    plt.ylabel("Price ($/MW)")
    plt.title("Price vs. Time")
    plt.legend(title="LMP_TYPE")
    plt.tight_layout()

    # Show plot
    plt.show()


def find_hourly_avg_price(
    LMP_csv_files, MCC_csv_files, MCE_csv_files, MCL_csv_files, output_dir
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    all_csv_files = {
        "LMP": LMP_csv_files,
        "MCC": MCC_csv_files,
        "MCE": MCE_csv_files,
        "MCL": MCL_csv_files,
    }
    len_total_files = sum(len(files) for files in all_csv_files.values())
    with tqdm(total=len_total_files) as p_bar:
        for type, files in all_csv_files.items():
            output_file = os.path.join(output_dir, f"hourly_average_{type}.csv")
            data_frames = []

            # Method 1: concatenate then save
            # for file in files:
            #     df = pd.read_csv(file, low_memory=False)
            #     # df = df.sort_values(by='INTERVALSTARTTIME_GMT')
            #     OPR_DT = df["OPR_DT"].loc[0]
            #     print(OPR_DT)
            #     avg_df = df.groupby('INTERVALSTARTTIME_GMT')['MW'].mean().reset_index()
            #     avg_df["OPR_DT"] = OPR_DT
            #     data_frames.append(avg_df)
            # combined_df = pd.concat(data_frames, ignore_index=True)
            # combined_df.to_csv(output_file, index=False)

            # Method 2: incremental saving
            print(f"Calculating hourly avarge price for {type}...")
            for file in files:  # each file is for one day only
                p_bar.set_description(f"Processing file: {os.path.basename(file)}")
                df = pd.read_csv(file, low_memory=False)

                # exclude row with GRP_TYPE = 'ALL_APNODES'
                df = df[df["GRP_TYPE"] != "ALL_APNODES"]

                OPR_DT = df["OPR_DT"].loc[0]
                LMP_TYPE = df["LMP_TYPE"].loc[0]
                #         print(f"type: {type}, OPR_DT: {OPR_DT}")
                avg_df = df.groupby("INTERVALSTARTTIME_GMT")["MW"].mean().reset_index()
                avg_df["OPR_DT"] = OPR_DT
                avg_df["LMP_TYPE"] = LMP_TYPE
                data_frames.append(avg_df)

                # Write the processed DataFrame to the CSV file
                if not os.path.exists(output_file):
                    avg_df.to_csv(
                        output_file, index=False
                    )  # create new CSV if it doesn't exist
                else:
                    avg_df.to_csv(
                        output_file, mode="a", header=False, index=False
                    )  # if it exists, append to it.
                p_bar.update(1)
    print(f"Successfully calculate average prices for LMP, MCE, MCC and MCL")


def main():
    # data_folder = "data/LMP_one_day"
    data_folder = "data/LMP"
    # data_folder = "data/LMP_one_month"
    # Unzip the files
    zip_data_folder = os.path.join(data_folder, "raw")
    unzip_data_folder = os.path.join(data_folder, "unzip")
    # unzip_files(input_dir=zip_data_folder, output_dir=unzip_data_folder)

    # Prepare LMP, MCC, MCE, MCL files
    current_dir = os.path.dirname(__file__)
    data_dir = os.path.join(current_dir, unzip_data_folder)

    LMP_csv_files = sorted(glob(os.path.join(data_dir, "*_LMP_DAM_LMP_*.csv")))
    MCC_csv_files = sorted(glob(os.path.join(data_dir, "*_LMP_DAM_MCC_*.csv")))
    MCE_csv_files = sorted(glob(os.path.join(data_dir, "*_LMP_DAM_MCE_*.csv")))
    MCL_csv_files = sorted(glob(os.path.join(data_dir, "*_LMP_DAM_MCL_*.csv")))

    print("===LMP===\n")
    print([os.path.basename(file) for file in LMP_csv_files[:5]], "\n")
    print("===MCC===\n")
    print([os.path.basename(file) for file in MCC_csv_files[:5]], "\n")
    print("===MCE===\n")
    print([os.path.basename(file) for file in MCE_csv_files[:5]], "\n")
    print("===MCL===\n")
    print([os.path.basename(file) for file in MCL_csv_files[:5]], "\n")

    # Calculate hourly average price for LMP, MCC, MCE, MCL and save the results into separate csv files
    avg_price_folder = os.path.join(data_folder, "avg_hourly_price")
    # find_hourly_avg_price(
    #     LMP_csv_files=LMP_csv_files,
    #     MCC_csv_files=MCC_csv_files,
    #     MCE_csv_files=MCE_csv_files,
    #     MCL_csv_files=MCL_csv_files,
    #     output_dir=avg_price_folder,
    # )

    # draw_avg_price_for_one_day(avg_price_folder)
    draw_avg_price_for_multiple_days(avg_price_folder)


if __name__ == "__main__":
    sys.exit(main())
