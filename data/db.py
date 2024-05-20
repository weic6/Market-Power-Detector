import pandas as pd
from sqlalchemy import create_engine, text
import os
from utils import printStats
from glob import glob
import sys
from dotenv import load_dotenv


def create_bid_table(table_name, engine):
    bid_table_entries = """
        row_id INT AUTO_INCREMENT PRIMARY KEY,
        STARTDATE DATE,

        RESOURCE_TYPE VARCHAR(255),
        RESOURCEBID_SEQ INT,
        SCHEDULINGCOORDINATOR_SEQ INT,
        SCH_BID_TIMEINTERVALSTART DATETIME,
        SCH_BID_TIMEINTERVALSTOP DATETIME,
        SCH_BID_XAXISDATA FLOAT,
        SCH_BID_Y1AXISDATA FLOAT,

        TIMEINTERVALSTART DATETIME,
        TIMEINTERVALEND DATETIME,
        SELFSCHEDMW FLOAT,
        
        STARTTIME DATETIME,
        STOPTIME DATETIME,
        
        SCH_BID_Y2AXISDATA FLOAT,
        
        SCH_BID_CURVETYPE VARCHAR(255),
        MINEOHSTATEOFCHARGE VARCHAR(255),
        MAXEOHSTATEOFCHARGE VARCHAR(255),
        PRODUCTBID_DESC INT,
        PRODUCTBID_MRID INT,
        MARKETPRODUCT_DESC VARCHAR(255),
        MARKETPRODUCTTYPE VARCHAR(255),
        row_repeat INT DEFAULT 0
    """

    sql_drop_table = f"DROP TABLE IF EXISTS {table_name};"
    sql_create_table = f"""
    CREATE TABLE {table_name} (
        {bid_table_entries}
    );
    """

    # sql for creating index
    sql_create_index_sch_bid_start = (
        f"CREATE INDEX idx_sch_bid_start ON {table_name} (SCH_BID_TIMEINTERVALSTART);"
    )
    sql_create_index_start_date = (
        f"CREATE INDEX idx_start_date ON {table_name} (STARTDATE);"
    )
    sql_create_index_resource_id = (
        f"CREATE INDEX idx_resource_id ON {table_name} (RESOURCEBID_SEQ);"
    )
    sql_create_index_sc_id = (
        f"CREATE INDEX idx_sc_seq ON {table_name} (SCHEDULINGCOORDINATOR_SEQ);"
    )

    with engine.connect() as connection:

        # create table
        connection.execute(text(sql_drop_table))
        connection.execute(text(sql_create_table))

        # create index
        connection.execute(text(sql_create_index_sch_bid_start))
        connection.execute(text(sql_create_index_start_date))
        connection.execute(text(sql_create_index_resource_id))
        connection.execute(text(sql_create_index_sc_id))

    print(f"Table {table_name} has been created successfully.")


def populate_table(table_name, csv_files, engine):
    """
    read csv file and populate into table
    """
    from tqdm import tqdm  # add progress bar

    with tqdm(total=len(csv_files)) as p_bar:
        for file in csv_files:
            p_bar.set_description(f"Processing file: {os.path.basename(file)}")
            df = pd.read_csv(file, low_memory=False)

            # filter out only generator and EN
            df = df[
                (df["RESOURCE_TYPE"] == "GENERATOR") & (df["MARKETPRODUCTTYPE"] == "EN")
            ]

            df = df.drop(
                columns=[
                    # "RESOURCE_TYPE",
                    # "MARKETPRODUCTTYPE",
                    "STARTTIME_GMT",
                    "STOPTIME_GMT",
                    # "STARTDATE",
                    "MARKET_RUN_ID",
                    "TIMEINTERVALSTART_GMT",
                    "TIMEINTERVALEND_GMT",
                    "SCH_BID_TIMEINTERVALSTART_GMT",
                    "SCH_BID_TIMEINTERVALSTOP_GMT",
                ],
                axis="columns",
            )

            # convert to datetime
            datetime_columns = [
                "STARTTIME",
                "STOPTIME",
                "STARTTIME_GMT",
                "STOPTIME_GMT",
                "SCH_BID_TIMEINTERVALSTART",
                "SCH_BID_TIMEINTERVALSTOP",
                "SCH_BID_TIMEINTERVALSTART_GMT",
                "SCH_BID_TIMEINTERVALSTOP_GMT",
                "TIMEINTERVALSTART_GMT",
                "TIMEINTERVALEND_GMT",
                "TIMEINTERVALSTART",
                "TIMEINTERVALEND",
            ]

            for col in datetime_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])

            df["STARTDATE"] = pd.to_datetime(df["STARTDATE"]).dt.date

            df.to_sql(name=table_name, con=engine, if_exists="append", index=False)

            p_bar.update(1)

    print(f"Successfully import all csv files into database")


def create_db_engine(db_name="siemens_proj"):
    # load variables
    load_dotenv(dotenv_path="../.env")
    username = os.getenv("DB_USERNAME")
    password = os.getenv("DB_PASSWORD")
    db_url = os.getenv("DB_URL")
    engine = create_engine(
        f"mysql+mysqlconnector://{username}:{password}@{db_url}/{db_name}"
    )
    return engine


def main():
    engine = create_db_engine()

    # create table
    table_name = "bid"
    create_bid_table(table_name=table_name, engine=engine)

    # populate table
    pub_bid_data_folder = os.path.join(os.getcwd(), "/PUB_BID/unzip")
    pub_bid_csv_files = sorted(
        glob(os.path.join(pub_bid_data_folder, "*_PUB_BID_DAM_*.csv"))
    )
    populate_table(table_name=table_name, csv_files=pub_bid_csv_files, engine=engine)


if __name__ == "__main__":
    sys.exit(main())
