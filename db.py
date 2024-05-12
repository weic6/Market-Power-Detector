import pandas as pd
from sqlalchemy import create_engine
import os
from utils import printStats

# Replace 'username', 'password', 'localhost', '3306', 'my_data' with your actual MySQL credentials and database details
username = "root"
password = "123456"
database = "siemens_proj"
engine = create_engine(
    f"mysql+mysqlconnector://{username}:{password}@localhost:3306/{database}"
)

directory = "data/PUB_BID/unzip"
filename = "20230313_20230313_PUB_BID_DAM_v3.csv"
filepath = os.path.join(directory, filename)
df = pd.read_csv(filepath)
printStats(df)

# Read the data from CSV
df.to_sql("bids", con=engine, if_exists="append", index=False)
