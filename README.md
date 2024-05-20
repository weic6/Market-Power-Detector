# Capstone_Project
Siemens-sponsored project: strategic bidding identification using pattern recognition techniques

## Overview
Apply clustering on past bidding data in [CAISO](https://www.caiso.com/Pages/default.aspx) and find abnormal patterns in new bid to detect potential market power.

## Key Features
- Web Scraping: Collect public bids from CAISO
- Principle Component Analysis (PCA): Apply PCA on selected features of bidding curves to reduce feature dimention. 
- Database Storage: Store public bids in MySQL database.
- Big Data Processing with Spark (TBD)
- Clustering: Find patterns from history bidding data and infer abnormal behavior in new bids.
- Visualization: Harness Plotly to display interative bidding curves.
