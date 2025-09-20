import credstash
import gspread
import pygsheets
import pandas as pd
import credstash
import json
import os
from google.oauth2 import service_account

base_dir = '/opt/kairos/flight-data-metrics'
turn_time_archive = os.path.join(base_dir, 'turn_time_archive')

CREDSTASH_GOOGLE_API_SECRET_CONFIG_NAME = 'google_client_secret.json'
sheet_name = 'Flight Data Summary'
tab_name = 'Turn Time'

# Vendor mapping
vendor_dict = {
    'KCSI': ['N735AK','N91726','N759QH','N3397R','N7604F', 'N734VN', 'N9981F', 'N20039', 'N3489R','N3428F','N277J',
                 'N91775', 'N4293Q', 'N7334S','N42295', 'N8409S', 'N4707K', 'N21598', 'N3112S', 'N71220', 'N8409S', 'N9844E', 'N4293Q', 'N735VH'],
    'API': ['N73172', 'N64219', 'N20341', 'N12783', 'N80106', 'N7970G', 'N46655', 'N20823', 'N323SM', 'N3865Q', 'N64542', 'N3803Q','N241RG', 'N775MC', 'N903NS', 'N12456', 'N738CY', 'N19803', 'N80511', 'N64219', 'N1774V'],
    'Envirotech': ['C-FAFW', 'C-FAFB', 'C-GETX'],
    'Aerotec':['LV-IPY','LV-CFL'],
    'Atlantic Corp':['HK-4368', 'I-GIFE'],
    'SAI': ['HK-4701', 'HK-4781', 'PT-EGT'],
}

    
def get_gsheet(CREDSTASH_GOOGLE_API_SECRET_CONFIG_NAME, sheet_name):
    secret = credstash.getSecret(CREDSTASH_GOOGLE_API_SECRET_CONFIG_NAME, region='us-west-2')
    service_account_info = json.loads(secret)
    SCOPES = ('https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive')
    google_credentials = service_account.Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
    gc = pygsheets.authorize(custom_credentials=google_credentials)
    spreadsheet = gc.open(sheet_name)
    return spreadsheet

def get_gsheet_tab(CREDSTASH_GOOGLE_API_SECRET_CONFIG_NAME, sheet_name, tab_name, spreadsheet=None):
    if not spreadsheet:
        spreadsheet = get_gsheet(CREDSTASH_GOOGLE_API_SECRET_CONFIG_NAME, sheet_name)
    wks = spreadsheet.worksheet_by_title(tab_name)
    sheet = wks.get_all_values(returnas='matrix')
    df_sheet = pd.DataFrame(sheet).T.set_index(0).T
    return df_sheet

def update_gsheet_tab(CREDSTASH_GOOGLE_API_SECRET_CONFIG_NAME, sheet_name, tab_name, df_input):
    secret = credstash.getSecret(CREDSTASH_GOOGLE_API_SECRET_CONFIG_NAME, region='us-west-2')
    service_account_info = json.loads(secret)
    SCOPES = ('https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive')
    google_credentials = service_account.Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
    
    # Authorize and open the Google Sheet
    gc = pygsheets.authorize(custom_credentials=google_credentials)
    spreadsheet = gc.open(sheet_name)
    
    # Open the specific tab
    wks = spreadsheet.worksheet_by_title(tab_name)
    
    # Update the entire tab with new data
    wks.set_dataframe(df_input, start='A1', fit=True)


def aggregate_results(output_dir):
    """
    Aggregate intermediate results from CSV files.
    
    Parameters:
        output_dir (str): The directory where the output files are saved.
    
    Returns:
        pd.DataFrame: The aggregated results.
    """
    # List to store DataFrames
    dfs = []
    
    # Iterate through all CSV files in the output directory
    for filename in os.listdir(output_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(output_dir, filename)
            df = pd.read_csv(filepath, converters={'turn_times': eval})  # Convert 'turn_times' back to list
            dfs.append(df)
    
    # Concatenate all DataFrames
    aggregated_df = pd.concat(dfs, ignore_index=True)
    
    return aggregated_df


def add_flight_vendor(df, vendor_dict):
    """
    Add a 'Flight Vendor' column to the DataFrame based on the 'Aircraft Tail Number'.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - vendor_dict (dict): A dictionary where keys are vendor names and values are lists of tail numbers.

    Returns:
    - pd.DataFrame: The DataFrame with an added 'Flight Vendor' column.
    """
    # Initialize a new 'Flight Vendor' column with None
    df['Flight Vendor'] = None  
    
    # Loop through the vendor_dict to assign vendor names based on tail numbers
    for vendor, tail_numbers in vendor_dict.items():
        df.loc[df['Aircraft Tail Number'].isin(tail_numbers), 'Flight Vendor'] = vendor
    
    return df


def rename_columns(df):
    rename_dict = {
        "dataset_id": "Dataset ID",
        "turn_times": "Turn Times",
        "mean_turn_time": "Mean Turn Time",
        "median_turn_time": "Median Turn Time",
        "num_turns": "Num of Turns",
        "collection_time": "Collection Time",
        "non_collection_time": "Non Collection Time",
        'total_turn_time': "Total Turn Time",
        "total_flight_time": "Total Flight Time",
        "collection_start": "Collection Start",
        "collection_end": "Collection End"
    }
    return df.rename(columns=rename_dict)

def add_and_sort_by_date(df):
    df['Date'] = pd.to_datetime(df['Dataset ID'].str.extract(r'(\d{12})')[0], format='%Y%m%d%H%M')
    return df.sort_values(by='Date', ascending=False)


def drop_unwanted_columns(df):
    columns_to_drop = ["Ops Support Engineers", "Backup Ops Support Engineer"]
    return df.drop(columns=columns_to_drop)

def reorder_columns(df):
    final_column_order = [
        "Date", "Dataset ID", "Collection Start", "Collection End", "Collection Time", "Non Collection Time",
        "Total Turn Time", "Total Flight Time", "Num of Turns", "Mean Turn Time",
        "Median Turn Time", "Turn Times", "Operational Base Location",
        "Pilot", "Operator", "Uploader", "Aircraft Tail Number", "Flight Vendor"
    ]
    return df[final_column_order]

def format_flight_data_df(df, vendor_dict):
    df = rename_columns(df)
    df = add_and_sort_by_date(df)
    df = add_flight_vendor(df, vendor_dict)
    df = drop_unwanted_columns(df)
    df = reorder_columns(df)
    return df

df = aggregate_results(turn_time_archive)
formatted_df = format_flight_data_df(df, vendor_dict)
# formatted_df.to_csv('output_pitime.csv')
update_gsheet_tab(CREDSTASH_GOOGLE_API_SECRET_CONFIG_NAME, sheet_name, tab_name, formatted_df)