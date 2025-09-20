import os
import pandas as pd
import numpy as np
import datetime
import logging
import credstash
import json 
import matplotlib.pyplot as plt
from matplotlib import dates
import matplotlib.patches as patches
import ipywidgets as widgets
from IPython.display import display
import re  # for regular expressions
import seaborn as sns
import pytz
from timezonefinder import TimezoneFinder
from geopy.geocoders import Nominatim

# JIRA API v3 imports
import requests
import base64
from typing import Dict, List, Optional, Any
from requests.auth import HTTPBasicAuth


class JiraClient:
    """JIRA API v3 client"""
    
    # Custom field mappings by project and issue type
    REQUIRED_CUSTOM_FIELDS = {
        'OPER': {
            'Survey V2': [
                'Job ID', 'Customer Name', 'Survey Type', 'Survey Period', 'Country', 'State', 'Basin',
                'Collection Start Date', 'Collection End Date', 'Contracted Start Date', 'Contracted End Date',
                'CS Start Date', 'CS End Date', 'Target Start Date', 'Target End Date', 'Track Progress',
                'Contract Type', 'Survey Sensitivity', 'Coverage Count', 'Survey Status', 'Priority in Basin', 'Flight Notes'
            ],
            'Data Collection': [
                'Instrument Identifier', 'Flight Vendor', 'Pilot', 'Operational Base Location',
                'Aircraft Tail Number', 'Uploader', 'Hobbs Time', 'Type of Flight', 'Reason for Flight End',
                'Upload', 'Priority Analysis', 'Ops Support Engineers', 'Backup Ops Support Engineer', 'Operator'
            ]
        },
        'OP': {
            'Ops Issues': [
                'Observer number', 'Prevent collection', 'Help required', 'Ops Issue Tag', 'Pod status',
                'Post troubleshooting notes', 'Time spent troubleshooting (minutes)', 'Collection ticket',
                'Resolution status', 'Ops issue follow up', 'Ongoing ops issue', 'Add to backup-ops manual',
                'Dataset', 'Pod swap'
            ]
        },
        'PIPE': {
            'Data Processing Exception': ['dataset_id', 'submission_id', 'processing_run_timestamp', 'Person-hours required to resolve'],
            'Uplink Exception': ['dataset_id', 'submission_id', 'processing_run_timestamp', 'Person-hours required to resolve']
        }
    }
    
    STANDARD_FIELDS = ['key', 'summary', 'status', 'created', 'updated', 'assignee', 'project', 'issuetype']
    
    def __init__(self, base_url: str = None, email: str = None):
        """Initialize JIRA client"""
        self.base_url = (base_url or 'https://kairosaerospace.atlassian.net').rstrip('/')
        self.email = email or 'jirabot@kairosaerospace.com'
        
        try:
            self.api_token = credstash.getSecret('jirabot_api_token', region='us-west-2')
        except Exception as e:
            raise ValueError(f"JIRA API token is required: {e}")
        
        auth_string = f"{self.email}:{self.api_token}"
        self.headers = {
            'Authorization': f'Basic {base64.b64encode(auth_string.encode()).decode()}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        self._field_mappings = {}
    
    def test_connection(self) -> bool:
        """Test JIRA connection"""
        try:
            url = f'{self.base_url}/rest/api/3/myself'
            response = requests.get(url, headers=self.headers, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def get_field_mappings(self) -> Dict[str, str]:
        """Get custom field name to ID mappings"""
        if self._field_mappings:
            return self._field_mappings
            
        try:
            url = f'{self.base_url}/rest/api/3/field'
            response = requests.get(url, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                mappings = {}
                for field in response.json():
                    if field.get('custom', False) and field.get('id', '').startswith('customfield_'):
                        field_name = field.get('name', '')
                        field_id = field.get('id', '')
                        if field_name and field_id:
                            mappings[field_name] = field_id
                
                self._field_mappings = mappings
                return mappings
            return {}
        except:
            return {}
    
    def search_issues(self, jql: str, max_results: int = 100, next_page_token: str = None) -> Dict[str, Any]:
        """
        Search JIRA issues with optional pagination
        
        Returns:
            Dict with 'issues', 'nextPageToken', and 'isLast' keys
        """
        try:
            url = f'{self.base_url}/rest/api/3/search/jql'
            
            payload = {
                'jql': jql,
                'maxResults': max_results,
                'fields': ['*all']
            }
            
            if next_page_token:
                payload['nextPageToken'] = next_page_token
            
            response = requests.post(url, json=payload, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Search failed: {response.status_code} - {response.text}")
                return {'issues': [], 'isLast': True}
                
        except Exception as e:
            print(f"Search failed: {e}")
            return {'issues': [], 'isLast': True}

    def get_all_issues(self, jql: str, max_results_per_page: int = 100) -> Dict[str, Any]:
        """Get all issues across all pages"""
        all_issues = []
        next_page_token = None
        total_pages = 0
        
        while True:
            result = self.search_issues(jql, max_results_per_page, next_page_token)
            issues = result.get('issues', [])
            
            if not issues:
                break
            
            all_issues.extend(issues)
            total_pages += 1
            print(f"Fetched {len(issues)} issues (total: {len(all_issues)})")
            
            if result.get('isLast', True):
                break
                
            next_page_token = result.get('nextPageToken')
            if not next_page_token:
                break
        
        return {
            'issues': all_issues,
            'total': len(all_issues),
            'pages': total_pages,
            'isLast': True
        }
    
    def get_issue(self, issue_key: str) -> Optional[Dict[str, Any]]:
        """Get single issue by key"""
        try:
            url = f'{self.base_url}/rest/api/3/issue/{issue_key}'
            response = requests.get(url, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def get_field_value(self, issue: Dict[str, Any], field_name: str) -> Any:
        """Get field value from issue"""
        fields = issue.get('fields', {})
        field_mappings = self.get_field_mappings()
        actual_field_name = field_mappings.get(field_name, field_name)
        
        if actual_field_name in fields:
            field_data = fields[actual_field_name]
            
            if field_data is None:
                return None
            elif isinstance(field_data, dict):
                for prop in ['value', 'displayName', 'name']:
                    if prop in field_data:
                        return field_data[prop]
                return field_data
            elif isinstance(field_data, list):
                if not field_data:
                    return []
                if isinstance(field_data[0], dict):
                    for prop in ['value', 'displayName', 'name']:
                        if prop in field_data[0]:
                            return [item.get(prop) for item in field_data if item.get(prop) is not None]
                return field_data
            else:
                return field_data
        return None
    
    def extract_issue_data(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant fields for the issue type"""
        fields = issue.get('fields', {})
        project = fields.get('project', {}).get('key', 'Unknown')
        issue_type = fields.get('issuetype', {}).get('name', 'Unknown')
        
        # Get required custom fields for this issue type
        custom_fields = self.REQUIRED_CUSTOM_FIELDS.get(project, {}).get(issue_type, [])
        
        extracted = {}
        
        # Extract standard fields
        for field in self.STANDARD_FIELDS:
            if field == 'key':
                extracted[field] = issue.get('key', '')
            elif field == 'status':
                status = self.get_field_value(issue, field)
                extracted[field] = status.get('name') if isinstance(status, dict) else str(status) if status else ''
            elif field == 'assignee':
                assignee = self.get_field_value(issue, field)
                extracted[field] = assignee.get('displayName') if isinstance(assignee, dict) else str(assignee) if assignee else ''
            elif field == 'project':
                extracted[field] = project
            elif field == 'issuetype':
                extracted[field] = issue_type
            else:
                value = self.get_field_value(issue, field)
                extracted[field] = str(value) if value is not None else ''
        
        # Extract custom fields
        for field_name in custom_fields:
            field_value = self.get_field_value(issue, field_name)
            if field_value is None:
                extracted[field_name] = ''
            elif isinstance(field_value, list):
                extracted[field_name] = ', '.join(str(x) for x in field_value if x is not None)
            else:
                extracted[field_name] = str(field_value)
        
        return extracted


def get_timezone_from_location(location):
    """
    Gets the timezone for a given location using geopy and timezonefinder.
    
    Args:
        location (str): The name of the location (e.g., "College Station, TX").
    
    Returns:
        str: The timezone string, or None if not found.
    """
    try:
        # Get coordinates for the location
        geolocator = Nominatim(user_agent="flight_data_summary")
        location_details = geolocator.geocode(location)
        
        if location_details:
            latitude = location_details.latitude
            longitude = location_details.longitude
            
            # Get timezone from coordinates
            tf = TimezoneFinder()
            timezone_str = tf.timezone_at(lng=longitude, lat=latitude)
            return timezone_str
        else:
            logging.warning(f"Could not find coordinates for location: {location}")
            return None
            
    except Exception as e:
        logging.warning(f"Error getting timezone for location {location}: {e}")
        return None


def convert_to_operational_timezone(timestamp, operational_base_location):
    """
    Convert a UTC timestamp to the timezone of the operational base location.
    
    Parameters:
        timestamp (pd.Timestamp): UTC timestamp to convert
        operational_base_location (str): Name of the operational base location
    
    Returns:
        pd.Timestamp: Timestamp converted to local timezone, or original timestamp if conversion fails
    """
    if pd.isna(timestamp) or not operational_base_location:
        return timestamp
    
    try:
        # Ensure timestamp is timezone-aware (assume UTC if naive)
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize('UTC')
        elif timestamp.tz != pytz.UTC:
            timestamp = timestamp.tz_convert('UTC')
        
        # Get timezone for operational base location
        timezone_name = get_timezone_from_location(operational_base_location.strip())
        
        if timezone_name:
            local_tz = pytz.timezone(timezone_name)
            return timestamp.tz_convert(local_tz)
        else:
            # If timezone not found, log warning and return UTC timestamp
            logging.warning(f"Timezone not found for operational base location: {operational_base_location}. Using UTC.")
            return timestamp
            
    except Exception as e:
        logging.warning(f"Error converting timestamp to local timezone for {operational_base_location}: {e}")
        return timestamp


def import_flight_track_df(flight_track_archive, input_dataset):
    csv_files = [f for f in os.listdir(flight_track_archive) if f.endswith('.csv')]
    dfs = []

    for file in csv_files:
        dataset_id = file[:17]
        if (dataset_id == input_dataset) and ('fake' not in file):
            # split the filename to extract the instrument and date
            parts = file.split("-")
            date_str = parts[0][:8]
            file_date = datetime.datetime.strptime(date_str, '%Y%m%d')
            instrument = parts[1]

            # Read in CSV and append DataFrame
            df = pd.read_csv(os.path.join(flight_track_archive, file))
            dfs.append(df)  # Append the DataFrame to the list

    concatenated_df = pd.concat(dfs, axis=0, ignore_index=True)  # Concatenate vertically
    concatenated_df.columns = concatenated_df.columns.str.strip()
    concatenated_df = concatenated_df.apply(pd.to_numeric, errors='coerce')
    concatenated_df['dataset_id'] = input_dataset
    
    # Drop rows with 0.0 values in 'latitude' or 'longitude' column (Remove Null Island)
    concatenated_df = concatenated_df[concatenated_df['latitude'] != 0.0]
    concatenated_df = concatenated_df[concatenated_df['longitude'] != 0.0]
    
    # Drop rows with NaN values in 'latitude' or 'longitude' column
    concatenated_df.dropna(subset=['latitude'], inplace=True)
    concatenated_df.dropna(subset=['longitude'], inplace=True)

    return concatenated_df


# Initialize JIRA client using custom JiraClient
jira_client = JiraClient()

def test_jira_connection():
    """Test the JIRA connection and basic functionality"""
    print("Testing JIRA connection...")
    
    # Test connection
    if jira_client.test_connection():
        print("✓ JIRA connection successful")
    else:
        print("✗ JIRA connection failed")
        return False
    
    # Test field mappings
    try:
        field_mappings = jira_client.get_field_mappings()
        print(f"✓ Retrieved {len(field_mappings)} custom field mappings")
        
        # Check for some expected fields
        expected_fields = ['Operational Base Location', 'Pilot', 'Aircraft Tail Number']
        found_fields = [field for field in expected_fields if field in field_mappings]
        print(f"✓ Found {len(found_fields)}/{len(expected_fields)} expected custom fields")
        
    except Exception as e:
        print(f"✗ Error retrieving field mappings: {e}")
        return False
    
    print("✓ JIRA client test completed successfully")
    return True

def get_ticket_data(dataset_id, jira_client, jira_project, jira_issue_type_data_collection):
    """
    This function pulls Jira ticket data by dataset ID. If there is more than one, it errors out.
    :param dataset_id: dataset ID of the flight in question
    :param jira_client: JiraClient object
    :param jira_project: JIRA project name
    :param jira_issue_type_data_collection: JIRA issue type for data collection
    :return: dictionary of ticket info
    """
    try:
        query = 'project = {} AND issuetype = "{}" AND summary ~ "{}" ORDER BY key ASC'.format(
            jira_project,
            jira_issue_type_data_collection,
            dataset_id
        )
        
        # Use the new JiraClient search method
        search_result = jira_client.search_issues(query, max_results=10)
        relevant_issues = search_result.get('issues', [])

        if len(relevant_issues) > 1:
            raise Exception('More than one issue with summary {}'.format(dataset_id))
        elif len(relevant_issues) == 0:
            logging.warning(f"{dataset_id} No Jira issue")
            return None
        else:
            issue = relevant_issues[0]
            ticket_data = {'dataset_id': dataset_id}
            
            # Fields to extract (these should match what's in the JiraClient.REQUIRED_CUSTOM_FIELDS)
            personnel_fields = ['Ops Support Engineers', 'Backup Ops Support Engineer']
            custom_fields = ['Operational Base Location', 'Operator', 'Pilot', 'Uploader', 'Aircraft Tail Number']
            custom_fields.extend(personnel_fields)
            
            for field in custom_fields:
                try:
                    field_value = jira_client.get_field_value(issue, field)
                    if field_value is None:
                        ticket_data[field] = ''
                    elif field in personnel_fields:
                        # For personnel fields, expect displayName format
                        if isinstance(field_value, dict) and 'displayName' in field_value:
                            ticket_data[field] = field_value['displayName']
                        else:
                            ticket_data[field] = str(field_value) if field_value else ''
                    else:
                        ticket_data[field] = str(field_value) if field_value else ''
                except Exception as field_error:
                    logging.warning(f"Error extracting field {field} for dataset {dataset_id}: {str(field_error)}")
                    ticket_data[field] = ''
            
            return ticket_data
            
    except Exception as e:
        logging.error(f"Error retrieving Jira ticket data for dataset_id {dataset_id}: {str(e)}")
        return None

def check_overlap(interval1, interval2):
    """Check if two intervals overlap."""
    return not (interval1[1] < interval2[0] or interval2[1] < interval1[0])

def plot_intervals(bounds_df):
    bounds_df['start_datetime'] = pd.to_datetime(bounds_df['start_time'], unit='ms')
    bounds_df['end_datetime'] = pd.to_datetime(bounds_df['end_time'], unit='ms')
    
    interval_series = list(zip(bounds_df['start_datetime'], bounds_df['end_datetime']))
    start_waypoint_series = bounds_df['start_waypoint']
    
    # Plotting the intervals
    plt.figure(figsize=(12, 15))

    ax = plt.gca()  # Get current axes

    # Adding a patch for overlapping areas
    for i in range(len(interval_series) - 1):
        interval1 = interval_series[i]
        interval2 = interval_series[i + 1]

        if check_overlap(interval1, interval2):
            overlap_start = max(interval1[0], interval2[0])
            overlap_end = min(interval1[1], interval2[1])
            rect = patches.Rectangle((overlap_start, i-0.5), overlap_end - overlap_start, 2,
                                     linewidth=0, edgecolor='r', facecolor='red', alpha=0.3)
            ax.add_patch(rect)

    # Drawing each interval
    for i, interval in enumerate(interval_series):
        plt.plot(interval, [i, i], marker = 'o', color = 'b', markersize = 4, linewidth=2)

    plt.yticks(range(len(interval_series)), start_waypoint_series, rotation=0)
    ax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M:%S'))
    plt.xlabel('Timestamps', fontsize=14)
    plt.ylabel('Start Waypoint', fontsize=14)
    plt.title('Sorted Intervals and Overlaps', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def match_track_to_lines(bounds_archive, dataset_id, df):
    # Load Bounds GeoJSON for Dataset ID
    with open(os.path.join(bounds_archive, dataset_id+'.geojson'), 'r') as f:
        geojson_data = json.load(f)
    
    # Extract properties data from geojson data to DataFrame
    properties_data = [feature['properties'] for feature in geojson_data['features']]
    bounds_df = pd.DataFrame(properties_data)
    
    # Create an interval index from the start_time and end_time
    bounds_df = handle_overlapping_indices(bounds_df)

    # Use the IntervalIndex to match 'pitime' from df to an interval in bounds_df
    intervals = pd.IntervalIndex(bounds_df['interval'])

    # Get the indexer for 'pitime' from 'df' using the intervals
    indexer = intervals.get_indexer(df['pitime'])

    # Initialize a new 'start_waypoint' column with NaNs
    df['start_waypoint'] = np.nan

    # Set 'start_waypoint' values only for those rows where there's a match in bounds_df
    df.loc[indexer != -1, 'start_waypoint'] = bounds_df.iloc[indexer[indexer != -1]]['start_waypoint'].values
    return df
    

def has_overlapping_intervals(bounds_df):
    intervals = pd.IntervalIndex.from_arrays(bounds_df['start_time'], bounds_df['end_time'])

    # Check each interval against all other intervals
    for i, intvl_i in enumerate(intervals):
        for j, intvl_j in enumerate(intervals):
            # Skip comparison with itself
            if i == j:
                continue
            
            # Check if interval i overlaps with interval j
            if intvl_i.overlaps(intvl_j):
                return True
    
    # No overlapping intervals found
    return False

def handle_overlapping_indices(bounds_df):
    # Sort by start_time
    bounds_df = bounds_df.sort_values('start_time').reset_index(drop=True)
    if has_overlapping_intervals(bounds_df):
        #plot_intervals(bounds_df)
        
        # Continue adjusting intervals until no overlaps are detected
        while has_overlapping_intervals(bounds_df):
            # Iterate through intervals, adjusting start_time to avoid overlap
            for i in range(1, len(bounds_df)):
                # If the current interval starts before the previous interval ends
                if bounds_df.at[i, 'start_time'] < bounds_df.at[i-1, 'end_time']:
                    # Adjust the start_time of the current interval
                    bounds_df.at[i, 'start_time'] = bounds_df.at[i-1, 'end_time']
            
            # Drop any indices completely within earlier index
            bounds_df = bounds_df[bounds_df['start_time'] <= bounds_df['end_time']].reset_index(drop=True)
        
        # Create IntervalIndex from the adjusted DataFrame
        bounds_df['interval'] = pd.IntervalIndex.from_arrays(bounds_df['start_time'], bounds_df['end_time'])
        print('- Overlapping Indices Removed')
        #plot_intervals(bounds_df)
    else:
        bounds_df['interval'] = pd.IntervalIndex.from_arrays(bounds_df['start_time'], bounds_df['end_time'])
    
    return bounds_df

def separate_df_by_collection(df):
    # Separate df into collection during flight line and non-collection
    grouped = df.groupby(df['start_waypoint'].notnull())
    
    if True in grouped.groups:
        collection_df = grouped.get_group(True)
    else:
        collection_df = pd.DataFrame(columns=df.columns)  # Empty DataFrame with same columns
    
    if False in grouped.groups:
        non_collection_df = grouped.get_group(False)
    else:
        non_collection_df = pd.DataFrame(columns=df.columns)  # Empty DataFrame with same columns
    
    return collection_df, non_collection_df



def downsample_df(df, sample_num=20):
    df = df.iloc[::sample_num, :]
    return df

def get_candidate_turns(df):
    df = df.copy() 
    df['pi_datetime'] = pd.to_datetime(df['pitime'], unit='ms')
    df['is_turn'] = False
    non_collection = df[df['start_waypoint'].isna()].copy()
    non_collection['candidate_turns'] = df['start_waypoint'].notnull().cumsum()
    return df, non_collection



def extract_polygon_number(waypoint):
    if pd.isna(waypoint):
        return None  # or return a placeholder value if desired
    match = re.search(r'P(\d+)-', waypoint)
    return int(match.group(1)) if match else None

def identify_turns(df, roll_threshold=5, duration_threshold=15, roll_percent=30, parallel_filter=False):
    df = df.copy() 
    df['pi_datetime'] = pd.to_datetime(df['pitime'], unit='ms')
    df['is_turn'] = False
    
    # Extract polygon number from 'start_waypoint' if parallel_filter is True
    if parallel_filter:
        df['polygon'] = df['start_waypoint'].apply(extract_polygon_number)

    non_collection = df[df['start_waypoint'].isna()].copy()
    non_collection['candidate_turns'] = df['start_waypoint'].notnull().cumsum()

    turn_indices = []

    for name, group in non_collection.groupby('candidate_turns'):
        if len(group[np.abs(group['roll']) > roll_threshold]) / len(group) * 100 > roll_percent:
            if (group['pi_datetime'].max() - group['pi_datetime'].min()).total_seconds() < duration_threshold:
                
                # If parallel_filter is True, check if the polygon before and after the turn are the same
                if parallel_filter:
                    before_idx = group.index.min() - 1
                    after_idx = group.index.max() + 1
                    
                    # Check if the polygon before and after the turn are the same
                    if (0 <= before_idx < len(df)) and (0 <= after_idx < len(df)):
                        if before_idx in df.index and after_idx in df.index:
                            if df.loc[before_idx, 'polygon'] == df.loc[after_idx, 'polygon']:
                                turn_indices.extend(group.index.tolist())
                else:
                    turn_indices.extend(group.index.tolist())

    df.loc[turn_indices, 'is_turn'] = True
    return df



def calculate_turn_times(df):
    # Ensure 'pi_datetime' is in datetime format
    df['pi_datetime'] = pd.to_datetime(df['pi_datetime'])
    
    # Identify turn sequences by assigning a unique ID to each sequence of True values in 'is_turn'
    df['turn_id'] = (df['is_turn'] != df['is_turn'].shift()).cumsum()
    
    # Calculate the time duration of each turn by finding the difference between the max and min
    # 'pi_datetime' within each 'turn_id', only for sequences where 'is_turn' is True
    turn_times = df[df['is_turn']].groupby('turn_id')['pi_datetime'].apply(lambda x: x.max() - x.min())
    
    # Convert turn times to seconds for easier interpretation
    if turn_times.empty:
        print("- No turn times identified.")
        return pd.Series(dtype='float64')
    else:
        turn_times = turn_times.dt.total_seconds()
        return turn_times


def calculate_flight_times(df, ticket_data=None):
    df = df.sort_values(by='pi_datetime')

    # Ensure 'pitime' is in datetime format for calculations
    df['pi_datetime'] = pd.to_datetime(df['pitime'], unit='ms')

    # Calculate the time differences between consecutive rows to get durations
    df['delta_time'] = (df['pi_datetime'].shift(-1) - df['pi_datetime']).dt.total_seconds()

    # Calculate the total flight time
    total_flight_time = round(df['delta_time'].sum() / 3600, 4)

    # Calculate the collection time
    collection_time = round(df.loc[df['start_waypoint'].notna(), 'delta_time'].sum() / 3600, 4)

    #NOT ALL TURN TIMES. ONLY NOMINAL TURNS DEPENDING ON FILTER
    # Calculate the turn time
    total_turn_time = round(df.loc[df['is_turn'], 'delta_time'].sum() / 3600, 4)

    # Calculate the non-collection time, 4)
    non_collection_time = round(df.loc[df['start_waypoint'].isna() & ~df['is_turn'], 'delta_time'].sum() / 3600, 4)

    # Calculate collection start and end timestamps
    collection_df = df.loc[df['start_waypoint'].notna()]
    if not collection_df.empty:
        collection_start_utc = collection_df['pi_datetime'].min()
        collection_end_utc = collection_df['pi_datetime'].max()
        
        # Convert to operational base timezone if ticket_data is available
        if ticket_data and 'Operational Base Location' in ticket_data:
            operational_base = ticket_data['Operational Base Location']
            collection_start = convert_to_operational_timezone(collection_start_utc, operational_base)
            collection_end = convert_to_operational_timezone(collection_end_utc, operational_base)
        else:
            collection_start = collection_start_utc
            collection_end = collection_end_utc
    else:
        collection_start = None
        collection_end = None

    # Display the calculated times
    return {
        'collection_time': collection_time,
        'non_collection_time': non_collection_time,
        'total_turn_time': total_turn_time,
        'total_flight_time': total_flight_time,
        'collection_start': collection_start,
        'collection_end': collection_end,
    }


def extract_date_from_filename(filename):
    try:
        # Extract the date string from the filename
        date_str = filename.split('-')[0]
        # Convert the date string to a datetime object
        return datetime.datetime.strptime(date_str, "%Y%m%d%H%M")
    except ValueError:
        print(f"Warning: Filename {filename} does not match expected format. Skipping.")
        

def extract_dataset_id(filename):
    second_hyphen_index = filename.find('-', filename.find('-') + 1)
    # Slice the string to get the dataset_id
    dataset_id = filename[:second_hyphen_index]
    return dataset_id

def save_intermediate_results(dataset_id, turn_times, ticket_data, durations, output_dir):
    """
    Save intermediate results to CSV files.
    
    Parameters:
        dataset_id (str): The ID of the dataset being processed.
        turn_times (pd.Series): The turn times data.
        ticket_data (dict): The ticket data extracted from Jira.
        durations (dict): Calculated durations (collection, non-collection, turn, total flight times).
        output_dir (str): The directory where the output files will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate average turn time and number of turns
    mean_turn_time = turn_times.mean()
    median_turn_time = turn_times.median()
    num_turns = len(turn_times)
    
    # Combine ticket data, calculated values, and durations
    data = {
        'dataset_id': dataset_id,
        'turn_times': turn_times.tolist(),
        'mean_turn_time': mean_turn_time,
        'median_turn_time': median_turn_time,
        'num_turns': num_turns,
        **durations,  # Merge durations into the data dictionary
        **ticket_data,  # Merge ticket_data into the data dictionary
    }
    
    # Convert to DataFrame and save to CSV
    data_df = pd.DataFrame([data])
    filepath = os.path.join(output_dir, f"{dataset_id}_intermediate_results.csv")
    data_df.to_csv(filepath, index=False)
    


def turn_time_all_datasets(flight_track_archive, turn_time_archive):
    dataset_files = os.listdir(flight_track_archive)
    dataset_files.sort(key=lambda filename: (extract_date_from_filename(filename) is None, extract_date_from_filename(filename)), reverse=True)

    # Iterate Through Datasets
    for dataset_file in dataset_files:
        try:
            # Extract dataset_id from dataset_file
            dataset_id = extract_dataset_id(dataset_file)
            print(f'\n{dataset_id}')
            
            # Check if output file already exists
            output_filepath = os.path.join(turn_time_archive, f"{dataset_id}_intermediate_results.csv")
            if os.path.exists(output_filepath):
                print(f"- Output for {dataset_id} already exists. Skipping.")
                continue  # Skip to the next iteration if output file exists

            # Data Import and Preprocessing
            df = import_flight_track_df(flight_track_archive, dataset_id)
            df = match_track_to_lines(bounds_archive, dataset_id, df)
            collection_df, non_collection_df = separate_df_by_collection(df)

            # Jira Data Extraction
            ticket_data = get_ticket_data(dataset_id, jira_client, jira_project="OPER", jira_issue_type_data_collection="Data Collection")

            # Turn Identification and Analysis
            df = identify_turns(df, roll_threshold=5, duration_threshold=300, roll_percent=30, parallel_filter=True)
            turn_times = calculate_turn_times(df)
            #turn_time_distribution(turn_times)
            
            # Calculate durations
            durations = calculate_flight_times(df, ticket_data)

            # Save Intermediate Results
            save_intermediate_results(dataset_id, turn_times, ticket_data, durations, turn_time_archive)
        except Exception as e:
            print(f'- Failed to calculate Turn Times for {dataset_id}. Error: {str(e)}')


if __name__ == "__main__":
    import sys
    
    # Base directory for all flight data metrics
    base_dir = '/opt/kairos/flight-data-metrics'
    flight_track_archive = os.path.join(base_dir, 'vec_csv_archive')
    bounds_archive = os.path.join(base_dir, 'bounds_geojson_archive')
    turn_time_archive = os.path.join(base_dir, 'turn_time_archive')
    
    # Ensure directories exist
    os.makedirs(flight_track_archive, exist_ok=True)
    os.makedirs(bounds_archive, exist_ok=True)
    os.makedirs(turn_time_archive, exist_ok=True)
    
    turn_time_all_datasets(flight_track_archive, turn_time_archive)