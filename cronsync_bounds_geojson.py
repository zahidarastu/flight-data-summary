import boto3
import os
from datetime import datetime, timedelta

def sync_bounds_geojson(output_dir, start_date_str, end_date_str, bucket='field-upload.kairos.ae'):
    """
    This function syncs the specified S3 bucket and downloads all geojson files in dataset/bounds folders
    that have not been downloaded before and are between a start and end date. It checks if a file has been downloaded
    before by checking if a file with the same name exists in the specified output directory.

    Args:
    - output_dir (str): Path to the directory where the files will be downloaded.
    - start_date_str (str): Files will only be downloaded if their dataset date is on or after this date.
      Date should be in the format 'yyyyMMdd'.
    - end_date_str (str): Files will only be downloaded if their dataset date is on or before this date.
      Date should be in the format 'yyyyMMdd'.
    - bucket (str): S3 bucket name. Default is 'field-upload.kairos.ae'.
    """
    # Set up the S3 client
    s3 = boto3.client('s3')

    # Initialize paginator
    paginator = s3.get_paginator('list_objects_v2')

    # Convert start and end date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y%m%d")
    end_date = datetime.strptime(end_date_str, "%Y%m%d")

    # Iterate through each day in the range
    current_date = start_date
    while current_date <= end_date:
        # Set prefix for pagination
        prefix = current_date.strftime("%Y%m%d")

        # Paginate through objects in the bucket starting from the prefix
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for object in page.get('Contents', []):
                file_name = object['Key']
                
                # Check if the file is in the 'bounds' directory and it's a '.geojson' file
                path_parts = file_name.split("/")
                if len(path_parts) >= 3 and path_parts[1] == 'bounds' and file_name.endswith('.geojson'):
                    # Ignore hidden files (those starting with .) or files that have extra components
                    if path_parts[-1].startswith('.') or len(path_parts[-1].split("-")) > 2:
                        continue

                    print(file_name)
                    dataset_id = path_parts[0]  # the dataset_id is the first part of the object key

                    # Create the download path
                    download_path = os.path.join(output_dir, path_parts[-1])

                    # Check if the file has already been downloaded
                    if not os.path.isfile(download_path):
                        s3.download_file(bucket, file_name, download_path)
                        print(f'-- Downloaded file {file_name} to {download_path}')
                    else:
                        print(f'-- File {file_name} already downloaded')

        # Move to next date
        current_date += timedelta(days=1)

# Directory where the files will be downloaded
base_dir = '/opt/kairos/flight-data-metrics'
bounds_geojson_archive = os.path.join(base_dir, 'bounds_geojson_archive')

# Ensure directory exists
os.makedirs(bounds_geojson_archive, exist_ok=True)

# Get the current date in UTC
current_date_utc = datetime.utcnow().date()

# Set the start date as the day before the current date
start_date = (current_date_utc - timedelta(days=10)).strftime("%Y%m%d")

# Set the end date as the current date
end_date = current_date_utc.strftime("%Y%m%d")

# Call the function with the generated start and end dates:
sync_bounds_geojson(bounds_geojson_archive, start_date, end_date)
