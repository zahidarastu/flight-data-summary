# Flight Data Summary

## Create directories
`sudo mkdir -p /opt/kairos/flight-data-metrics/{vec_csv_archive,bounds_geojson_archive,turn_time_archive}`
`mkdir -p /home/ubuntu/flight-data-summary/logs`


## Sync to EC2
`rsync -avr ~/repositories/flight-data-summary ubuntu@172.31.4.199:/home/ubuntu/`