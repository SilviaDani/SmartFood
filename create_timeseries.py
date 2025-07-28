from datetime import timedelta, datetime
from mockseries.trend import LinearTrend
from mockseries.seasonality import SinusoidalSeasonality
from mockseries.noise import RedNoise
from mockseries.utils import datetime_range, plot_timeseries, write_csv
import random
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate a timeseries in 2024 year with daily values in csv file.')
parser.add_argument('--outliers', required=True, type=int, help='Numbers of outliers.')
parser.add_argument('--csv_name', required=True, type=str, help='Name of the csv where to save the timeseries.')
args = parser.parse_args()

#Create a timeseries generator
trend = LinearTrend(coefficient=0, time_unit=timedelta(days=7), flat_base=50)
seasonality = SinusoidalSeasonality(amplitude=2, period=timedelta(days=7)) \
              + SinusoidalSeasonality(amplitude=4, period=timedelta(days=30))
noise = RedNoise(mean=0, std=3, correlation=0.5)

timeseries = trend + seasonality + noise

#Preview
# preview on minute, hour, day, month, year 
#timeseries.preview_month()

#Generate values
time_points = datetime_range(
    granularity=timedelta(days=1),
    start_time=datetime(2024, 1, 1),
    end_time=datetime(2024, 12, 31),
)
ts_values = timeseries.generate(time_points=time_points)

#sample n random outliers

random_sample = random.sample(range(1, len(ts_values)), args.outliers) 
for i in random_sample:
    if ts_values[i] + 20 <= 100:
        ts_values[i] += 20
    else:
        ts_values[i] = 100


#Plot or write to csv
plot_timeseries(time_points, ts_values)
write_csv(time_points, ts_values, "./simulated_data/" + args.csv_name + ".csv")