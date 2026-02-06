# Data Dictionary

This document describes the datasets and variables used in the NYC Mobility Analysis project.

## Data Sources

### NYC TLC Trip Record Data

**Source**: [NYC Taxi & Limousine Commission](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

**Coverage**: 2009 - Present

**Update Frequency**: Monthly

---

## Yellow Taxi Trip Data

Yellow taxis are the iconic vehicles that can pick up passengers anywhere in NYC.

### Key Fields

| Field | Type | Description |
|-------|------|-------------|
| `VendorID` | Integer | Provider of the trip record (1=Creative Mobile, 2=VeriFone) |
| `tpep_pickup_datetime` | DateTime | Date and time when the meter was engaged |
| `tpep_dropoff_datetime` | DateTime | Date and time when the meter was disengaged |
| `passenger_count` | Integer | Number of passengers (driver-entered value) |
| `trip_distance` | Float | Trip distance in miles |
| `RatecodeID` | Integer | Final rate code (1=Standard, 2=JFK, 3=Newark, 4=Nassau/Westchester, 5=Negotiated, 6=Group) |
| `store_and_fwd_flag` | String | Whether trip record was held in vehicle memory (Y/N) |
| `PULocationID` | Integer | Pickup taxi zone (based on NYC TLC zones) |
| `DOLocationID` | Integer | Dropoff taxi zone |
| `payment_type` | Integer | Payment method (1=Credit card, 2=Cash, 3=No charge, 4=Dispute, 5=Unknown, 6=Voided) |
| `fare_amount` | Float | Time-and-distance fare |
| `extra` | Float | Miscellaneous extras ($0.50, $1) |
| `mta_tax` | Float | MTA tax ($0.50) |
| `tip_amount` | Float | Tip amount (credit cards only) |
| `tolls_amount` | Float | Total tolls paid |
| `improvement_surcharge` | Float | Improvement surcharge ($0.30) |
| `total_amount` | Float | Total amount charged to passengers |
| `congestion_surcharge` | Float | Congestion surcharge (varies) |
| `Airport_fee` | Float | Airport fee ($1.25 for LaGuardia/JFK pickups) |

---

## Green Taxi Trip Data

Green taxis can pick up passengers in outer boroughs and northern Manhattan (above E 96th St and W 110th St).

### Key Fields

Similar structure to Yellow Taxi with prefix `lpep_*` instead of `tpep_*`:

| Field | Type | Description |
|-------|------|-------------|
| `VendorID` | Integer | Trip record provider |
| `lpep_pickup_datetime` | DateTime | Pickup time |
| `lpep_dropoff_datetime` | DateTime | Dropoff time |
| `trip_type` | Integer | 1=Street-hail, 2=Dispatch |
| `ehail_fee` | Float | E-hail fee |

*Other fields same as Yellow Taxi*

---

## For-Hire Vehicle (FHV) Trip Data

FHV includes ride-hailing services (Uber, Lyft, Via) and traditional black car services.

### Key Fields

| Field | Type | Description |
|-------|------|-------------|
| `dispatching_base_num` | String | TLC base license number of dispatching base |
| `pickup_datetime` | DateTime | Pickup time |
| `dropOff_datetime` | DateTime | Dropoff time |
| `PUlocationID` | Integer | Pickup location zone |
| `DOlocationID` | Integer | Dropoff location zone |
| `SR_Flag` | Integer | Shared ride flag (1=Yes, null=No) |
| `Affiliated_base_number` | String | Base number affiliated with trip |

### High Volume FHV (FHVHV)

Additional fields for high-volume services (Uber, Lyft):

| Field | Type | Description |
|-------|------|-------------|
| `hvfhs_license_num` | String | License number (HV0002=Juno, HV0003=Uber, HV0004=Via, HV0005=Lyft) |
| `request_datetime` | DateTime | Time passenger requested the ride |
| `on_scene_datetime` | DateTime | Time driver arrived at pickup |
| `trip_miles` | Float | Trip distance |
| `trip_time` | Integer | Trip time in seconds |
| `base_passenger_fare` | Float | Base passenger fare |
| `tolls` | Float | Tolls |
| `bcf` | Float | Black car fund fee |
| `sales_tax` | Float | Sales tax |
| `congestion_surcharge` | Float | Congestion surcharge |
| `airport_fee` | Float | Airport fee |
| `tips` | Float | Tips |
| `driver_pay` | Float | Driver payment |
| `shared_request_flag` | String | Shared ride flag |
| `shared_match_flag` | String | Whether a shared ride was matched |
| `access_a_ride_flag` | String | Access-a-Ride trip flag |
| `wav_request_flag` | String | Wheelchair-accessible vehicle request |
| `wav_match_flag` | String | WAV trip completion flag |

---

## NYC Taxi Zones

**Source**: NYC TLC Taxi Zones shapefile

**Description**: Taxi zones are based on NYC Department of City Planning's Neighborhood Tabulation Areas (NTAs).

### Zone Information

| Field | Description |
|-------|-------------|
| `LocationID` | Unique zone identifier |
| `Borough` | Manhattan, Queens, Brooklyn, Bronx, Staten Island, EWR |
| `Zone` | Zone name (neighborhood) |
| `service_zone` | Yellow=yellow taxi allowed, Boro=outer borough, Airports |

---

## Processed Features

Features engineered during data processing:

### Temporal Features

| Feature | Type | Description |
|---------|------|-------------|
| `pickup_year` | Integer | Year of pickup |
| `pickup_month` | Integer | Month of pickup (1-12) |
| `pickup_day` | Integer | Day of month |
| `pickup_hour` | Integer | Hour of day (0-23) |
| `pickup_dayofweek` | Integer | Day of week (0=Monday, 6=Sunday) |
| `pickup_weekday` | Boolean | True if weekday, False if weekend |
| `time_of_day` | Categorical | night/morning/afternoon/evening |
| `is_weekend` | Boolean | True if Saturday or Sunday |
| `is_holiday` | Boolean | True if federal holiday |
| `season` | Categorical | spring/summer/fall/winter |

### Trip Features

| Feature | Type | Description |
|---------|------|-------------|
| `trip_duration_seconds` | Integer | Trip duration in seconds |
| `trip_duration_minutes` | Float | Trip duration in minutes |
| `speed_mph` | Float | Average speed (distance/duration) |
| `fare_per_mile` | Float | Fare amount per mile |
| `fare_per_minute` | Float | Fare amount per minute |

### Spatial Features

| Feature | Type | Description |
|---------|------|-------------|
| `pickup_borough` | Categorical | Pickup borough |
| `dropoff_borough` | Categorical | Dropoff borough |
| `is_same_borough` | Boolean | Pickup and dropoff in same borough |
| `is_manhattan_trip` | Boolean | Either pickup or dropoff in Manhattan |
| `is_airport_trip` | Boolean | Trip to/from airport |

### Service Type

| Feature | Type | Description |
|---------|------|-------------|
| `vehicle_type` | Categorical | yellow/green/fhv/fhvhv |
| `is_rideshare` | Boolean | True if fhvhv |

---

## Data Quality Notes

### Known Issues

1. **Null Values**: Some fields have high percentages of null values
   - `passenger_count`: Often null or zero
   - `congestion_surcharge`: Only available after certain dates
   
2. **Outliers**: Some records contain unrealistic values
   - Extremely long trip durations (>24 hours)
   - Zero or negative fares
   - Impossible speeds (>100 mph in city traffic)

3. **Data Availability**:
   - FHVHV data only available from 2019 onwards
   - Column names and structure changed over time
   - Some months may have missing data

### Cleaning Rules Applied

1. Remove trips with negative or zero duration
2. Remove trips with duration > 24 hours
3. Remove trips with distance > 100 miles
4. Remove trips with fare < 0 or > $1000
5. Remove trips with passenger_count < 0 or > 8
6. Convert all datetime columns to pandas datetime
7. Standardize column names across datasets

---

## References

- [NYC TLC Trip Record Data Dictionary](https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf)
- [NYC Taxi Zones](https://data.cityofnewyork.us/Transportation/NYC-Taxi-Zones/d3c5-ddgc)
