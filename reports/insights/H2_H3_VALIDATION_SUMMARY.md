# Main Slide H2/H3 Validation Summary

## Final True/False
- `ml_hypothesis`: **FALSE**
- `h2_spatial_patterns_slide`: **FALSE**
- `h3_seasonal_effects_slide`: **FALSE**

## Why H2/H3 are limited on this sample
- Dataset contains monthly aggregate rows only (`pickup_year`, `pickup_month`, `sample_rows`).
- No zone, borough, airport, weather, or event-level columns are present.

## H2 component status
```csv
component,tested,support,p_value,reason
manhattan_dominance,False,False,,"Missing required columns: borough, trips, zone_id"
airport_predictability,False,False,,"Missing required columns: location_id, pickup_hour, trips"
district_specific_timing,False,False,,"Missing required columns: is_weekend, pickup_hour, trips, zone_type"
```

## H3 component status
```csv
component,tested,support,p_value,p_value_secondary,effect_summer_minus_nonsummer,reason
tourism_season_impact,True,False,1.0,1.0,0.0,
weather_driven_demand,False,False,,,,"Missing weather fields (e.g., precipitation, snowfall, temperature)"
event_based_spikes,False,False,,,,Missing event flags/calendar linkage for venue/event dates
```
