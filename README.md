# NYC 311 & Stop-and-Frisk Analysis

A data engineering pipeline analyzing 311 service requests and stop-and-frisk 
data across NYC census tracts (2010–2025).

## Stack
- Python, GeoPandas, pygris, DuckDB (tools will change so far Phase 1)

## Setup 
```bash
pip install -r requirements.txt
python -m pipeline.load_311
python -m pipeline.load_stop_and_frisk
python -m pipeline.load_crime
```
more coming pipelines coming soon...

## Data Sources
- [NYC 311 Service Requests](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9)
- [NYC 311 Service Requests](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-2019/76ig-c548/about_data)
- [NYPD Stop and Frisk](https://www.nyc.gov/site/nypd/stats/reports-analysis/stopfrisk.page)