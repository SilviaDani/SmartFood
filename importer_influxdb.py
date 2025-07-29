import pandas as pd
from influxdb_client import InfluxDBClient
from datetime import datetime, timezone

def query(query_type, start_time, end_time, bucket_name="smart_food_bucket_2023-2024-2025"):
    url = "http://localhost:8086"
    token = "9SUJ_bmJB7eSQz5OWS0nPLClLn2TByE-bnh6hyIjTBbC33mZBvZi51LEPWELdgJpoCXPxKWXs0Bx_CvXQOrSiw=="
    org = "smart_food"
    bucket_name = bucket_name
    timeout=1200_000

    client = InfluxDBClient(url=url, token=token, org=org, timeout=timeout)
    query_api = client.query_api()

    if query_type == "xScuola":
        query = f'from(bucket: "{bucket_name}")\
        |> range(start: time(v: "{start_time}"), stop: time(v: "{end_time}"))\
        |> filter(fn: (r) => r._measurement == "school_food_waste")\
        |> filter(fn: (r) => r._field == "porzspreco" or r._field == "presenze")\
        |> map(fn: (r) => ({{\
            r with\
            giorno: time(v: int(v: r._time) - int(v: r._time) % 86400000000000)\
        }}))\
        |> group(columns: ["giorno", "scuola", "_field"]) \
        |> sum()\
        |> pivot(rowKey: ["giorno", "scuola"], columnKey: ["_field"], valueColumn: "_value")\
        |> map(fn: (r) => ({{\
            _time: r.giorno,\
            scuola: r.scuola,\
            _value: if r.presenze == 0.0 then 0.0 else r.porzspreco / r.presenze\
        }}))\
        |> group(columns: ["scuola", "_time"])\
        |> mean()\
        |> yield(name: "result")'

    elif query_type == "xPiattoxScuola":
        query = f'from(bucket: "{bucket_name}")\
        |> range(start: time(v: "{start_time}"), stop: time(v: "{end_time}"))\
        |> filter(fn: (r) => r._measurement == "school_food_waste")\
        |> filter(fn: (r) => r._field == "porzspreco" or r._field == "presenze")\
        |> map(fn: (r) => ({{\
            r with\
            giorno: time(v: int(v: r._time) - int(v: r._time) % 86400000000) \
        }}))\
        |> group(columns: ["giorno", "scuola", "gruppopiatto", "_field"])\
        |> sum()\
        |> pivot(\
            rowKey: ["giorno", "scuola", "gruppopiatto"],\
            columnKey: ["_field"],\
            valueColumn: "_value"\
        )\
        |> map(fn: (r) => ({{\
            _time: r.giorno,\
            scuola: r.scuola,\
            gruppopiatto: r.gruppopiatto,\
            _value: if r.presenze == 0.0 then 0.0 else r.porzspreco / r.presenze\
        }}))'

    elif query_type == "globale":
        query = f'from(bucket: "{bucket_name}")\
        |> range(start: time(v: "{start_time}"), stop: time(v: "{end_time}"))\
        |> filter(fn: (r) => r._measurement == "school_food_waste")\
        |> filter(fn: (r) => r._field == "porzspreco" or r._field == "presenze")\
        |> map(fn: (r) => ({{\
            r with\
            giorno: time(v: int(v: r._time) - int(v: r._time) % 86400000000000)\
        }}))\
        |> group(columns: ["giorno", "_field"])\
        |> sum()\
        |> pivot(rowKey: ["giorno"], columnKey: ["_field"], valueColumn: "_value")\
        |> map(fn: (r) => ({{\
            _time: r.giorno,\
            _value: if r.presenze == 0.0 then 0.0 else r.porzspreco / r.presenze\
        }}))\
        |> yield(name: "result")'

    elif query_type == "xPiattoGlobale":
        query = f'from(bucket: "{bucket_name}")\
        |> range(start: time(v: "{start_time}"), stop: time(v: "{end_time}"))\
        |> filter(fn: (r) => r._measurement == "school_food_waste")\
        |> filter(fn: (r) => r._field == "porzspreco" or r._field == "presenze")\
        |> map(fn: (r) => ({{\
            r with\
            giorno: time(v: int(v: r._time) - int(v: r._time) % 86400000000000)\
        }}))\
        |> group(columns: ["giorno", "gruppopiatto", "_field"])\
        |> sum()\
        |> pivot(\
            rowKey: ["giorno", "gruppopiatto"],\
            columnKey: ["_field"],\
            valueColumn: "_value"\
        )\
        |> map(fn: (r) => ({{\
            _time: r.giorno,\
            gruppopiatto: r.gruppopiatto,\
            _value: if r.presenze == 0.0 then 0.0 else r.porzspreco / r.presenze\
        }}))'

    elif query_type == "xMacrocategoriaGlobale":
        query = f'from(bucket: "{bucket_name}")\
        |> range(start: time(v: "{start_time}"), stop: time(v: "{end_time}"))\
        |> filter(fn: (r) => r._measurement == "school_food_waste")\
        |> filter(fn: (r) => r._field == "porzspreco" or r._field == "presenze")\
        |> map(fn: (r) => ({{\
            r with\
            giorno: time(v: int(v: r._time) - int(v: r._time) % 86400000000)\
        }}))\
        |> group(columns: ["giorno", "macrocategoria", "_field"])\
        |> sum()\
        |> pivot(\
            rowKey: ["giorno", "macrocategoria"],\
            columnKey: ["_field"],\
            valueColumn: "_value"\
        )\
        |> map(fn: (r) => ({{\
            _time: r.giorno,\
            macrocategoria: r.macrocategoria,\
            _value: if r.presenze == 0.0 then 0.0 else r.porzspreco / r.presenze\
        }}))\
        |> yield(name: "result")'

    elif query_type == "xMacrocategoriaxScuola":
        query = f'from(bucket: "{bucket_name}")\
        |> range(start: time(v: "{start_time}"), stop: time(v: "{end_time}"))\
        |> filter(fn: (r) => r._measurement == "school_food_waste")\
        |> filter(fn: (r) => r._field == "porzspreco" or r._field == "presenze")\
        |> map(fn: (r) => ({{\
            r with\
            giorno: time(v: int(v: r._time) - int(v: r._time) % 86400000000)\
        }}))\
        |> group(columns: ["giorno", "scuola", "macrocategoria", "_field"])\
        |> sum()\
        |> pivot(\
            rowKey: ["giorno", "scuola", "macrocategoria"],\
            columnKey: ["_field"],\
            valueColumn: "_value"\
        )\
        |> map(fn: (r) => ({{\
            _time: r.giorno,\
            scuola: r.scuola,\
            macrocategoria: r.macrocategoria,\
            _value: if r.presenze == 0.0 then 0.0 else r.porzspreco / r.presenze\
        }}))\
        |> yield(name: "result")'

    else:
        print("ERROR: query not present in influxDB")
        return pd.DataFrame()

    result = query_api.query(org=org, query=query)

    results = []
    for table in result:
        for record in table.records:
            if query_type == "xScuola":
                results.append({
                    "datetime": record.get_time(),
                    "scuola": record.values.get("scuola"),
                    "value": record.get_value()                
                })
            elif query_type == "xPiattoxScuola":
                results.append({
                    "datetime": record.get_time(),
                    "scuola": record.values.get("scuola"),
                    "gruppopiatto": record.values.get("gruppopiatto"),
                    "value": record.get_value()                
                })
            elif query_type == "globale":
                results.append({
                    "datetime": record.get_time(),
                    "value": record.get_value()
                })
            elif query_type == "xPiattoGlobale":
                results.append({
                    "datetime": record.get_time(),
                    "gruppopiatto": record.values.get("gruppopiatto"),
                    "value": record.get_value()
                })
            elif query_type == "xMacrocategoriaGlobale":
                results.append({
                    "datetime": record.get_time(),
                    "macrocategoria": record.values.get("macrocategoria"),
                    "value": record.get_value()
                })
            elif query_type == "xMacrocategoriaxScuola":
                results.append({
                    "datetime": record.get_time(),
                    "scuola": record.values.get("scuola"),
                    "macrocategoria": record.values.get("macrocategoria"),
                    "value": record.get_value()
                })

    df = pd.DataFrame(results)
    df.sort_values(by="datetime", inplace=True)
    return df
