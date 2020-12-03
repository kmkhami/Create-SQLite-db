#!/usr/bin/env python
import glob
import pandas as pd
import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns

big_df = pd.read_csv("bigPoppa.csv")

data = big_df.drop_duplicates(subset=['Name'])

data["lifetime_exp"].describe() # should be measured in years

def clean_dataset(df):
    df = df.dropna()
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

from scipy import stats

clean_data = clean_dataset(data.drop(["Name"], axis=1))
clean_data = clean_data[np.abs(stats.zscore(clean_data) < 2.5).all(axis=1)] 

from yellowbrick.regressor import ResidualsPlot
from sklearn.ensemble import RandomForestRegressor

def show_residusal(model, train_tup, test_tup):
    resPlot = ResidualsPlot(model)
    resPlot.fit(*train_tup)
    resPlot.score(*test_tup)
    resPlot.show()

drop_columns = ['cum_production', 'cum_production_other', 'lifetime_hyp', 'lifetime_exp','frac stages', "proppant weight (lbs)",  "pump rate (cubic feet/min)", "well length", "oil saturation"]
# Changed this to just look at the values we are supposed to 
x = clean_data.drop(drop_columns, axis = 1)
#x = clean_data[["ppf", "pr", "well length", "frac stages"]]
y = np.log(clean_data['cum_production']) # predict on the ln(cum) hopefully will linearlize it

from sklearn.ensemble import ExtraTreesRegressor
etr = ExtraTreesRegressor(random_state=86)
etr.fit(x, y)

def create_regressor(data:pd.DataFrame, x_names: list, y_name:str):
    ds = clean_dataset(data)
    x = ds[x_names]
    y = ds[y_name]
    
    rf = RandomForestRegressor(random_state=86)
    rf.fit(x, y)
    return rf


vals = ["easting","ppf", "pr", "northing"]
dicto = {}
optim = list(x.drop(vals, axis=1).keys())

for var in optim:
    reg = create_regressor(big_df.drop("Name", axis=1), ["easting", "northing"], var)
    dicto[var] = reg


from tqdm import tqdm

special_ranges = ["northing", "easting"]

def iterate_over(df, var_name, number=1000):
    min_ = min(df[var_name])
    max_ = max(df[var_name])

    step = ((max_ - min_) / number)
    if var_name in special_ranges:
        return tqdm( np.arange(min_, max_, step), desc=var_name)
    else: 
        return np.arange(min_, max_, step)

import copy
from progressbar import ProgressBar
import sqlite3
conn = sqlite3.connect("permutationsDatabase.db")

c = conn.cursor()
schema = """
CREATE TABLE biggest_poppa_table (
    porosity FLOAT(20),
    permeability FLOAT(20),
    Poissonzsratio FLOAT(20),
    YoungzsModulus FLOAT(20),
    watersaturation FLOAT(20),
    pr FLOAT(20),
    northing FLOAT(20),
    ppf FLOAT(20),
    easting FLOAT(20)
);
"""
c.execute(schema)

def iter_constants():
    for north in iterate_over(big_df, "northing", number=25):
        for east in iterate_over(big_df, "easting", number=40):
            values = {}
            values["easting"] = east
            values["northing"] = north
            values["pr"] = 0.0
            values["ppf"] = 0.0
            values["permeability"] = 0.0  #Weird complaining...

            for key in dicto.keys():
                into_key = key.replace("'", "z").replace(" ", "")
                arr = np.array([north, east]).reshape(1, -1)
                values[into_key] = float( "{:20f}".format( dicto[key].predict(arr)[0] ) )
            yield copy.deepcopy(values)

def iter_hypers():
    c = conn.cursor()
    result = c.execute("SELECT rowid from biggest_poppa_table")

    for tup in tqdm(result.fetchall(), desc="Row"):
        rowid = tup[0]
        for prop in tqdm(iterate_over(big_df, "ppf", number=44), desc="ppf"):
            for pump in iterate_over(big_df, "pr", number=44):
                # Predict Our previous values
                values = addAllParams(rowid)

                values["pr"] = pump
                values["ppf"] = prop
                yield copy.deepcopy(values)

lines_to_add_second = ""
lines_to_add_first_pass = """(:porosity, :permeability, :Poissonzsratio, :YoungzsModulus, :watersaturation, :pr, :northing, :ppf, :easting)"""


def addAllParams(rowid):
    values = {}
    c = conn.cursor()
    result = c.execute("SELECT * from biggest_poppa_table WHERE rowid= :rowid", {"rowid": rowid})
    tup = result.fetchall()
    por, perm, Poiss, Young, water, pr, north, ppf, east = tup[0]

    values["porosity"] = por
    values["permeability"] = perm
    values["Poisson's ratio"] = Poiss
    values["Young's Modulus"] = Young
    values["water saturation"] = water
    values["northing"] = north
    values["easting"] = east
    values["rowid"] = rowid

    return values



def f(x:dict):
    c = conn.cursor()
    c.execute(f"""
        INSERT into biggest_poppa_table VALUES {lines_to_add_first_pass}""", x)
    conn.commit()

def finish(x:dict):
    with conn:
        update_task(conn, x)

def update_task(conn, dictonary):
  values = {}
  for key in dictonary.keys():
    into_key = key.replace("'", "z").replace(" ", "")
    values[into_key] = dictonary[key]

  values["YoungzsModulus"] = dictonary["Young's Modulus"]

  sql = f'''
    INSERT into biggest_poppa_table VALUES {lines_to_add_first_pass}
  '''

  cur = conn.cursor()
  cur.execute(sql, values)
  conn.commit()

from multiprocessing import Pool

print("The goods are beging to flow")

# p = Pool(12)
try:
   list(map(f, iter_constants()))
   # p.map(f, iter_hypers())
except Exception as e:
    #conn.close()
    raise e
finally:
    pass
    # p.close()
    # p.join()
import requests
requests.get("https://emergencycontact.herokuapp.com/call/2144028404")

try:
    list(map(finish , iter_hypers()))
finally:
    conn.close()

requests.get("https://emergencycontact.herokuapp.com/call/2144028404")
