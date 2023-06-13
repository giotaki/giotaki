import pandas as pd
import numpy as np
import torch

class Dataset():
    def __init__(self, path):
        df = pd.read_csv(path)

        df['dist_km'] = self.haversine_distance(df,'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')
        df['EDTdate'] = pd.to_datetime(df['pickup_datetime'].str[:19]) - pd.Timedelta(hours=4)
        df['Hour'] = df['EDTdate'].dt.hour
        df['AMorPM'] = np.where(df['Hour']<12,'am','pm')
        df['Weekday'] = df['EDTdate'].dt.strftime("%a")

        cat_cols = ['Hour', 'AMorPM', 'Weekday']
        cont_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'passenger_count', 'dist_km']
        y_col = ['fare_amount']  # this column contains the labels

        # Convert our three categorical columns to category dtypes.
        for cat in cat_cols:
            df[cat] = df[cat].astype('category')

        hr = df['Hour'].cat.codes.values
        ampm = df['AMorPM'].cat.codes.values
        wkdy = df['Weekday'].cat.codes.values

        self.cats = np.stack([hr, ampm, wkdy], 1)
        # Convert categorical variables to a tensor
        self.cats = torch.tensor(self.cats, dtype=torch.int64) 
        # Convert continuous variables to a tensor
        self.conts = np.stack([df[col].values for col in cont_cols], 1)
        self.conts = torch.tensor(self.conts, dtype=torch.float)

        # Convert labels to a tensor
        self.y = torch.tensor(df[y_col].values, dtype=torch.float).reshape(-1,1)

        # This will set embedding sizes for Hours, AMvsPM and Weekdays
        cat_szs = [len(df[col].cat.categories) for col in cat_cols]
        self.emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]

    def haversine_distance(self, df, lat1, long1, lat2, long2):
        #Calculates the haversine distance between 2 sets of GPS coordinates in df
        
        r = 6371  # average radius of Earth in kilometers
        
        phi1 = np.radians(df[lat1])
        phi2 = np.radians(df[lat2])
        
        delta_phi = np.radians(df[lat2]-df[lat1])
        delta_lambda = np.radians(df[long2]-df[long1])
        
        a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        d = (r * c) # in kilometers

        return d   