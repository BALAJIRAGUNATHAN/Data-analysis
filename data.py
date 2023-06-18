import pandas as pd
from sklearn.ensemble import IsolationForest
data = pd.read_csv('cyclone_data.csv')
features = ['Cyclone_Inlet_Gas_Temp', 'Cyclone_Material_Temp', 'Cyclone_Outlet_Gas_draft', 'Cyclone_cone_draft', 'Cyclone_Gas_Outlet_Temp', 'Cyclone_Inlet_Draft']
X = data[features]
model = IsolationForest(contamination='auto', random_state=42)

model.fit(X)
anomaly_scores = model.decision_function(X)
anomalies = model.predict(X)
data['Anomaly_Score'] = anomaly_scores
data['Is_Anomaly'] = anomalies
abnormal_periods = data[data['Is_Anomaly'] == -1]
print(abnormal_periods['time'])
