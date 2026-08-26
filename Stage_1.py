#Stage 1 — Energy Demand Prediction (Regression)

#Problem statement: When an EV plugs into a JPL workplace charging station, can we predict how much energy (kWh)
#that session will consume — using only information available at the moment of connection
#(time of day, day of week, station/cluster, whether the driver used the app, their requested energy/miles if provided)?

#Why it matters operationally: A charge point operator needs to forecast electricity draw per session to manage
#grid load, plan station capacity, and avoid overloading circuits — especially at a workplace site where dozens of
#employees plug in around the same morning window. Knowing "this session will likely draw ~7 kWh" the moment
#someone connects is genuinely useful for real-time load balancing.

#Target: kWhDelivered
# we Type: Regression
import urllib.parse
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
SERVER_NAME = r"Bibhanshu\SQLEXPRESS01"
DB_NAME = "EVChargingDB"
TABLE_NAME = "feature_jpl_sessions_imputed"

params = urllib.parse.quote_plus(
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={SERVER_NAME};"
    f"DATABASE={DB_NAME};"
    f"Trusted_Connection=yes;"
)
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", engine)

##print(df.shape)
##print(df.head())


X = df.drop(columns=[
    "sessionID",
    "siteID",
    "userID",
    "connectionTime",
    "disconnectTime",
    "doneChargingTime",
    "requestedDeparture",
    "kWhDelivered",
    "kWhRequested",
    "paymentRequired",
    "WhPerMile",
    "used_app",
    "stayed_after_full",
    "session_duration_minutes",
    "actual_charging_minutes",
    "idle_connected_minutes",
    "availability_estimate_error_minutes",
    "departure_estimate_error_minutes",
    "departure_vs_charge_done_minutes",
    "connection_day_of_week_name",
    "connection_date",
])
Y = df['kWhDelivered']
X_Encoding = pd.get_dummies(X,columns=["spaceID", "stationID", "clusterID"], drop_first=True)
##pd.set_option('display.max_rows', None)
##pd.set_option('display.max_columns', None)
##print(X_Encoding.head())

##Train test split :
X_Encoding_train,X_Encoding_test,Y_train,Y_test = train_test_split(
    X_Encoding,Y,
    test_size=0.3,
    random_state=42
)
Model_Linn_Regg = LinearRegression()
Model_Linn_Regg.fit(X_Encoding_train,Y_train)
Prediction_Linn_Regg = Model_Linn_Regg.predict(X_Encoding_test)
print('Actual : ',Y_test[:10].values)
##print('Predicted : ',Prediction_Linn_Regg[:10])
Linn_Regg_MSE = mean_squared_error(Y_test, Prediction_Linn_Regg)
Linn_Regg_RMSE = np.sqrt(Linn_Regg_MSE)
Linn_Regg_R2 = r2_score(Y_test, Prediction_Linn_Regg)
##print("RMSE : ",Linn_Regg_RMSE,"R2 :",Linn_Regg_R2)## RMSE :  7.4539764970446605 R2 : 0.529283168155761
##print(Y.describe())

##Random Forest :
Model_Random_forest =  RandomForestRegressor(
    n_estimators=500,
    random_state=42
)
Model_Random_forest.fit(X_Encoding_train,Y_train)
Prediction_Random_Forest = Model_Random_forest.predict(X_Encoding_test)
##print("Predicted : ",Prediction_Random_Forest[:10])
Random_Forest_MSE = mean_squared_error(Y_test,Prediction_Random_Forest)
Random_Forest_RMSE = np.sqrt(Random_Forest_MSE)
Random_Forest_R2 = r2_score(Y_test, Prediction_Random_Forest)
##print("RMSE:",Random_Forest_RMSE,"R2:",Random_Forest_R2) ##RMSE: 5.817483643469544 R2: 0.7132826119069606

##Gradient Boosting:
Model_GB = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
Model_GB.fit(X_Encoding_train,Y_train)
Prediction_GB = Model_GB.predict(X_Encoding_test)
##print('Predicted : ',Prediction_GB[:10])
GB_MSE = mean_squared_error(Y_test, Prediction_GB)
GB_RMSE = np.sqrt(GB_MSE)
GB_R2 = r2_score(Y_test, Prediction_GB)
#print("RMSE : ",GB_RMSE,"R2:",GB_R2) ##RMSE :  5.918631846834222 R2: 0.7032256631434857(400,0.05,4)
##RMSE :  5.907887570433887 R2: 0.7043021725060037(300,0.05,5) --Final

##RandomizedSearchCV to get more close values of N ,Max depth Etc ..
#param_grid_Random_forest = {
#    "n_estimators": [200, 300, 400, 600],
#    "min_samples_leaf": [3, 4, 5, 6],
#    "max_depth": [3, 4, 5, 6]
#}

#random_search_Random_forest = RandomizedSearchCV(
#    estimator=RandomForestRegressor(random_state=42),
#    param_distributions=param_grid_Random_forest,
#    n_iter=20,
#    cv=5,
#    scoring="r2",
#    random_state=42,
#    n_jobs=-1
#)
#random_search_Random_forest.fit(X_Encoding_train, Y_train)
#print(random_search_Random_forest.best_params_)
#print(random_search_Random_forest.best_score_)

## Scaling :
##Appling scaling to only Non binary coloumns
# Step 1: define which of your ORIGINAL (pre-encoding) columns are truly numeric
numeric_cols = [
    "milesRequested",
    "minutesAvailable",
    "num_user_inputs",
    "connection_day_of_week_num",
    "connection_hour",
    "connection_month",
]

# is_weekend is already 0/1, same nature as one-hot columns -- leave unscaled
# spaceID, stationID, clusterID become one-hot columns -- leave unscaled

# Step 2: scale ONLY the numeric columns (fit on train, transform both)
Scalar = StandardScaler()
X_train_numeric_scaled = Scalar.fit_transform(X_Encoding_train[numeric_cols])
X_test_numeric_scaled = Scalar.transform(X_Encoding_test[numeric_cols])

# convert back to DataFrame so we can recombine with column names intact
X_train_numeric_scaled = pd.DataFrame(
    X_train_numeric_scaled, columns=numeric_cols, index=X_Encoding_train.index
)
X_test_numeric_scaled = pd.DataFrame(
    X_test_numeric_scaled, columns=numeric_cols, index=X_Encoding_test.index
)

# Step 3: grab the untouched columns (everything NOT in numeric_cols)
# this includes is_weekend + all the one-hot encoded station/cluster/space columns
other_cols = []
for c in X_Encoding_train.columns:
    if c not in numeric_cols:
        other_cols.append(c)

# Step 4: recombine -- scaled numeric columns + untouched dummy columns
X_Train_Final = pd.concat([X_train_numeric_scaled, X_Encoding_train[other_cols]], axis=1)
X_Test_Final = pd.concat([X_test_numeric_scaled, X_Encoding_test[other_cols]], axis=1)


##KNN:
for i in range(5,30):
    model_KNN = KNeighborsRegressor(
        n_neighbors=i,
        weights='uniform',
        algorithm='auto',
        leaf_size=30,
        p=2,
        metric='minkowski',
        metric_params=None,
        n_jobs=-1
    )
    model_KNN.fit(X_Train_Final, Y_train)
    Prediction_KNN = model_KNN.predict(X_Test_Final)
    KNN_MSE = mean_squared_error(Y_test, Prediction_KNN)
    KNN_RMSE = np.sqrt(KNN_MSE)
    KNN_R2 = r2_score(Y_test, Prediction_KNN)
    print(f'when K is {i}','KNN_RMSE : ', KNN_RMSE, 'KNN_R2 : ', KNN_R2)


##KNN Not scaled !

model_KNN = KNeighborsRegressor(
    n_neighbors=5,
    weights='uniform',
    algorithm='auto',
    leaf_size=30,
    p=2,
    metric='minkowski',
    metric_params=None,
    n_jobs=-1
)
model_KNN.fit(X_Encoding_train,Y_train)
Prediction_KNN = model_KNN.predict(X_Encoding_test)
print('Predicted : ',Prediction_KNN[:10])
KNN_MSE = mean_squared_error(Y_test, Prediction_KNN)
KNN_RMSE = np.sqrt(KNN_MSE)
KNN_R2 = r2_score(Y_test, Prediction_KNN)
print('KNN_RMSE : ',KNN_RMSE,'KNN_R2 : ',KNN_R2)


######################################################################################################################
"""
Stage 1 -- Energy Demand Prediction (kWhDelivered) -- Model Comparison
========================================================================

Model                           | RMSE  | R2
--------------------------------|-------|-------
Linear Regression                | 7.454 | 0.529
KNN (unscaled)                   | 6.540 | 0.638
KNN (scaled, best K=28)          | 6.983 | 0.587
Gradient Boosting (tuned)        | 5.908 | 0.704
Random Forest (untuned, WINNER)  | 5.817 | 0.713

Notes:
- Random Forest (n_estimators=500, random_state=42, otherwise default)
  outperformed every other model, including a RandomizedSearchCV-tuned
  version of itself and a tuned Gradient Boosting model.
- KNN required feature scaling (StandardScaler) applied only to true
  numeric columns (milesRequested, minutesAvailable, num_user_inputs,
  connection_day_of_week_num, connection_hour, connection_month) --
  one-hot encoded columns (stationID, clusterID, spaceID) and is_weekend
  were left unscaled since they are already binary.
- Interestingly, scaled KNN performed WORSE than unscaled KNN here --
  likely due to the high cardinality of one-hot encoded station columns
  (52 unique stations) creating a sparse, high-dimensional space that
  hurts KNN's distance-based approach (curse of dimensionality).
- kWhRequested was deliberately excluded from training features (target
  leakage -- it's a human's own guess at the same answer we're
  predicting) but kept aside for a requested-vs-predicted-vs-actual
  comparison.
"""
