import urllib.parse
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import numpy as np
from xgboost import XGBRegressor
SERVER_NAME = r"Bibhanshu\SQLEXPRESS01"
DB_NAME = "EVChargingDB"
TABLE_NAME = "feature_jpl_sessions_raw"

params = urllib.parse.quote_plus(
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={SERVER_NAME};"
    f"DATABASE={DB_NAME};"
    f"Trusted_Connection=yes;"
)
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", engine)


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
    "idle_connected_formatted"
])
Y = df['kWhDelivered']



X_Encoding = pd.get_dummies(X,columns=["spaceID", "stationID", "clusterID"], drop_first=True)

##Train test split :
X_Encoding_train,X_Encoding_test,Y_train,Y_test = train_test_split(
    X_Encoding,Y,
    test_size=0.3,
    random_state=42
)

##XGBoost:

##for i in range(100, 600, 50):
##    Model_XGB = XGBRegressor(
##    n_estimators=i,
##    learning_rate=0.08,
##    max_depth=4,
##    random_state=42,
##    use_label_encoder=False
##    )
##    Model_XGB.fit(X_Encoding_train,Y_train)
##    Prediction_XGB = Model_XGB.predict(X_Encoding_test)
##    XG_Boost_MSE = mean_squared_error(Y_test, Prediction_XGB)
##    XG_Boost_RMSE = np.sqrt(XG_Boost_MSE)
##    XG_Boost_R2 = r2_score(Y_test, Prediction_XGB)
##    print(f'for {i} RMSE is {XG_Boost_RMSE} and R2 is {XG_Boost_R2}')

##Randomized CV:
Param_grid_XGB = {
    "n_estimators": [150, 250, 350],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [4, 6, 8],
}
random_search_XGB =RandomizedSearchCV(
    estimator=XGBRegressor(
        random_state=42
    ),
    param_distributions = Param_grid_XGB,
    n_iter = 20, ## 3*3*3 = 27 out of 27 combination i want 20
    cv = 5,
    scoring="r2",
    n_jobs=-1
)
random_search_XGB.fit(X_Encoding_train,Y_train)
print(random_search_XGB.best_params_)
print(random_search_XGB.best_score_)


