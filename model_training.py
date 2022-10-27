#df=deserialize_json('/content/train_input_cv.json')
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error

#encoder les données catégorielle et cleaning des données
def clean_trainData(data):
   data.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)
   data.dropna()
   one_hot = pd.get_dummies(data[["asset_infos.var4", "asset_infos.var6", "asset_infos.var8","asset_infos.var9"]])
   data= data.drop(["asset_infos.var4", "asset_infos.var6", "asset_infos.var8", "asset_infos.var9"], axis = 1)
   data=data.join(one_hot)
   data["asset_infos.target"] = data["asset_infos.target"] /data["asset_infos.target"].abs().max()
   return(data)
#df=clean_trainData(df)
#---------------------------------------------------------------------##
"""
Définition de x et y ainsi que le split en test et train 

X=df.drop(['asset_infos.target'],axis=1)
y=df["asset_infos.target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled, X_test_scaled = scale(X_train), scale(X_test)
X_train_scaled
X_train_scaled = scale(X_train)
X_test_scaled = scale(X_test)
"""
#-------------------------------------------------------------------##
#Cross validation de modèle
"""
cv = KFold(n_splits=10, shuffle=True, random_state=42)
lr_scores = -1 * cross_val_score(lin_reg, 
                                 X_train_scaled, 
                                 y_train, 
                                 cv=cv, 
                                 scoring='neg_root_mean_squared_error')
lr_scores
"""
#-------------------------------------------------------------------##

#Fonction des trois modèles d'entrainement pour la prediction 
def train_model(data,X,y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  X_train_scaled = scale(X_train)
  X_test_scaled = scale(X_test)
  lin_reg = LinearRegression().fit(X_train_scaled, y_train)
  lasso_reg = LassoCV().fit(X_train_scaled, y_train)
  ridge_reg = RidgeCV().fit(X_train_scaled, y_train)

  
 #Obtenire les different metrics

 def get_parameters(X_train_scaled,y_train):
  # Get R2 score
  lin_reg.score(X_train_scaled, y_train)
  lasso_reg.score(X_train_scaled, y_train)
  # Get RMSE score
  lasso_score_test = mean_squared_error(y_test, y_predicted, squared=False)
  lr_score_test = mean_squared_error(y_test, y_predicted, squared=False) # RMSE instead of MSE
  #get scor training 
  lr_score_train = np.mean(lr_scores)
  
  #enregistrement du modèle dans un fichier 
  def save_model(path):
   model= LinearRegression().fit(X_train_scaled, y_train)
   model.save(path)
