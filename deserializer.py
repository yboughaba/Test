import json
import pandas as pd
import numpy as np
#Vérifier si le json est valide 
def check_json(file_path):
    try:
        json.loads(open(file_path).read())
    except ValueError as err:
        return False
    return True
  #check_json('/content/train_input_cv.json')
  
  #convertire le fichier json en dataframe
  
  def deserialize_json(file_path):
   file = json.loads(open(file_path).read())
   df=pd.json_normalize(file['data_train_assets'])
   return(df)

#deserialize_json('/content/train_input_cv.json')
