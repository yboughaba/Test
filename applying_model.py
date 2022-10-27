#preparation des données de test
def prepare_testData(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_scaled = scale(X_test)
    
  
