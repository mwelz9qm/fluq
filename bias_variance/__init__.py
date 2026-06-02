import pandas as pd
from sklearn.model_selection import train_test_split

def __init__(
        self, 
        base_architecture, 
        dataset, 
        test_size:float = 0.2,
        random_state:int | None = None,
        save_predictions:bool = True
        ) -> None:
    self.base_architecture = base_architecture
    self.dataset = dataset.copy()

    self.test_size = test_size
    self.random_state = random_state
    self.save_predictions = save_predictions

    # Initialize results and predictions dataframes

    self.results = pd.DataFrame()
    self.predictions = pd.DataFrame() \
        if self.save_predictions else None
    
    # Prepare data

    self.X = dataset.drop('target', axis=1)
    self.y = dataset['target']

    self.n_features = self.X.shape[1]

    # Base train/test split

    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        self.X, self.y, test_size=self.test_size, random_state=self.random_state)
