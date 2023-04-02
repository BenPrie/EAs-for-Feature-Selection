# Imports as always...
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# The fitness score for a candidate solution.
def fitness_function(candidate_solution):
    '''
    The argument for this function is a candidate solution.
    In context, this is an array of 0s and 1s (which will be cast to Boolean).
    Each entry denotes whether the corresponding feature is selected.

    We know that our dataset has 30 features.
    Hence, we require that the candidate solution is of length 30.
    This is unchanging, so we are not upset with the concrete value.
    '''

    # Handling bad arguments.
    try:
        if len(candidate_solution) != 30:
            raise ValueError("Candidate solution was not of length 30.")

        for elem in candidate_solution:
            if elem not in [0, 1]:
                raise ValueError("An element ({}) of the candidate solution was not in [0, 1].".format(elem))
            
    except:
        raise ValueError("Invalid argument for fitness function.")
    
    # Special case for all 0s.
    if not any(candidate_solution):
        # With no features selected, the model is 0% accurate!
        return 0

    # Read in the data (ignoring the 'Unnamed: 32' feature, and removing the id).
    data = pd.read_csv(r'data.csv').drop(columns=['id', 'Unnamed: 32'])

    # Convert diagnoses into binary output (0 or 1).
    data = pd.get_dummies(data, 'diagnosis').drop(columns=['diagnosis_B']).rename({'diagnosis_M' : 'diagnosis'}, axis=1)

    # Cast the candidate solution to a boolean array.
    selected_features = [bool(x) for x in candidate_solution]

    # Our X and y are the selected features and the diagnosis.
    y = data['diagnosis']
    data = data.drop(columns=['diagnosis'])
    X = data[data.columns[selected_features]]

    # Split the data into training and testing sets (arbitrarily an 80-20% split).
    X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=0)

    # Normlaise the data for numerical stability.
    # We normalise after splitting to prevent data leakage.
    ss_train = StandardScaler()
    X_train = ss_train.fit_transform(X_train)

    ss_test = StandardScaler()
    X_test = ss_test.fit_transform(X_test)

    # Define and train a logistic regression model.
    model = LogisticRegression()

    model.fit(X_train, y_train)

    # Determine the accuracy of the model (and hence the fitness of the candidate solution).
    y_pred = model.predict(X_test)
    return accuracy_score(y_true=y_test, y_pred=y_pred)