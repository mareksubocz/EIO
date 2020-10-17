import pandas
import numpy as np

df = pandas.read_csv('titanic-homework.csv')

# variable - decision variable in our dataset
def entropy(df, variable: str):
    classes = df[variable].unique()
    result = 0
    for c in classes:
        p = len(df[df[variable] == c]) / len(df)
        if(p == 0): continue
        result -= p*np.log2(p)
    return result

# attribute - attribute the condition is checked on
def conditional_entropy(df, variable: str, attribute: str):
    classes = df[attribute].unique()
    result = 0
    for c in classes:
        df_filtered = df[df[attribute] == c]
        result += len(df_filtered) / len(df) * entropy(df_filtered, variable)
    return result

def information_gain(df, variable: str, attribute: str):
    return entropy(df, variable) - conditional_entropy(df, variable, attribute)

if __name__ == "__main__":
    print('elo')
    print(information_gain(df, "Survived", "Pclass"))



