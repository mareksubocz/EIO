import pandas
import numpy as np
from math import log2
from ete3 import Tree

#global variable
continuous_attributes = ["Age"]

# variable - decision variable in our dataset
def entropy(df, variable: str):
    classes = df[variable].unique()
    result = 0
    for c in classes:
        p = len(df[df[variable] == c]) / len(df)
        if p == 0: continue
        result -= p*log2(p)
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


def intrinsic_info(df, attribute: str):
    # is intrinsic_info just entropy of an attribute?
    return entropy(df, attribute)


def gain_ratio(df, variable: str, attribute: str):
    if information_gain(df, variable, attribute) == 0:
        return 0
    return information_gain(df, variable, attribute) / intrinsic_info(df, attribute)

# finding threshold that gives best gain_ratio
def set_thresholds(df, attribute):
    values = sorted(set(df[attribute]))
    filtered_values = values[3:-3:3]
    max_ratio = 0.0
    value_found = 0
    for v in filtered_values:
        new_df = df.loc[:,("Age","Survived")]
        new_df["Old"] = np.where(new_df["Age"] > v, True, False)
        ratiogain = gain_ratio(new_df, "Survived", attribute)
        if ratiogain > max_ratio:
            max_ratio = ratiogain
            value_found = v
    return max_ratio, value_found
        # ratios.append(gain_ratio(

def construct_tree(t, df_original, branch):
    global continuous_attributes
    from copy import deepcopy
    df = deepcopy(df_original)
    distribution = (len(df[df["Survived"] == 0]), len(df[df["Survived"] == 1]))
    if distribution[0] == 0 or distribution[1] == 0 or len(df.columns) == 1:
        t.add_child(name=f" {branch}: {distribution}")
        return t

    max_ratio = -1
    chosen_attrib = ""
    max_ratio_age = -1
    for a in df.columns[:-1]:
        if a in continuous_attributes:
            tmp, value_found = set_thresholds(df, a)
            if tmp > max_ratio:
                max_ratio = tmp
                chosen_attrib = a
                max_ratio_age = value_found
        else:
            tmp = gain_ratio(df, "Survived", a)
            if tmp > max_ratio:
                max_ratio = tmp
                chosen_attrib = a
    if chosen_attrib in continuous_attributes:
        df[chosen_attrib] = np.where(df[chosen_attrib] > max_ratio_age, True, False)

    new_name = f" {branch}: {distribution} - - - "+chosen_attrib
    if chosen_attrib in continuous_attributes:
        new_name += " > " + str(max_ratio_age)
    t.add_child(name=new_name)

    classes = df[chosen_attrib].unique()
    for c in classes:
        df_filtered = df[df[chosen_attrib] == c]
        del df_filtered[chosen_attrib]
        construct_tree(t.search_nodes(name=new_name)[0], df_filtered,c)
    return t


if __name__ == "__main__":
    df = pandas.read_csv('titanic-homework.csv')
    ignore_attributes = ["Name", "PassengerId"]
    # deleting unused columns
    for a in ignore_attributes:
        del df[a]
    t = construct_tree(Tree(), df,'')
    print(t.get_ascii(show_internal=True))
    print('\n*** Nodes are in format \
"branch: (notSurvived, Survived) - - - differentiating_attribute" ***')
