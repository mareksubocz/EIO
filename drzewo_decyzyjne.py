import pandas
from math import log2
from ete3 import Tree

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
    return information_gain(df, variable, attribute) / intrinsic_info(df, attribute)


def construct_tree(t: Tree, df, branch):
    distribution = (len(df[df["Survived"] == 0]), len(df[df["Survived"] == 1]))
    if distribution[0] == 0 or distribution[1] == 0 or len(df.columns) == 1:
        t.add_child(name=f" {branch}: {distribution}")
        return t

    ratios = [(gain_ratio(df, "Survived", a), a) for a in df.columns[:-1]]
    chosen_attrib = max(ratios)[1]

    new_name = f" {branch}: {distribution} - - - "+chosen_attrib
    t.add_child(name=new_name)

    classes = df[chosen_attrib].unique()
    for c in classes:
        df_filtered = df[df[chosen_attrib] == c]
        del df_filtered[chosen_attrib]
        construct_tree(t.search_nodes(name=new_name)[0], df_filtered,c)
    return t


if __name__ == "__main__":
    df = pandas.read_csv('titanic-homework.csv')
    continuous_attributes = ["Age"]
    #FIXME: delete Age from here
    ignored_attributes = ["Name", "PassengerId", "Age"]
    # deleting unused columns
    for a in ignored_attributes:
        del df[a]
    t = construct_tree(Tree(), df,'')
    print(t.get_ascii(show_internal=True))
    print('\n*** Nodes are in format \
"branch: (notSurvived, Survived) - - - differentiating_attribute" ***')
