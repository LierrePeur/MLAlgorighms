import pandas as pd


def naive_byes(X_known, X_new, y_known):
     X_known = X_known.reset_index().drop("index", axis=1)
     y_known = list(y_known.iloc[:])
     row_count = len(y_known)
     X_conditional_probabilities = {}
     y_unique_values = list(set((y_known)))
     for X_column in X_known.columns:
        X_conditional_probabilities[X_column] = {}
        for y_unique_value in y_unique_values:
            X_conditional_probabilities[X_column][y_unique_value] = {}
     for index, X_column_value in X_known[X_column].items():
        y_current_value = y_known[index]
        if X_column_value in X_conditional_probabilities[X_column][y_current_value]:
            X_conditional_probabilities[X_column][y_current_value][X_column_value] += 1
        else:
            X_conditional_probabilities[X_column][y_current_value][X_column_value] = 1
        for y_unique_value in y_unique_values:
            for X_unique_column_value in X_conditional_probabilities[X_column][y_unique_value]:
                if X_unique_column_value in X_conditional_probabilities[X_column][y_unique_value]:
                    X_conditional_probabilities[X_column][y_unique_value][X_unique_column_value] /= row_count
                else:
                    X_conditional_probabilities[X_column][y_unique_value][X_unique_column_value] = 0
                    y_probabilities = dict(collections.Counter(y_known))
            for y_value in y_probabilities:
                    y_probabilities[y_value] /= row_count
                    y_conditional_probabilities = {}
                    for y_unique_value in y_unique_values:
                        y_conditional_probabilities[y_unique_value] = 1
            for X_column in X_conditional_probabilities:
                X_new_value = list(X_new[X_column])[0]
                if X_new_value in X_conditional_probabilities[X_column][y_unique_value]:
                    y_conditional_probabilities[y_unique_value] *= X_conditional_probabilities[X_column][y_unique_value][X_new_value]
                    y_conditional_probabilities[y_unique_value] *= y_probabilities[y_unique_value]
            predicted_value = ""
            max_probability = 0
            for y_unique_value in y_conditional_probabilities:
                if y_conditional_probabilities[y_unique_value] > max_probability:
                    predicted_value = y_unique_value
                    max_probability = y_conditional_probabilities[y_unique_value]
     return predicted_value


def discretize_dataframe_columns(dataset, columns):
 for column in columns:
    dataset[column] = pd.qcut(dataset[column], 2)
 return dataset


def split_and_evaluate(dataset, target_attribute, test_size, discretize_columns=None):
     if discretize_columns is not None:
        dataset = discretize_dataframe_columns(dataset, discretize_columns)
        X_train, X_test, y_train, y_test = train_test_split(dataset.drop(target_attribute, axis=1), dataset[target_attribute],
        test_size=test_size, random_state=12345)
     test_rows = len(y_test)
     correct_classifications = 0
     false_classifications = 0
     for i in range(test_rows):
        predicted_value = naive_byes(X_train, X_test.iloc[[i]], y_train)
        if predicted_value == list(y_test.iloc[[i]])[0]:
            correct_classifications += 1
        else:
            false_classifications += 1
     return correct_classifications / test_rows
