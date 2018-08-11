from data_loader import *
from transformer import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from future_encoders import OneHotEncoder, ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def make_train_and_test_datasets(dataset):
    # Получаем 5 страт на основании медианного дохода
    dataset["income_cat"] = np.ceil(dataset["median_income"] / 1.5)

    # Объединяем все категории > 5 в категорию 5
    dataset["income_cat"].where(dataset["income_cat"] < 5, 5.0, inplace=True)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = dataset.loc[train_index]
        strat_test_set = dataset.loc[test_index]

    return strat_train_set, strat_test_set


if __name__ == "__main__":
    fetch_housing_data()
    housing = load_housing_data()

    train_set, test_set = make_train_and_test_datasets(housing)

    for set_ in (train_set, test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = train_set.drop("median_house_value", axis=1)
    housing_labels = train_set["median_house_value"].copy()

    num_attribs = list(housing.drop("ocean_proximity", axis=1))
    cat_attribs = ["ocean_proximity"]

    num_pipeline = Pipeline([
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

    housing_prepared = full_pipeline.fit_transform(housing)
    print(housing_prepared)

    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)

    housing_predictions = lin_reg.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    print("RMSE: ", np.sqrt(lin_mse))

