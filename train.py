from data_loader import *
from transformer import *
from sklearn.model_selection import StratifiedShuffleSplit


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

    housing = train_set.drop("median_house_value", axis=1)

    housing_num = housing.drop("ocean_proximity", axis=1)
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    housing_prepared = make_transforms(num_attribs, cat_attribs).fit_transform(housing)
