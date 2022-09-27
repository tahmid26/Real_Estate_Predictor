import numpy as np
import pandas as pd
import sys
import part2
import part1
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sns

url = 'https://personal.utdallas.edu/~sxs190355/realEstateDataset.csv'


# The function reads in and normalizes the dataset.
def processDatasets():
    df = pd.read_csv(url,
                     names=["age", "distance_MRT", "num_stores", "price"])
    # print(df.head())

    mm = MinMaxScaler()
    df_transformed = pd.DataFrame(mm.fit_transform(df), columns=df.columns)
    # print(df_transformed.head())

    X = df_transformed[["age", "distance_MRT", "num_stores"]]
    Y = df_transformed["price"]

    # sns.relplot(data=df_transformed, x="age", y="price")
    # sns.relplot(data=df_transformed, x="distance_MRT", y="price")
    # sns.relplot(data=df_transformed, x="num_stores", y="price")
    # sns.displot(df['price'])
    # plt.show()
    # sns.boxplot(df['price'])
    # plt.show()
    # correlation_matrix = df_transformed.corr().round(2)
    # sns.heatmap(data=correlation_matrix, annot=True)
    # plt.show()

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=5)

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }


def main():

    dataset = processDatasets()

    # Part 1: training and testing using own implemntation of gradient descent
    print('\nimplementing gradient descent...')
    model = part1.GradientDescent()
    part1.train(model, dataset)
    p1_Test_MSE = part1.test(model, dataset)
    print('Test RMSE: %.5f' % p1_Test_MSE)

    # Part 2 Training and Testing
    print('\n\nPART2: Training')
    part2.library_reg(dataset['X_train'], dataset['y_train'],
                      dataset['X_test'], dataset['y_test'])


if __name__ == "__main__":
    main()
