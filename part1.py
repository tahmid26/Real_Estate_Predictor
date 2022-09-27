import random
import math
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.metrics import mean_squared_error, r2_score


class GradientDescent(object):

    def calcRMSE(self, err_v):
        mse_m = err_v.transpose().dot(err_v) / (2 * len(err_v))
        return (np.sqrt(mse_m[0, 0]))

    def calcErrorVector(self, x_m, w_v, y_v):
        h_v = x_m.dot(w_v)
        err_v = h_v - y_v
        return err_v

    def calcR2(self, x_m, final_weights, y_v):
        h_v = x_m.dot(final_weights)
        r2 = r2_score(y_v, h_v)
        print('Training R2: ', round(r2, 5))
        return r2

    def updateWeights(self, old_weight, learning_rate, err_v, xi_v):
        sum = 0

        for j, xi in enumerate(xi_v):

            sum += err_v[j].item() * xi
        gradient = sum / len(xi_v)
        new_weight = old_weight - learning_rate * gradient
        return new_weight

    def train(self, training_x_df, training_y_df, descents=1, learning_rate=0.010, delta_weight_threshold=0.00001):
        self.regressands = list(training_x_df)
        self.regressor = list(training_y_df)[0]

        true_v = training_y_df.to_numpy().reshape(-1, 1)
        data_m = training_x_df.to_numpy()
        data_m = np.hstack((np.ones((len(data_m), 1)), data_m))
        best_MSE = math.inf

        for descent in range(descents):

            # initialize weights randomly
            weights_v = (np.random.random_sample(
                (len(data_m[0]), 1)) - (1/2)) / 5
            # calculate error vector
            err_v = self.calcErrorVector(x_m=data_m, w_v=weights_v, y_v=true_v)

            step = 1
            while (step < 50000):

                old_weights_v = np.array(weights_v, copy=True)
                old_err_v = err_v
                old_MSE = self.calcRMSE(err_v)

                new_weights_v = np.zeros((len(weights_v), 1))
                # keep adjusting weights
                for i, xi_v in enumerate(data_m.transpose()):
                    new_weights_v[i] = self.updateWeights(
                        old_weight=weights_v[i], learning_rate=learning_rate, err_v=err_v, xi_v=xi_v)
                weights_v = new_weights_v
                err_v = self.calcErrorVector(
                    x_m=data_m, w_v=weights_v, y_v=true_v)
                new_MSE = self.calcRMSE(err_v)

                delta_weights_v = np.absolute(old_weights_v - new_weights_v)
                if ((delta_weights_v < delta_weight_threshold).all()):
                    iter_MSE = new_MSE
                    break
                elif (new_MSE > old_MSE):
                    learning_rate = learning_rate * 0.99
                    weights_v = old_weights_v
                    err_v = self.calcErrorVector(
                        x_m=data_m, w_v=weights_v, y_v=true_v)
                else:
                    if (step % 100 == 0):
                        print(
                            f'Step {step} \t MSE {new_MSE} \t Weights {np.transpose(weights_v)}')
                    # accelerate learning rate
                    learning_rate = learning_rate * 1.002
                    step += 1

            if (new_MSE < best_MSE):
                best_MSE = new_MSE
                self.weights_v = weights_v

        print('\nP1 -- Training')
        print('The respective attribute weights are: ',
              np.delete(np.transpose(self.weights_v), 0))
        print('The intercept is: ',
              self.weights_v[1])
        self.calcR2(data_m, self.weights_v, true_v)
        print('Training RMSE: %.5f' % (best_MSE))

        return best_MSE

    def test(self, X_test, y_test):
        y_test_vector = y_test.to_numpy().reshape(-1, 1)
        data_m = X_test.to_numpy()
        data_m = np.hstack((np.ones((len(data_m), 1)), data_m))
        predicted_v = data_m.dot(self.weights_v).reshape(-1, 1)
        mse = self.calcRMSE(predicted_v - y_test_vector)
        return predicted_v


def train(model, datasets):
    mse = model.train(
        training_x_df=datasets['X_train'],
        training_y_df=datasets['y_train'],
        descents=1, learning_rate=0.008, delta_weight_threshold=0.00001
    )
    return mse, model.weights_v


def test(model, dataset):
    y_test_predict = model.test(
        dataset['X_test'],
        dataset['y_test']
    )

    print('\nP1 -- Testing')

    print('Test R2: ', round(
        r2_score(dataset['y_test'], y_test_predict), 5))

    return model.calcRMSE(y_test_predict - dataset['y_test'].to_numpy())
