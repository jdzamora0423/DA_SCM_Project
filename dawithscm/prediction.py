# A pipeline for prediction
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from .utils import makeDir
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Regression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2
# Classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Time Series
import pmdarima
from tensorflow.keras.callbacks import EarlyStopping
from scalecast.Forecaster import Forecaster
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from scalecast.auxmodels import auto_arima


class prediction():
    def __init__(self, model, X_train, X_test, y_train, y_test):
        self.model = model
        self.X_train = pd.DataFrame(X_train)
        self.X_test = pd.DataFrame(X_test)
        self.y_train = pd.Series(y_train)
        self.y_test = pd.Series(y_test)
        self.features = self.X_train.columns
        self.target = self.y_train.name
        self.pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
            ])
        self.process()
        
    def process(self):
        # Train model
        self.pipe.fit(self.X_train, self.y_train)
        # Predict
        self.y_train_pred = self.pipe.predict(self.X_train)
        self.y_test_pred = self.pipe.predict(self.X_test)
        return self.y_test_pred

    def tuning(self, param_grid, select_best=True, cv=5, n_jobs=1, scoring=None, return_train_score = False, input_parm=None):
        # Make folder if not exist
        path = "./GridSearchCV"
        makeDir(path)

        # Start GridSearhCV if not exist
        if input_parm is not None:
            self.searchCV = joblib.load(f"{input_parm}")
        else:
            self.searchCV = GridSearchCV(
                self.pipe, 
                param_grid, 
                cv = cv,
                n_jobs = n_jobs,
                scoring = scoring,
                return_train_score = return_train_score  # If true, overfitting/underfitting is take into account with a computational power trade-off 
                ).fit(self.X_train, self.y_train)
            # Save result
            time_now = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = type(self.model).__name__
            output_folder = f"{path}/{model_name}_{time_now}"
            makeDir(output_folder)
            joblib.dump(self.searchCV, filename=f"{output_folder}/hyperparameter.pkl")
            with open(f"{output_folder}/hyperparameter.log",'w') as data: 
                data.write(str(param_grid))
                data.write("\n")
                data.write(str(self.searchCV.best_estimator_))
        # Use best hyperparameters
        if select_best:
            self.pipe = self.searchCV.best_estimator_
            self.process()
        return self.searchCV.cv_results_
        

class prediction_regression(prediction):

    def eval(self, eval_dict, name):
        a_dict = {}
        a_dict['train_mse'] = mse(self.y_train_pred, self.y_train)
        a_dict['train_mae'] = mae(self.y_train_pred, self.y_train)
        a_dict['train_r2'] = r2(self.y_train_pred, self.y_train) * 100
        a_dict['test_mse'] = mse(self.y_test_pred, self.y_test)
        a_dict['test_mae'] = mae(self.y_test_pred, self.y_test)
        a_dict['test_r2'] = r2(self.y_test_pred, self.y_test) * 100
        self.evals = a_dict
        eval_dict[name] = a_dict

    def plot(self, figsize=(15,5), x_unit="", y_unit=""):
    
        def annotate(**kws):
            bbox = dict(boxstyle="round,pad=0.3", alpha=0.6, fc="white", ec="grey", lw=1)
            annotate_value = '\n'.join(f"{key}: {eval:0.2f}" for key, eval in self.evals.items())
            axe.annotate(annotate_value, xy=(.05, .85), xycoords=axe.transAxes, bbox=bbox)
        
        def scatter(df_plot, axe, x_unit="", y_unit="", eval_name='train_'):
            fig.suptitle("Scatter plot of actual and prediction.")
            axe.scatter(x=df_plot.index.to_list(), y=df_plot['actual'], label='Actual')
            axe.scatter(x=df_plot.index.to_list(), y=df_plot['predict'], label='Predict')
            axe.set_xlabel(f"index {x_unit}")
            axe.set_ylabel(f"{self.target} {y_unit}")
            axe.set_title(', '.join([f"{key}: {eval:.2e}" for key, eval in self.evals.items() if eval_name in key]))
            axe.legend(loc=1)
            #annotate()

        fig, axe = plt.subplots(1, 2, figsize=figsize)
        df_plot = pd.DataFrame({'actual': self.y_train.values, 'predict': self.y_train_pred})
        scatter(df_plot, axe[0], x_unit, y_unit, eval_name='train_')

        df_plot = pd.DataFrame({'actual': self.y_test.values, 'predict': self.y_test_pred})
        scatter(df_plot, axe[1], x_unit, y_unit, eval_name='test_')

        return fig, axe


# Create a pipeline for prediction

class prediction_classification(prediction):

    def eval(self, eval_dict, name):
        a_dict = {}
        # Accuracy: How offen the model is correct -> (True Positive + True Negative) / Total Predictions
        a_dict['train_accuracy'] = accuracy_score(self.y_train, self.y_train_pred) * 100
        a_dict['test_accuracy'] = accuracy_score(self.y_test, self.y_test_pred) * 100
        # Precision: Of the positives predicted, what percentage is truly positive? -> True Positive / (True Positive + False Positive)
        a_dict['train_precision'] = precision_score(self.y_train, self.y_train_pred) * 100
        a_dict['test_precision'] = precision_score(self.y_test, self.y_test_pred) * 100
        # #Sensitivity: Of all the positive cases, what percentage are predicted positive? -> True Positive / (True Positive + False Negative)
        # a_dict['train_sensitivity'] = recall_score(self.y_train, self.y_train_pred) * 100
        # a_dict['test_sensitivity'] = recall_score(self.y_test, self.y_test_pred) * 100
        # # Specificity: How well the model is at prediciting negative results? -> True Negative / (True Negative + False Positive)
        # a_dict['train_specificity'] = recall_score(self.y_train, self.y_train_pred, pos_label=0) * 100
        # a_dict['test_specificity'] = recall_score(self.y_test, self.y_test_pred, pos_label=0) * 100
        # F1 Score
        a_dict['train_f1'] =  f1_score(self.y_train, self.y_train_pred) * 100
        a_dict['test_f1'] =  f1_score(self.y_test, self.y_test_pred) * 100
        self.evals = a_dict
        eval_dict[name] = a_dict

    def plot(self, labels, figsize=(15, 5)):
        def confusionMatrix(conf_matrix, axe, labels, eval_name='train_'):
            fig.suptitle("Confusion matrix of actual and prediction.")
            eval_matrix_plot = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
            eval_matrix_plot.plot(ax=axe)
            axe.set_title(', '.join([f"{key}: {eval:0.2f}" for key, eval in self.evals.items() if eval_name in key]))

        fig, axe = plt.subplots(1, 2, figsize=figsize)

        eval_matrix = confusion_matrix(self.y_train, self.y_train_pred)
        confusionMatrix(eval_matrix, axe[0], labels, eval_name='train_')

        eval_matrix = confusion_matrix(self.y_test, self.y_test_pred)
        confusionMatrix(eval_matrix, axe[1], labels, eval_name='test_')

        return fig, axe



class predict_timeseries:
    def __init__(self, df, target, time, test_timestep=12, predict_timestep=12):
        self.f = Forecaster(y=df[target], current_dates=df[time])
        self.f.set_test_length(test_timestep)       # 1. 12 observations to test the results
        self.f.generate_future_dates(predict_timestep)
        self.df = df
        self.target = target
        self.test_timestep = test_timestep
        self.predict_timestep = predict_timestep

    def plotTSA(self, lags=12, figsize=(15, 5)):
        self.stationary = self.f.adf_test(full_res=False)
        print(f"{self.target}: {self.f.adf_test(full_res=True)}")
        print(f"{self.target}:Is series stationary? {self.stationary}")

        fig, axe = plt.subplots(1, 2, figsize=figsize)
        plot_pacf(
            self.df[self.target], 
            title=f"Partial Autocorrelation for {self.target}", 
            lags=lags, 
            ax=axe[0])
        plot_acf(
            self.df[self.target], 
            title=f"Autocorrelation for {self.target}", 
            lags=lags, 
            ax=axe[1])
        return fig, axe

    def arima(self, name="auto_arima", period=12):
        auto_arima(self.f, m=period, call_me=name)

    def holtwinters(self, name='holtwinters'):
        self.f.set_estimator('hwes')
        grid = {
            'trend':['add','mul'],
            'seasonal':['add','mul'],
            'damped_trend':[True,False],
            'initialization_method':[None,'estimated','heuristic']
        }
        self.f.ingest_grid(grid)
        self.f.tune()
        self.f.auto_forecast(call_me=name)
    
    def lstm(
        self,
        name='lstm', 
        lags=1, 
        epochs=5, 
        lstm_layer_sizes=(8, ),
        dropout=(0.0,),
        learning_rate=0.001,
        validation_split=0.2,
        batch_size=32,
        plot_loss=True):
        
        self.f.set_estimator('lstm')
        callbacks=EarlyStopping(
            monitor='val_loss',
            patience=5
            )
        self.f.manual_forecast(
            call_me=name,
            lags=lags,
            epochs=epochs,
            lstm_layer_sizes=lstm_layer_sizes,
            validation_split=validation_split,
            callbacks=callbacks,
            learning_rate=learning_rate,
            dropout=dropout,
            batch_size=batch_size,
            plot_loss=plot_loss
            )

    def prophet(self, name='prophet'):
        self.f.set_estimator('prophet') 
        self.f.manual_forecast(call_me=name)

    def plotTrain(self, models='all', figsize=(15, 5)):
        axe = self.f.plot_test_set(models=models, ci=False, figsize=figsize)
        axe.set_ylabel("Value ($)")
        axe.set_title(f"{self.target}: A line plot for model evaluation with test timestep(s) equals to {self.test_timestep}.")
        axe.grid()
        return axe

    def plotFuture(self, models='all', figsize=(15, 5)):
        axe = self.f.plot(models=models, figsize=figsize)
        axe.set_ylabel("Value ($)")
        axe.set_title(f"{self.target}: A line plot forecasting {self.predict_timestep} timestep(s) in the future.")
        axe.grid()
        return axe

    def eval(self):
        df_temp = self.f.export('model_summaries', determine_best_by='LevelTestSetR2')
        df_temp = df_temp[['ModelNickname', 
                           'LevelInSampleRMSE', 'LevelInSampleMAPE', 'LevelInSampleR2',
                           'LevelTestSetRMSE', 'LevelTestSetMAPE', 'LevelTestSetR2'
                           ]]
        df_temp = df_temp.set_index("ModelNickname")
        column = ['LevelTestSetMAPE', 'LevelTestSetR2',
                  'LevelInSampleMAPE', 'LevelInSampleR2']
        df_temp[column] = 100 * df_temp[column]
        return df_temp