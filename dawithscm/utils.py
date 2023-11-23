import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm

def makeDir(path):
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

def distributionValue(df):
    df_temp = df.copy()
    df_temp.dropna(inplace=True)
    a_dict = {}
    a_dict['mean'] = df_temp.mean()
    a_dict['median'] = df_temp.median()
    a_dict['std'] = df_temp.std()
    a_dict['skewness'] = df_temp.skew()
    a_dict['kurtosis'] = df_temp.kurtosis()
    return a_dict

def plotpdf(df, x, axe, bins=15, kde=True, hue=None):
    df_temp = df.copy()
    def annotate(**kws):
        value = distributionValue(df_temp[x])
        bbox = dict(boxstyle="round,pad=0.3", alpha=0.3, fc="white", ec="grey", lw=1)
        annotate_value = (f"mean: {value['mean']:0.2f}\n"
                          f"std: {value['std']:0.2f}\n"
                          f"skew: {value['skewness']:0.2f}\n"
                          f"kurt: {value['kurtosis']:0.2f}")
        #ax = plt.gca()
        if value['skewness'] < 0:
            x_anotate = 0.03
        else:
            x_anotate = 0.70
        axe.annotate(annotate_value, xy=(x_anotate, .10), xycoords=axe.transAxes, fontsize=12, bbox=bbox)

    sns.histplot(df_temp, x=x, kde=kde, ax=axe, bins=bins, hue=hue)
    axe.set_title(x)
    axe.tick_params(axis='x', width=2, length=7, direction='inout', rotation=15, labelsize=12)
    axe.tick_params(axis='y', width=2, length=7, direction='inout', rotation=0, labelsize=12)
    axe.set(xlabel=None)
    axe.set(ylabel=None)
    plt.tight_layout()
    annotate()
    return axe

def plotPie(df, axe, y=None, colors = None, startangle=0):
    if colors is not None:
        colors = np.array(colors)/256
    df.plot(
    kind = 'pie', 
    y = y if y is not None else df.columns[0], 
    title = ' ',
    legend = False,
    ylabel = '',
    autopct = '%1.1f%%',
    colors = colors,
    startangle = startangle,
    fontsize = 16,
    ax = axe
    )

def correlation(x, y, **kws):
    import warnings
    from scipy import stats
    warnings.filterwarnings('ignore')
    
    corr_pearson, p_value_pearson = stats.pearsonr(x, y)
    corr_spearman, p_value_spearman = stats.spearmanr(x, y)
    corr_kendal, p_value_kendal = stats.kendalltau(x, y)
    bbox = dict(boxstyle="round,pad=0.3", alpha=0.3, fc="lightgrey", ec="grey", lw=1)
    annotate_value = (f"Pearson: {corr_pearson:0.2f}\n"
                    f"Spearman: {corr_spearman:0.2f}\n"
                    f"Kendall: {corr_kendal:0.2f}")
    axe = plt.gca()
    result = axe.annotate(annotate_value, xy=(.6, .7), xycoords=axe.transAxes, fontsize=7, bbox=bbox)

    warnings.filterwarnings('always')

    return result

def feature_importance(df, features=None, target=None):
    from sklearn.ensemble import RandomForestRegressor
    if features is None:
        X_train = df.drop(target)
    else:
        X_train = df[features]
    y_train = df[target]
    rf = RandomForestRegressor(n_estimators=150)
    rf.fit(X_train, y_train)
    sort = rf.feature_importances_.argsort()
    plt.barh(X_train.columns[sort], rf.feature_importances_[sort])
    plt.xlabel("Feature Importance")

def convertToPercentage(df, column):
    df = 100. * df[column] / df[column].sum().round(1)
    return df

def plotMetric(df_metric, width = 0.1, figsize=(15, 5)):
    # Find number of columns
    col_len = len(df_metric.columns)

    # Find bars position
    x = np.arange(len(df_metric))
    x_loc = x if col_len%2 == 0 else x - width/2

    # Start plot
    fig, axe = plt.subplots(figsize=figsize)
    fig.suptitle('Comparison between each model.')
    bars = []
    for i in list(range(0, col_len, 1)):
        df_plot = df_metric.iloc[:,i]
        bar = axe.bar(x_loc - width*(int(col_len/2)-i), df_plot.to_list(), width, label=df_plot.name, align='edge')
        bars = bars + bar.get_children()
    axe.set_xticks(ticks=x, labels=df_metric.index, rotation=0)
    axe.grid(alpha=0.5)
    axe.legend()

    # Add counts above the two bar graphs
    for rect in bars:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, 
            height, f'{height:.2f}', 
            ha='center', va='bottom', 
            size=8)

    return fig, axe

def decompositionAnalysis(df, column):

    result = {}
    decompose_result_mult = sm.tsa.seasonal_decompose(df[column], period = 4)
    xaxis = df[column].index

    trend = decompose_result_mult.trend
    seasonal = decompose_result_mult.seasonal
    residual = decompose_result_mult.resid

    trend.index = xaxis
    seasonal.index = xaxis
    residual.index = xaxis

    result['original'] = df[column]
    result['trend'] = trend
    result['seasonal'] = seasonal
    result['residual'] =  residual

    fig, axe = plt.subplots(len(result),1 ,figsize=(13,5), sharex=True)
    fig.suptitle(f'{column}: Monthly Decomposition', size=15)

    for i, [key, value] in enumerate(result.items()):   
        axe[i].plot(value)
        axe[i].set_ylabel(f"{key}", size=13)
        axe[i].xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
        axe[i].yaxis.set_label_coords(-0.05, 0.5)
        axe[i].tick_params(axis='x', which='major', width=0.8, length=5, direction='inout', rotation=90, labelsize=13)
        axe[i].tick_params(axis='y', which='major', width=0.8, length=5, direction='inout', rotation=0, labelsize=10)
        axe[i].tick_params(axis='x', which='minor', width=0.4, length=5, direction='inout', rotation=90, labelsize=13)
        axe[i].tick_params(axis='y', which='minor', width=0.4, length=5, direction='inout', rotation=0, labelsize=10)
        axe[i].grid(which='major', alpha=1)
        axe[i].grid(which='minor', alpha=0.4)
    axe[i].set_xlabel("Datetime (year)", size=13)
    return fig, axe

def plotLine(df, axe, x, y, hue=None):
    sns.lineplot(df, x=x, y=y, hue=hue, ax=axe)
    axe.set_title(f'{y}')
    axe.grid()
    plt.xticks(rotation=50, horizontalalignment='right')

def coolReshape(a_list, row_num, col_num):
    a_list = np.pad(a_list,
        (0, row_num*col_num - len(a_list)), 
        mode='constant', 
        constant_values=np.nan).reshape(row_num,col_num)
    return a_list.tolist()