import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def plot_hists(data,features_to_include):
    plt.figure(figsize=(20, 15))
    for i, feature in enumerate(features_to_include):
        plt.subplot(3, 3, i + 1)
       # plt.title(data[feature])
        sns.histplot(data[feature])
        plt.tight_layout()
    plt.show()
        
def plot_scatter(data,features, target):
    plt.figure(figsize=(18, 15))
    for i, feature in enumerate(features):
        plt.subplot(3, 3, i + 1)
        plt.title(f'{feature} by {target}')
        sns.scatterplot(data=data, x=feature, y=target)
        plt.tight_layout()
    plt.show()
    
def plot_eval(actual, predictions):
    plt.figure(figsize=(20, 6))
    plt.subplot(1, 3, 1)     
    sns.histplot(x=actual, alpha=0.4, kde=True, label='Actual')
    sns.histplot(x=predictions, alpha=0.4, kde=True, label='Predictions')   
    plt.title('Distribution of actual and predicted values')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(actual, label='Actual')
    plt.plot(predictions, label='Predictions')   
    plt.title(f'Line plot of actual and predicted values')
    plt.legend()     
    plt.subplot(1, 3, 3)
    sns.scatterplot(x=actual, y=predictions, hue=actual)
    plt.title(f'Scatter plot of actual and predicted values')
    plt.legend()
    plt.suptitle("Evaluation Plots")
    plt.show()
