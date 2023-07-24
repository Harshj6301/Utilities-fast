{"metadata":{"kernelspec":{"language":"python","display_name":"Python 3","name":"python3"},"language_info":{"pygments_lexer":"ipython3","nbconvert_exporter":"python","version":"3.6.4","file_extension":".py","codemirror_mode":{"name":"ipython","version":3},"name":"python","mimetype":"text/x-python"}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"code","source":"# %% [code] {\"execution\":{\"iopub.status.busy\":\"2023-07-22T07:30:48.123227Z\",\"iopub.execute_input\":\"2023-07-22T07:30:48.123670Z\",\"iopub.status.idle\":\"2023-07-22T07:30:49.731768Z\",\"shell.execute_reply.started\":\"2023-07-22T07:30:48.123630Z\",\"shell.execute_reply\":\"2023-07-22T07:30:49.730393Z\"}}\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nwarnings.filterwarnings('ignore')\n\n# %% [code] {\"execution\":{\"iopub.status.busy\":\"2023-07-22T07:31:15.382798Z\",\"iopub.execute_input\":\"2023-07-22T07:31:15.383538Z\",\"iopub.status.idle\":\"2023-07-22T07:31:15.398436Z\",\"shell.execute_reply.started\":\"2023-07-22T07:31:15.383496Z\",\"shell.execute_reply\":\"2023-07-22T07:31:15.397207Z\"}}\nclass Plot:\n    \n    def __init__(self, dataset, features, target):\n        self.dataset = dataset\n        self.features = features\n        self.target = target\n        \n    def plot_hists(self, features_to_include):\n        plt.figure(figsize=(20,15))\n        for i,feature in enumerate(features_to_include):\n            plt.subplot(3,3,i+1)\n            plt.title(feature)\n            sns.histplot(self.features[feature])\n            plt.tight_layout()\n        plt.show()\n        \n    def plot_scatter(self,features,target):\n        plt.figure(figsize=(18,15))\n        for i,feature in enumerate(features):\n            plt.subplot(3,3,i+1)\n            plt.title(f'{feature} by {target}')\n            sns.scatterplot(data=self.dataset,x=self.features[feature],y=self.target[target])\n            plt.tight_layout()\n        plt.show()\n    \n    def plot_eval(self, actual, predictions):\n        plt.figure(figsize=(20,6))\n        plt.subplot(1,3,1)\n        sns.histplot(x=actual, alpha=0.4,kde=True,label='Actual')\n        sns.histplot(x=predictions, alpha=0.4,kde=True,label='Predictions')\n        plt.title('Distribution of actual and predicted values')\n        plt.legend()\n        plt.subplot(1,3,2)\n        plt.plot(actual, label='Actual')\n        plt.plot(predictions, label='Predictions')\n        plt.title(f'Line plot of {actual} and {predictions}')\n        plt.legend()\n        plt.subplot(1,3,3)\n        sns.scatterplot(x=actual,y=predictions,hue=actual)\n        plt.title(f'Scatter plot of {actual} by {predictions}')\n        plt.legend()\n        plt.suptitle(\"Evaluation Plots\")\n        plt.show()\n\n# %% [code]\n","metadata":{"_uuid":"f11eb8c2-5b3f-4a4b-aa34-05c6a84050d0","_cell_guid":"6393977d-bcfa-45e4-9aec-36d8f0e963e0","collapsed":false,"jupyter":{"outputs_hidden":false},"trusted":true},"execution_count":null,"outputs":[]}]}