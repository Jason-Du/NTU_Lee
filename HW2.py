import pandas as pd
import csv,os
import numpy as np


def makeDataProcessing(dfData):
	dfDataX=dfData.drop(["education_num","sex"],axis=1)
	list_object_col=[col for col in dfDataX.columns if dfDataX[col].dtypes=="object"]#column name
	list_no_object_col=[col for col in dfDataX.columns if dfDataX[col].dtypes!="object"]

	data_no_col=dfDataX[list_no_object_col]
	data_no_col.insert(2,"sex",(dfData["sex"]==" Male").astype(np.int))

	data_col=dfDataX[list_object_col]
	data_col=pd.get_dummies(data_col)
	dfDataX=data_no_col.join(data_col)
	dfDataX=dfDataX.astype("int64")
	return dfDataX



if __name__ == "__main__":
	dftrain=pd.read_csv(os.path.join(os.path.dirname(__file__),"01-DATA/train.csv"))
	dftest = pd.read_csv(os.path.join(os.path.dirname(__file__), "01-DATA/test.csv"))
	trainsize = len(dftrain)
	testsize = len(dftest)

	dftrainY=dftrain["income"]#取出income_colimn
	dftrainY2=pd.DataFrame((dftrainY==" >50K").astype("int64"),columns=["income"])#labeling
	dftrainY2.to_csv(os.path.join(os.path.dirname(__file__),"Y_train.csv"),index=False) #不保存行索引
	dftrain = dftrain.drop(["income"], axis=1)



	alldata=pd.concat([dftrain,dftest],axis=0,ignore_index=True)
	alldata=makeDataProcessing(dfData=alldata)#labeling

	dftrain=alldata[0:trainsize]
	dftest=alldata[trainsize:trainsize+testsize]

	dftrain.to_csv(os.path.join(os.path.dirname(__file__),"X_train.csv"),index=False)#index不保留
	dftest.to_csv(os.path.join(os.path.dirname(__file__),"X_test.csv"),index=False)
	dftrainY2.to_csv(os.path.join(os.path.dirname(__file__),"Y_train.csv"),index=False)










"""
print()
"""