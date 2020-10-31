import pandas as pd
import csv,os
import numpy as np
import matplotlib.pyplot as plt
def Shuffel(arrayin1):
	array_random_index=np.arange(arrayin1.shape[0])
	np.random.shuffle(array_random_index)
	return arrayin1[array_random_index]


def Normalize(arrayin1,arrayin2):#標準化
	array_t=np.concatenate((arrayin1,arrayin2),axis=0)
	arraymean=np.mean(array_t,axis=0)
	arraysigma=np.std(array_t,axis=0)

	arraynormalize=(array_t-arraymean)/arraysigma

	array_normalize_1=arraynormalize[0:arrayin1.shape[0]]
	array_normalize_2=arraynormalize[arrayin1.shape[0]:]
	return array_normalize_1,array_normalize_2

def Make_Validation_Data(trainarrayX,trainarrayY,percentage):
	length_train=len(trainarrayX)
	length_valid=int(np.floor(length_train*percentage))
	vaildarrayX_output=trainarrayX[0:length_valid]
	vaildarrayY_output=trainarrayY[0:length_valid]
	trainarrayY_output=trainarrayY[length_valid:]
	trainarrayX_output=trainarrayX[length_valid:]
	return trainarrayX_output,trainarrayY_output,vaildarrayX_output,vaildarrayY_output

def get_sigmoid_value(array_x,array_w,arrayb):
	array_predict=array_x.dot(array_w)+arrayb

	array_sigmoid=1/(1+np.exp(-array_predict))
	array_sigmoid=np.clip(array_sigmoid,1e-8,1-(1e-8))
	return array_sigmoid

def get_cross_entrophy_value(sigmoid_value,array_y):
	cross_entrophy_value=  -1 * (         (np.log(sigmoid_value)).T.dot(array_y)  +  (np.log(1-sigmoid_value)).T.dot(1-array_y)     )  /  (len(array_y))
	return cross_entrophy_value


def trainMBGD(arraytrainx,arraytrainy,batchsize,epoch,learningrate):
	arrayW=np.zeros((arraytrainx.shape[1]))
	arrayB=np.zeros(1)
	accuracylist=[]
	Loss_epoch_array=[]
	vaildcost=[]
	TotalLoss = 0
	for epo in range(epoch):
		if epo >0:
			Loss_epoch=TotalLoss/(len(arraytrainx))
			Loss_epoch_array.append(Loss_epoch)
			print("Epoch:{}, Epoch average loss:{} ".format(epo, Loss_epoch))


			vaild_predict=get_sigmoid_value(vaildx,arrayW,arrayB)




			vaildloss=get_cross_entrophy_value(vaild_predict,np.squeeze(vaildy))
			vaildcost.append(vaildloss)
			vaild_predict=np.round(vaild_predict)

			predict_array=(vaild_predict==np.squeeze(vaildy))

			vaild_accuracy=(sum(predict_array))   /    (len(predict_array))
			accuracylist.append(vaild_accuracy)
			print("Validition Accuracy:{} ".format(vaild_accuracy))
			TotalLoss = 0

		arraytrainx=Shuffel(arraytrainx)
		arraytrainy=Shuffel(arraytrainy)



		for iter in range(int(len(arraytrainx)/batchsize)):
			X=arraytrainx[iter*batchsize:batchsize*(iter+1),:]
			Y=arraytrainy[iter*batchsize:batchsize*(iter+1),0]




			array_predict_sigmoid=get_sigmoid_value(X,arrayW,arrayB)


			Loss=get_cross_entrophy_value(array_predict_sigmoid,np.squeeze(Y))
			Loss=Loss*(len(array_predict_sigmoid))
			TotalLoss+=Loss

			# arrayGradientW = np.mean(-1 * X * (np.squeeze(Y) - s).reshape((intBatchSize,1)), axis=0) # need check
			# arrayGradientW = -1 * np.dot(np.transpose(X), (np.squeeze(Y) - array_predict_sigmoid).reshape((batchsize, 1)))
			# arrayW -= learningrate * np.squeeze(arrayGradientW)
			# arrayW += learningrate * (np.mean((np.squeeze(Y) - array_predict_sigmoid).dot(X), axis=0))
			arrayW+=learningrate*((np.squeeze(Y)-array_predict_sigmoid).dot(X))
			arrayB+=learningrate*(np.sum(np.squeeze(Y)-array_predict_sigmoid))
			# arrayGradientB = np.mean(-1 * (np.squeeze(Y) - array_predict_sigmoid))
			# arrayB -= learningrate * arrayGradientB

	plt.plot(np.arange(len(Loss_epoch_array)), Loss_epoch_array, "r--", label="train Cost")
	plt.plot(np.arange(len(vaildcost)), vaildcost, "b--", label="val Cost")
	plt.title("Train Process")
	plt.xlabel("Iteration")
	plt.ylabel("Cost Function (Cross Entropy)")
	plt.legend()
	plt.savefig(os.path.join(os.path.dirname(__file__), "02-Output/TrainProcess"))
	print(vaildcost)
	plt.show()



	return arrayW,arrayB


if __name__ =="__main__":
	trainx=pd.read_csv(os.path.join(os.path.dirname(__file__), "X_train.csv"))
	trainy=pd.read_csv(os.path.join(os.path.dirname(__file__), "Y_train.csv"))
	testx=pd.read_csv(os.path.join(os.path.dirname(__file__), "X_test.csv"))

	arraytrainx=np.array(trainx)
	arraytrainy=np.array(trainy)
	arraytestx=np.array(testx)


	arraytrainx_normalize,arraytestx_normalize=Normalize(arraytrainx,arraytestx)

	arraytrainy=Shuffel(arraytrainy)
	arraytrainx=Shuffel(arraytrainx_normalize)


	trainx,trainy,vaildx,vaildy=Make_Validation_Data(trainarrayX=arraytrainx,trainarrayY=arraytrainy,percentage=0.2)
	trainW,trainB=trainMBGD(arraytrainx=trainx, arraytrainy=trainy, batchsize=32, epoch=50, learningrate=0.001)

	# TEST

	TestY=pd.read_csv(os.path.join(os.path.dirname(__file__),"02-Output/correct_answer.csv"))
	TestY=TestY["label"]
	Test_predict=get_sigmoid_value(array_x=arraytestx_normalize,array_w=trainW,arrayb=trainB)
	Test_predict=np.round(Test_predict)
	dict_predict={"ANS":TestY,"Target":Test_predict}
	csvframe=pd.DataFrame(dict_predict)
	csvframe.to_csv(os.path.join(os.path.dirname(__file__),"Prediction"))


	Test_predict=(Test_predict==TestY)
	Test_accuracy=sum(Test_predict)/len(TestY)
	print("accuracy:{}".format(Test_accuracy))





	""""
		print(arraytrainx.shape)
	print(arraytrainy.shape)
	print(arraytestx.shape)
	print(arraytrainy)
	
	print(arraytestx_normalize)
	print("---------------------")
	
	print(len(arraytrainx))
	print(arraytrainx.shape)	
	
	
	print(arraytrainx.shape)
	print(arraytrainy.shape)
	print(arraytestx.shape)
	print("---------------------")
	print(arraytrainx)
	"""



