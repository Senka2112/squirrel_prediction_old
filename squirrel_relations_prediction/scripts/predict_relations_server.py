#!/usr/bin/python

from squirrel_prediction_msgs.srv import *
import rospy

import numpy as np

import mvm_mmmvr_lib.mvm_mvm_cls as mvm_mvm_cls
import mvm_mmmvr_lib.mmr_setparams as mmr_setparams
import mvm_mmmvr_lib.load_data as load_data
import mvm_mmmvr_lib.mvm_validation_cls as mvm_validation_cls

rospy.init_node('relations_prediction')

def test_mvm_main(workmode):

  params=mmr_setparams.cls_params()

  xdatacls=mvm_mvm_cls.cls_mvm()
  nfold=xdatacls.nfold
  
  nfold0=nfold    ## n-fold cross validation
  npar = 1
  
  for ipar in range(npar):

    Y0=np.array([0,1,2,3])
    
    ctables=load_data.cls_label_files()  ## data loading object
    ctables.irowcol=xdatacls.rowcol  ## set the row-col or col-row processing
    
    (xdata,nrow2,ncol2,ifixtrain,ifixtest)=ctables.load_twofiles()
    xdatacls.categorymax=xdata[2].max()-xdata[2].min()+1                    
    xdatacls.load_data(xdata,[],xdatacls.categorymax, \
                     int(nrow2),int(ncol2),Y0)
    xdatacls.ifixtrain=ifixtrain
    xdatacls.ifixtest=ifixtest

    xdatacls.YKernel.ymax=1 # it will be recomputed in mvm_ranges
    xdatacls.YKernel.ymin=0
    xdatacls.YKernel.yrange=100 # it will be recomputed in classcol_ranges
    xdatacls.YKernel.ystep=(xdatacls.YKernel.ymax-xdatacls.YKernel.ymin) \
                            /xdatacls.YKernel.yrange
 
    xdatacls.prepare_repetition_folding(init_train_size=100)
    nrepeat0=xdatacls.nrepeat0
    nfold0=xdatacls.nfold0

    # ----------------------------------------------------------------------

    nval=max(xdatacls.YKernel.valrange)+1
    xconfusion3=np.zeros((nrepeat0,nfold0,xdatacls.YKernel.ndim,nval,nval))

    ireport=0
    for irepeat in range(nrepeat0):

      xdatacls.nfold0=xdatacls.nfold
      xdatacls.prepare_repetition_training()
   
      for ifold in range(nfold0):

        xdatacls.prepare_fold_training(ifold)

    # validation to choose the best parameters
        xdatacls.set_validation()
        cvalidation=mvm_validation_cls.cls_mvm_validation()
        cvalidation.validation_rkernel=xdatacls.XKernel[0].title
        best_param=cvalidation.mvm_validation(xdatacls)
        
        #print('Parameters:',best_param.c,best_param.d, \
        #      best_param.par1,best_param.par2)

    # training with the best parameters
        rospy.loginfo('(squirrel prediction) Training.')
        cOptDual= xdatacls.mvm_train()
     
    # check the test accuracy
        rospy.loginfo('(squirrel prediction) Predicting.')
        cPredict=xdatacls.mvm_test()
       
      ctables.export_test_prediction('',xdatacls,cPredict.Zrow)


  print('(squirrel prediction) Prediction done.')    
  
  return -1

def callback(data):
    #input
    #data.inputFile data.outputFile data.initilization
    test_mvm_main(0)
    #RecommendRelationsResponse.SUCCESS = uint8(1)
    #RecommendRelationsResponse.FAILURE = uint8(0)
    #RecommendRelationsResponse.result = uint8(1)
    #return RecommendRelationsResponse

if __name__ == "__main__":
	rospy.sleep(2)
	s = rospy.Service("/squirrel_relations_prediction", RecommendRelations, callback)
	rospy.loginfo("(squirrel prediction) Ready for predicting missing values.")
	rospy.spin()
