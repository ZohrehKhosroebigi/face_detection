#Loading the pre-trained model , we load a FaceNet  trained model
import fr_utils
from Triplet_loss import Triplet_loss

class Load_Pre_traind_model():
    def __init__(self):
        pass
        #myFRmodel.FRmodel, 'adam', myloss.triplet_loss(y_true, y_pred, alpha=0.2),['accuracy']

    def compile_model(self,FRmodel,myoptimizer, myloss, mymetrics):
        #print("---1---"); print(FRmodel);print("---2---");print(myoptimizer);print("---3---");print(myloss)
        #print("---4---");print(mymetrics);print("---5---")
        FRmodel.compile(optimizer=myoptimizer, loss=myloss, metrics=mymetrics)
        fr_utils.load_weights_from_FaceNet(FRmodel)
        #print("Loaded the pre-trained model by created model")
        return FRmodel



