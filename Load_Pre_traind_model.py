#Loading the pre-trained model , we load a FaceNet  trained model
import fr_utils
class Load_Pre_traind_model():
    def __init__(self):
        self.FRmodel = None
    def compile_model(self,FRmodel,myoptimizer, myloss, mymetrics):
        self.FRrmodel=FRmodel.compile(optimizer=myoptimizer, loss=myloss, metrics=mymetrics)
        fr_utils.load_weights_from_FaceNet(self.FRmodel)
        return self.FRmodel


