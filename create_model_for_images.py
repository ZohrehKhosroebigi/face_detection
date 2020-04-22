#create the model from inception model for face images
import inception_blocks_v2
class Create_model_Images():
    def createmdl_img(self,input_shape):
        self.FRmodel = inception_blocks_v2.faceRecoModel(input_shape)
        #print("model is created and Total Params:" + str(self.FRmodel.count_params()))
        return self.FRmodel
    def __repr__(self):
        return f'{self.FRmodel}'
