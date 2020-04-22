import fr_utils
import os
import numpy as np
class Recognition ():
    def __init__(self):
        self.min_dist = 200
    def recognition(self,image_path, database, model):
            """
            Implements face recognition for the office by finding who is the person on the image_path image.

            Arguments:
            image_path -- path to an image
            database -- database containing image encodings along with the name of the person on the image
            model -- your Inception model instance in Keras

            Returns:
            min_dist -- the minimum distance between image_path encoding and the encodings from the database
            identity -- string, the name prediction for the person on image_path
            """

            ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
            self.encoding = fr_utils.img_to_encoding(image_path,model)
            if os.path.exists(image_path):
                os.remove(image_path)
            else:
                print("The file of image_path does not exist, maybe there is a problem with resizing")
            ## Step 2: Find the closest encoding ##

            # Initialize "min_dist" to a large value, say 100 (≈1 line)

            # Loop over the database dictionary's names and encodings.
            for (name, db_enc) in database.items():

                # Compute L2 distance between the target "encoding" and the current db_enc from the database. (≈ 1 line)
                self.dist = np.linalg.norm(self.encoding-db_enc)

                # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
                if self.dist<self.min_dist:
                    self.min_dist = self.dist
                    self.identity = name
            if self.min_dist > 0.7:
                #print("Not in the database.")
                self.trueperson = False
            else:
                self.trueperson = True
                #print ("it's " + str(self.identity) + ", the distance is " + str(self.min_dist))

            return self.min_dist, self.identity,self.trueperson
    def __repr__(self):
        return f'{self.min_dist}{self.identity}{self.trueperson}'
    def __str__(self):
        return f'dist: {self.min_dist}  person is: {self.identity}  door status: {self.trueperson}'