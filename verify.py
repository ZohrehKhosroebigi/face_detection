import fr_utils
import os
import numpy as np
class Verify():
    def verify(self,image_path, identity, database, model):
    # GRADED FUNCTION: verify
        """
        Function that verifies if the person on the "image_path" image is "identity".

        Arguments:
        image_path -- path to an image
        identity -- string, name of the person you'd like to verify the identity. Has to be an employee who works in the office.
        database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
        model -- your Inception model instance in Keras

        Returns:
        dist -- distance between the image_path and the image of "identity" in the database.
        door_open -- True, if the door should open. False otherwise.
        """

        # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
        self.encoding = fr_utils.img_to_encoding(image_path, model)

        if os.path.exists(image_path):
            os.remove(image_path)
        else:
            print("The file of image_path does not exist, maybe there is a problem with resizing")
        # Step 2: Compute distance with identity's image (≈ 1 line)
        self.dist = np.linalg.norm(self.encoding - database[identity])
        # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
        if self.dist < 0.7:
            #print("It's " + str(identity) + ", welcome in!")
            self.door_open = True
        else:
            #print("It's not " + str(identity) + ", please go away")
            self.door_open = False
        return self.dist, self.door_open
    def __repr__(self):
        return f'{self.dist}{self.door_open}'
    def __str__(self):
        return f'dist: {self.dist}\n  door status: {self.door_open}'