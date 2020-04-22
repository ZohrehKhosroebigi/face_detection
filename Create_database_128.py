import fr_utils
class Create_database_128():
    def __init__(self):
        self.database = {}
    def create_data_128(self,FRmodel):

        self.database["danielle"] = fr_utils.img_to_encoding("images/danielle.png", FRmodel)
        self.database["younes"] = fr_utils.img_to_encoding("images/younes.jpg", FRmodel)
        self.database["tian"] = fr_utils.img_to_encoding("images/tian.jpg", FRmodel)
        self.database["andrew"] = fr_utils.img_to_encoding("images/andrew.jpg", FRmodel)
        self.database["kian"] = fr_utils.img_to_encoding("images/kian.jpg", FRmodel)
        self.database["dan"] = fr_utils.img_to_encoding("images/dan.jpg", FRmodel)
        self.database["sebastiano"] = fr_utils.img_to_encoding("images/sebastiano.jpg", FRmodel)
        self.database["bertrand"] = fr_utils.img_to_encoding("images/bertrand.jpg", FRmodel)
        self.database["kevin"] = fr_utils.img_to_encoding("images/kevin.jpg", FRmodel)
        self.database["felix"] = fr_utils.img_to_encoding("images/felix.jpg", FRmodel)
        self.database["benoit"] = fr_utils.img_to_encoding("images/benoit.jpg", FRmodel)
        self.database["arnaud"] = fr_utils.img_to_encoding("images/arnaud.jpg", FRmodel)
        return self.database
    def __str__(self):
        return f'{self.database}'
    def __repr__(self):
        return f'{self.database}'