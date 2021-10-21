STRINGS_ENCODING = 'ISO-8859-1'

class Product:
    def __init__(self, p_id, name, caption, image, category, subcategory, pose="id_gridfs_1"):
        self.p_id = p_id
        self.pose = pose
        self.name = name
        self.caption = caption
        self.image = image
        self.category = category
        self.subcategory = subcategory
        self._image_features = None
        self._embedding = None
        self._caption_embedding = None

    @property
    def image_features(self):
        return self._image_features

    @image_features.setter
    def image_features(self, value):
        self._image_features = value

    @property
    def embedding(self):
        return self._embedding

    @embedding.setter
    def embedding(self, value):
        self._embedding = value

    @property
    def caption_embedding(self):
        return self._caption_embedding

    @caption_embedding.setter
    def caption_embedding(self, value):
        self._caption_embedding = value

    def decoded_caption(self):
        return self.caption.decode(STRINGS_ENCODING).replace('ÃÂÃÂÃÂÃÂ©', '')

    def __lt__(self, other):
        return self.p_id < other.p_id

    def __eq__(self, other):
        return self.p_id == other.p_id and self.pose == other.pose

    def __hash__(self):
        return hash((self.p_id, self.pose))

    def __str__(self):
        return f"product_id: {self.p_id}\nname: {self.name.decode(STRINGS_ENCODING) }\ncaption: {self.caption.decode(STRINGS_ENCODING) }\ncategory: {self.category.decode(STRINGS_ENCODING) } \nsubcategory: {self.subcategory.decode(STRINGS_ENCODING) }"
