class ImagePreprocessor:
    def __init__(self, filepath):
        self.filepath = filepath

    def preprocess_images_in_df(self, df):
        df['image_path'] =  f"{self.filepath}/train/image_train/image_" + df['imageid'].astype(str) + "_product_" + df['productid'].astype(str) + '.jpg'