from tqdm import tqdm
import tensorflow as tf
import tensorflow_datasets as tfds
import pickle
import textwrap
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image
import numpy as np
from absl import app
from absl import flags
from pathlib import Path
from .product import Product

FLAGS = flags.FLAGS

def main(argv):
    DATASET_FOLDER = Path("dataset/validation_dataset_w_embeddings")
    # load dataset
    validation_dataset = tf.data.experimental.load(str(DATASET_FOLDER), 
                                                (tf.TensorSpec(shape=(), dtype = tf.int32), #id
                                                tf.TensorSpec(shape=(), dtype = tf.string), #name 
                                                tf.TensorSpec(shape=(), dtype = tf.string), #category
                                                tf.TensorSpec(shape=(), dtype = tf.string), #subcategory
                                                tf.TensorSpec(shape=(), dtype = tf.string), #caption
                                                tf.TensorSpec(shape=(256,256, 3), dtype = tf.uint8), #image 
                                                tf.TensorSpec(shape=(131072,), dtype = tf.float32), #image features
                                                tf.TensorSpec(shape=(768,), dtype = tf.float32), #image_embedding
                                                tf.TensorSpec(shape=(768,), dtype = tf.float32)), #caption embedding
                                                compression="GZIP")

    products_dict = {} # a dictionary that maps ids to products
    for p in tqdm(tfds.as_numpy(validation_dataset)):
        product = Product(p_id = p[0], name = p[1], caption = p[4], image = p[5], category = p[2], subcategory = p[3])
        product.image_features = p[6]
        product.embedding = p[7]
        product.caption_embedding = p[8]
        products_dict[p[0]] = product

    # Load candidate sets: lists of elements of the form: query_id -> [(candidate_id, label), ...].
    # The relevant document has label=1, the other ones 0
    txt2img_file = "./dataset/txt2img.pkl"
    img2txt_file = "./dataset/img2txt.pkl"

    #load
    txt2img_candidate_sets = pickle.load(open(txt2img_file, "rb"))
    img2txt_candidate_sets = pickle.load(open(img2txt_file, "rb"))

    # Rank@K Evaluation
    def euclidean_distance(x, y, axis = None):
        return np.linalg.norm(x-y, axis = axis)

    def compute_rank_at_k(sorted_documents, Ks):
        fount_at_top_k = {k:0 for k in Ks}
        for _, documents in sorted_documents.items():
            for i, document in enumerate(documents):
                if document[1]: #if label is equal to 1 we found the relevant document
                    fount_at_top_k = {k:v + (1 if k>=i+1 else 0) for k, v in fount_at_top_k.items()}
        return fount_at_top_k

    Ks = [1, 5, 10]

    txt2img_sorted_documents = {}
    print("=== Text-to-Image Retrieval ===")
    for query_id, candidates in txt2img_candidate_sets.items():
        caption_emb = np.array(products_dict[query_id].caption_embedding)
        image_embs = np.array([products_dict[candidate_id].embedding for candidate_id, _ in candidates])
        scores = np.array(euclidean_distance(caption_emb, image_embs, axis=1))
        sorted_indexes = np.argsort(scores)
        txt2img_sorted_documents[query_id] = list(map(candidates.__getitem__, list(sorted_indexes)))   
    rank_at_k = compute_rank_at_k(txt2img_sorted_documents, Ks)            
    for k in Ks:
        print(f"Rank @ {k}: {float(rank_at_k[k])/float(len(txt2img_sorted_documents) + 1e-5)}") 
                        
    print("=== Image-to-Text Retrieval ===")
    img2txt_sorted_documents = {}
    for query_id, candidates in img2txt_candidate_sets.items():
        caption_embs = np.array([products_dict[candidate_id].caption_embedding for candidate_id, _ in candidates])
        image_emb = np.array(products_dict[query_id].embedding)
        scores = np.array(euclidean_distance(image_emb, caption_embs, axis=1))
        sorted_indexes = np.argsort(scores)
        img2txt_sorted_documents[query_id] = list(map(candidates.__getitem__, list(sorted_indexes)))    
    rank_at_k = compute_rank_at_k(img2txt_sorted_documents, Ks)
    for k in Ks:
        print(f"Rank @ {k}: {float(rank_at_k[k])/float(len(img2txt_sorted_documents) + 1e-5)}") 



if __name__ == "__main__":
    app.run(main)
