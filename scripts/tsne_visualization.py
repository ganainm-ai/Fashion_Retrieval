from .product import Product
import tensorflow as tf
import matplotlib.pyplot as plt
import random as rnd
from sklearn.manifold import TSNE
from tqdm import tqdm
import tensorflow_datasets as tfds
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea
import numpy as np
from pathlib import Path
from absl import app
from absl import flags
import textwrap


FLAGS = flags.FLAGS
flags.DEFINE_integer("perplexity", 65, "perplexity")
flags.DEFINE_float("early_exaggeration", 12.0, "early_exaggeration")
flags.DEFINE_integer("n_iter", 2000, "n_iter")
flags.DEFINE_integer("random_state", 42, "random_state")
flags.DEFINE_integer("learning_rate", 200, "learning_rate")

def textscatter(x, y, queries, box_alignments, ax=None, text_size = 15):
  if ax is None:
    ax = plt.gca()
    
  artists = []
  for x0, y0, q, box_align in zip(x, y, queries, box_alignments):
    ax.plot(x0,y0, ".r", markersize=120)
    offsetbox = TextArea(f"{textwrap.fill(q, width=42)}", minimumdescent=False, textprops ={"size":text_size})
          
        
    ab = AnnotationBbox(offsetbox, (x0, y0),
                      xycoords='data',
                      boxcoords=None,
                      box_alignment=box_align,
                      arrowprops=dict(arrowstyle="->"))
    ab.set_zorder(-1)
    artists.append(ax.add_artist(ab))

  ax.update_datalim(np.column_stack([x, y]))
  ax.autoscale()
  return artists

def imscatter(x, y, images, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    artists = []
    for x0, y0, i in zip(x, y, images):
        im = OffsetImage(i, zoom=zoom)
        #x, y = np.atleast_1d(x, y)
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        ab.set_zorder(-2)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

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
    print("LOADING DATASET...")
    products_list = [] # a list of products
    for p in tqdm(tfds.as_numpy(validation_dataset)):
        product = Product(p_id = p[0], name = p[1], caption = p[4], image = p[5], category = p[2], subcategory = p[3])
        #product.image_features = p[6]
        product.embedding = p[7]
        product.caption_embedding = p[8]
        products_list.append(product)

    print("COMPUTING TSNE...")
    captions_ids = [91765, 108990, 88120, 1582273]
    caption_embs = [p.caption_embedding for p in products_list if p.p_id in captions_ids]
    tsne = TSNE(n_components=2,perplexity=65, early_exaggeration=12.0, n_iter=2000, random_state=42, learning_rate=200, init='pca')

    tsne = tsne.fit_transform(np.concatenate(([p.embedding for p in products_list], caption_embs)))


    fig, ax = plt.subplots(figsize=(160,100))
    plt.axis('off')

    num_images = len(products_list)
    ROUND_COORDINATES = True # round t-sne coordinates to nearest integer to have a better result
    if not ROUND_COORDINATES:
        imscatter(tsne[:,0], tsne[:,1], [p.image for p in products_list], ax, zoom=0.3)
    else:
        imscatter(list(map(round, tsne[:,0])), list(map(round, tsne[:,1])),  [p.image for p in products_list], ax, zoom=0.3)
    
    textscatter(tsne[num_images:, 0], tsne[num_images:, 1], 
                [p.decoded_caption() for p in products_list if p.p_id in captions_ids], 
                [(1.5, 1.5), (-0.3, -0.3), (1.5, 5.5), (0.5, 4)], ax, text_size=100)
    filename = "tsne.svg"
    print(f"SAVING RESULT TO {filename}...")
    fig.savefig(filename, bbox_inches='tight')

if __name__ == "__main__":
    app.run(main)
