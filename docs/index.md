
### Abstract
In this paper, we address the problem of cross-modal retrieval in e-commerce, focusing in particular on text-to-image and image-to-text retrieval of fashion products. 
State-of-the-art (SOTA) solutions for this task use transformers with quadratic attention, which cannot scale in training time and memory requirements. 
Moreover, they design the retrieval as a classification task that learns to output a similarity score for pairs of text and image coupled in a single embedding, 
with the drawback that each query is resolved with quadratic complexity by pairing it with every text or image in the entire dataset, precluding the scalability to real scenarios.
We propose a novel approach for efficient cross-modal retrieval of text and images by combining linear attention and metric learning to create a latent space where spatial 
proximity among instances translates into a semantic similarity score. Moreover, differently from existing contributions, our metric learning approach separately embeds text 
and images decoupling them and allowing to collocate and search in the space, after training, even for new images with missing text and vice versa. Experiments show that our 
solution significantly outperforms, both in efficacy and efficiency, existing SOTA contributions on the benchmark dataset FashionGen. Finally, we show that our approach enables 
the adoption of multi-dimensional indexing, with which cross-modal retrieval scales in logarithmic time up to millions, and potentially billions, of text and images.

### WebApp

[Web Application](http://137.204.107.42:37336/) 
