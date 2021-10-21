import gdown
import os


# Download dataset with embeddings
dataset_name = "validation_dataset_w_embeddings"
dataset_folder = os.path.join("dataset", dataset_name)

if (not os.path.exists(dataset_folder)):
    gdown.download(f"https://drive.google.com/uc?id=1KcDFzv4JjuEQyyIvC7BkHgxN7LzbInPl", f"./dataset/{dataset_name}.tar.gz", False)


# Download test candidate sets
txt2img_file = os.path.join("dataset", "txt2img.pkl")
img2txt_file = os.path.join("dataset", "img2txt.pkl")
#download files
if (not os.path.exists(txt2img_file)):
    gdown.download(f"https://drive.google.com/uc?id=101zaYhWws6CkWePEdBg5WVb9C8HXflC5", txt2img_file, False)
if (not os.path.exists(img2txt_file)):
    gdown.download(f"https://drive.google.com/uc?id=1IV61fyjUt7Pgve2t4ZuqS7XpAPw9vDLO", img2txt_file, False)