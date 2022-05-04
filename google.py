import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import re
import tensorflow_datasets as tfds
from sklearn.feature_extraction import DictVectorizer
from collections import Counter
from scipy.sparse import find, csr_matrix, linalg, diags
from tkinter import *
from tkinter import ttk

def prepare_data():
    # loding dataset
    ds = tfds.load('wikipedia/20201201.simple', shuffle_files=False, split="train")
    ds_np = tfds.as_numpy(ds)
    
    # spliting articles into lists of words and preparing raw texts for quick querying
    arts_split = []
    texts = []
    for art in ds_np:
        words = re.sub(r"[^a-zA-Z0-9]", " ", art['text'].decode("utf-8").lower()).split(' ')
        arts_split.append(words)
        texts.append(art['title'].decode("utf-8") + "\n\n" + art['text'].decode("utf-8") + "\n\n" + "-" * 80)
    
    # freeing memory  
    del ds
    del ds_np
    
    # building sparse matrix
    vectorizer = DictVectorizer(sparse=True)
    doc_term_mat = vectorizer.fit_transform(Counter(art) for art in arts_split)
    
    # multiplying by IDF
    IDF = np.log(doc_term_mat.shape[1] / (doc_term_mat != 0).sum(axis = 0))
    doc_term_mat_processed = doc_term_mat.multiply(IDF)
    
    # using svd and low rank approximation to decrease noise
# NOTE WE"RE NOT USING IT AS MULTIPYING THOSE MATRICIES AT FULL DATASET TAKES AGES
#     u, s, vh = linalg.svds(doc_term_mat_processed, k=10)

#     u = csr_matrix(u)
#     s = csr_matrix(diags(s))
#     vh = csr_matrix(vh)

#     doc_term_mat_processed = u @ s @ vh
    
    return doc_term_mat, doc_term_mat_processed, vectorizer, texts


def make_query(doc_term_mat, vectorizer, texts, query_string, no_of_results = 5):
    # split query string into words and vectorize it
    query_words = re.sub(r"[^a-zA-Z0-9]", " ", query_string.lower()).split(' ')
    query_vector = vectorizer.transform(Counter(query_words))
    
    # computer correlations
    corr = doc_term_mat @ query_vector.transpose() / linalg.norm(query_vector) / linalg.norm(doc_term_mat)
    
    # find best results
    result = find(corr)

    idx = np.argpartition(result[2], -no_of_results)[-no_of_results:]
    indices = idx[np.argsort((-result[2])[idx])]
    
    # print teasers
    counter = 1
    text_result.delete("1.0", END)
    for ind in indices:
        
        text_result.insert(END, "\n\n" + texts[result[0][ind]])
        counter += 1


doc_term_mat, doc_term_mat_processed, vectorizer, texts = prepare_data()

# initializers
root = Tk()
content = ttk.Frame(root)

# gui elements
query_string = StringVar()
l_title = ttk.Label(content, text="Totally not Google!")
b_search = ttk.Button(content, text="Search!", command = lambda: make_query(doc_term_mat, vectorizer, texts, query_string.get()))
e_query_string = ttk.Entry(content, textvariable=query_string)

text_result = Text(content, width = 100, height = 100)
text_result.insert("1.0", "results...")

# attaching to grid
content.grid(column=0, row=0)
l_title.grid(column=0, row=0)
e_query_string.grid(column=0, row=1)
b_search.grid(column=0, row=2)
text_result.grid(column=0, row=3)


root.mainloop()