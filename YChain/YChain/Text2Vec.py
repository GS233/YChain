import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# 文本向量化部分

# 将文本或本文的列表转化为一组（1维或者2维）向量
def text_2_vec(text2vec_model,texts,batch_size=1,device='cpu'):
    embeddings = text2vec_model.encode(texts,batch_size=batch_size,device=device)
    return embeddings



# 计算query和数据库 的相似度 
# 找出最相似的
def similarity(input_embeddings, data_embeddings):
    one_dim_matrix = input_embeddings.reshape(1, -1)
    similarities = cosine_similarity(one_dim_matrix, data_embeddings)
    most_similar_index = np.argmax(similarities)
    most_similar_vector = data_embeddings[most_similar_index]
    return most_similar_index,similarities,most_similar_vector

# 同上 但可以输入文本
def find_most_similar_data(input, texts,text2vec_model):
    if type(texts[0]) == type('ab'):
        input_embeddings = text_2_vec(text2vec_model,input,batch_size=1)
        data_embeddings = text_2_vec(text2vec_model,texts,batch_size=1)
        most_similar_index,similarities,most_similar_vector = similarity(input_embeddings, data_embeddings)
        return most_similar_index,similarities
    else:
        input_embeddings = text_2_vec(text2vec_model,input,batch_size=1)
        most_similar_index,similarities,most_similar_vector = similarity(input_embeddings, texts)
        return most_similar_index,similarities








