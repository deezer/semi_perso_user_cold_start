import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import pickle
from options import config

def train_kmeans(dataset_path, master_path, clustering_path, nb_clusters, max_iter, random_state, embeddings_version="svd", clusters_filename=None):

    # user embeddings (target = only for train users)

    user_embeddings = pd.read_parquet(dataset_path + "/user_embeddings.parquet", engine = 'fastparquet')
    list_embeddings = ["embedding_"+str(i) for i in range(len(user_embeddings[embeddings_version + "_embeddings"][0]))]
    user_embeddings[list_embeddings] = pd.DataFrame(user_embeddings[embeddings_version + "_embeddings"].tolist(), index= user_embeddings.index)
    user_embeddings_values = user_embeddings[list_embeddings].values

    # clustering

    kmeans = KMeans(n_clusters=nb_clusters, random_state=random_state, max_iter = max_iter, n_jobs=None, precompute_distances='auto').fit(user_embeddings_values)
    with open(master_path + "/" + clustering_path + "/" + clusters_filename, "wb") as f:
        pickle.dump(kmeans, f)

    # top songs by cluster

    user_embeddings["cluster"] = kmeans.labels_
    user_clusters = user_embeddings[["user_index", "cluster"]]

    # load songs listened between D1 and D30 for train users

    features_train_path = dataset_path  + "/user_features_train_" + embeddings_version + ".parquet"
    features_train = pd.read_parquet(features_train_path, engine = 'fastparquet').fillna(0)
    features_train = features_train.sort_values("user_index")
    features_train = features_train.reset_index(drop=True)#to check it is ok for train data

    listd1d30 = features_train[["user_index", "d1d30_songs"]]
    listd1d30 = pd.merge(listd1d30, user_clusters, left_on = "user_index", right_on = "user_index")
    listd1d30_exploded = listd1d30.explode('d1d30_songs')
    listd1d30_exploded["count"] = np.ones(len(listd1d30_exploded))
    listd1d30_by_cluster = pd.DataFrame(listd1d30_exploded.groupby(["cluster", "d1d30_songs"])['count'].count())

    # most popular songs by cluster

    nb_songs = config["nb_songs"]
    arrays = (np.repeat(np.arange(nb_clusters), repeats = nb_songs), np.tile(np.arange(nb_songs), nb_clusters))
    tuples = list(zip(*arrays))
    index_perso = pd.MultiIndex.from_tuples(tuples, names=["cluster", "song_index"])
    df = pd.DataFrame(index=["default"], columns=index_perso).T
    both = pd.concat([listd1d30_by_cluster, df], axis=1)[["count"]].fillna(0)
    both = both.reset_index(drop=False)
    both.columns = ["cluster", "song_index", "nb_streams"]
    data_by_cluster = pd.DataFrame(both.groupby("cluster")['nb_streams'].sum())
    data_by_cluster.columns = ["nb_streams_by_cluster"]
    data_by_cluster_and_song = pd.merge(both, data_by_cluster, left_on = "cluster", right_on = "cluster")
    data_by_cluster_and_song["segment_proba"]=data_by_cluster_and_song["nb_streams"]/data_by_cluster_and_song["nb_streams_by_cluster"]

    if not os.path.exists("{}/{}/".format(master_path, clustering_path + "_probas_" + embeddings_version)):
        os.mkdir("{}/{}/".format(master_path, clustering_path + "_probas_" + embeddings_version))
    for cluster_id in range(nb_clusters):
        if cluster_id % 100 == 0:
            print("probas by cluster and by song computed for cluster : "+ str(cluster_id))
        list_proba = []
        for song_index in range(nb_songs):
            list_proba.append(data_by_cluster_and_song.iloc[cluster_id*nb_songs+song_index]["segment_proba"])
        pickle.dump(list_proba, open("{}/{}/list_proba_{}.pkl".format(master_path, clustering_path + "_probas_" + embeddings_version, cluster_id), "wb"))