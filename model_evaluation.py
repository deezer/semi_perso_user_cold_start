import os
import pandas as pd
import numpy as np
import torch
import torch.nn
import pickle
from model import RegressionTripleHidden
from options import config
from sklearn.metrics import ndcg_score, dcg_score
import statistics
from sklearn.preprocessing import Normalizer

def evaluation(dataset_path, master_path, eval_type = "full_perso", embeddings_version="svd", model_filename=None, clustering_path=None, clusters_filename=None, nb_clusters=config["nb_clusters"]):

    use_cuda = config['use_cuda']
    target_dim = config['embeddings_dim']
    input_dim = config['input_dim']
    k_val_list = config["k_val_list"]
    indic_eval_evolution = config["indic_eval_evolution"]
    cuda = torch.device(0)
    model_filename = master_path + "/" + model_filename + ".pt"

    # Load testing dataset.
    print("--- Load testing dataset ---")
    testing_set_size = int((len(os.listdir("{}/{}/test".format(master_path, embeddings_version)))) / 3)
    test_xs = []
    listened_songs_test_ys = []
    goundtruth_list_test = []
    for idx in range(testing_set_size):
        if eval_type in ["full_perso", "semi_perso", "popularity"] :
            test_xs.append(pickle.load(open("{}/{}/test/x_{}.pkl".format(master_path, embeddings_version, idx), "rb")))
        elif eval_type in ["inputfeatures"] :
            vector = pickle.load(open("{}/{}/test/x_{}.pkl".format(master_path, embeddings_version, idx), "rb"))
            transformer = Normalizer().fit(vector.reshape(1, -1))
            norm_vector = torch.FloatTensor(transformer.transform(vector.reshape(1, -1))[0])
            test_xs.append(norm_vector)
        listened_songs_test_ys.append(pickle.load(open("{}/{}/test/y_listened_songs_{}.pkl".format(master_path, embeddings_version, idx), "rb")))
        goundtruth_list_test.append(pickle.load(open("{}/{}/test/groundtruth_list_{}.pkl".format(master_path, embeddings_version, idx), "rb")))
    if eval_type in ["avgd0stream"] : 
        listd1d30streams = pd.read_parquet(dataset_path+"/user_features_test_"+embeddings_version+".parquet", engine ='fastparquet')
        colavgd0stream = list(listd1d30streams)[2+target_dim*10:2+target_dim*10+target_dim]
        avgd0stream = listd1d30streams[["user_index"]+colavgd0stream]
        avgd0stream_df = avgd0stream.set_index("user_index", drop = True).sort_index()
        test_xs = avgd0stream_df.values
    total_test_dataset = list(zip(test_xs, listened_songs_test_ys, goundtruth_list_test))
    del(test_xs, listened_songs_test_ys, goundtruth_list_test)
    print("--- nb of test samples : "+str(len(total_test_dataset))+" ---")

    if eval_type in ["full_perso", "semi_perso", "avgd0stream"] :
        
        # Load song embeddings

        print("--- Load song embeddings ---")
        song_embeddings_path = dataset_path + "/song_embeddings.parquet"
        song_embeddings = pd.read_parquet(song_embeddings_path, engine = 'fastparquet').fillna(0)
        list_features = ["feature_"+str(i) for i in range(len(song_embeddings["features_" + embeddings_version][0]))]
        song_embeddings[list_features] = pd.DataFrame(song_embeddings["features_" + embeddings_version].tolist(), index= song_embeddings.index)
        song_embeddings_values = song_embeddings[list_features].values
        song_embeddings_values_ = torch.FloatTensor(song_embeddings_values.astype(np.float32))
        print("--- nb of songs : "+str(len(song_embeddings_values_))+" ---")

        if eval_type in ["full_perso", "semi_perso"] :
            
            # Load model saved

            print("--- Load model ---")
            regression_model = RegressionTripleHidden(input_dim = input_dim, output_dim = target_dim)
            regression_model.load_state_dict(torch.load(model_filename))
            reg = regression_model.eval()
            if use_cuda:
                reg = reg.to(device=cuda)
            print(reg)

        # if evaluation semi perso :
        if eval_type in ["semi_perso"]:

            print("--- Load centroids for semi perso evaluation ---")
            #centroids to assign segment
            with open(master_path + "/" + clustering_path + "/" + clusters_filename, "rb") as f:
                kmeans = pickle.load(f)
            centroids = kmeans.cluster_centers_
            centroids_df = pd.DataFrame(centroids)
            if use_cuda:
                centroid_ = torch.FloatTensor(centroids_df.values).to(device=cuda)
            else:
                centroid_ = torch.FloatTensor(centroids_df.values)
            print("--- nb of centroids : "+str(len(centroid_))+" ---")

            #proba by segment for all song ids
            print("--- Load proba by segment for all song ids ---")
            song_proba_by_segment = []
            for cluster_id in range(nb_clusters):
                song_proba_by_segment.append(pickle.load(open("{}/{}/list_proba_{}.pkl".format(master_path, clustering_path + "_probas_" + embeddings_version, cluster_id), "rb")))
            print("--- nb of proba by segment for all song ids : "+str(len(song_proba_by_segment))+" ---")

    elif eval_type in ["popularity"] :
        list_proba = generate_for_popularity_evaluation(dataset_path, embeddings_version="svd")
        print("list of probabilities for each song for popularity baseline loaded")

    elif eval_type in ["inputfeatures"]:

        print("--- Load centroids for inputfeatures evaluation ---")
        #centroids to assign segment
        with open(master_path + "/" + clustering_path + "/" + clusters_filename, "rb") as f:
            kmeans = pickle.load(f)
        centroids = kmeans.cluster_centers_
        centroids_df = pd.DataFrame(centroids)
        if use_cuda:
            centroid_ = torch.FloatTensor(centroids_df.values).to(device=cuda)
        else:
            centroid_ = torch.FloatTensor(centroids_df.values)
        cuda = torch.device(0)
        print("--- nb of centroids : "+str(len(centroid_))+" ---")

        #proba by segment for all song ids
        print("--- Load proba by segment for all song ids ---")
        song_proba_by_segment = []
        for cluster_id in range(nb_clusters):
            song_proba_by_segment.append(pickle.load(open("{}/{}/list_proba_{}.pkl".format(master_path, clustering_path + "_probas_" + embeddings_version, cluster_id), "rb")))
        print("--- nb of proba by segment for all song ids : "+str(len(song_proba_by_segment))+" ---")
        
    # Compute evaluation metrics : avg precision, recall and ndcg

    testing_set_size = len(total_test_dataset)
    a,b,c = zip(*total_test_dataset)
    batch_size = 1
    num_batch_test = int(testing_set_size / batch_size)
    current_ndcg = {}
    current_recalls = {}
    current_precisions = {}
    for k_val in k_val_list:
        current_ndcg[k_val] = []
    for k_val in k_val_list:
        current_recalls[k_val] = []
        current_precisions[k_val] = []
    print("--- Evaluation running : average precision, recall and ndcg ---")
    print(eval_type)
    with torch.set_grad_enabled(False):
        for i in range(num_batch_test):
            if i % indic_eval_evolution == 0 & i != 0 :
                print("eval done for "+str(i)+" users")
            if eval_type in ["full_perso", "semi_perso"] :
                if use_cuda:
                    batch_features_tensor_test = torch.stack(a[batch_size*i:batch_size*(i+1)]).cuda(device = cuda)
                else:
                    batch_features_tensor_test = torch.stack(a[batch_size*i:batch_size*(i+1)])
                    predictions_test = reg(batch_features_tensor_test)
            elif eval_type in ["avgd0stream"]:
                predictions_test = torch.FloatTensor(a[batch_size*i:batch_size*(i+1)])
            elif eval_type in ["inputfeatures"]:
                if use_cuda:
                    predictions_test = torch.stack(a[batch_size*i:batch_size*(i+1)]).cuda(device = cuda)
                else:
                    predictions_test = torch.stack(a[batch_size*i:batch_size*(i+1)])
            # list of song indexes listened by user - index
            groundtruth_test_list_id = list(b[batch_size*i:batch_size*(i+1)])[0]
            groundtruth_test_list = list(c[batch_size*i:batch_size*(i+1)])
            k_val_max = max(k_val_list)

            if eval_type in ["full_perso", "avgd0stream"] :
                proba_values = torch.mm(predictions_test.cpu(), song_embeddings_values_.transpose(0, 1))
                recommended_songs = (proba_values.topk(k= k_val_max, dim = 1)[1]).tolist()[0]

            elif eval_type in ["semi_perso", "inputfeatures"] :
                predicted_segment = segment_pred(predictions_test, centroid_, k = 1, cuda_name = cuda)[0]
                proba_values = song_proba_by_segment[int(predicted_segment)-1]
                recommended_songs = np.argsort(proba_values)[::-1]
            
            elif eval_type == "popularity" :
                proba_values = list_proba
                recommended_songs = np.argsort(proba_values)[::-1]
                    
            else :
                "error eval_type unknown"

            for k_val in k_val_list:
                intersection = set(groundtruth_test_list_id) & set(recommended_songs[:k_val])
                denom_precision = float(len(groundtruth_test_list_id)) if len(groundtruth_test_list_id) < k_val else float(k_val)
                precision = len(intersection)/denom_precision
                current_precisions[k_val].append(precision)
                denom_recall = float(len(groundtruth_test_list_id))
                recall = len(intersection)/denom_recall
                current_recalls[k_val].append(recall)
            groundtruth_array = np.array(groundtruth_test_list, int)
            if eval_type in ["full_perso", "avgd0stream"] :
                scores = np.asarray([proba_values.numpy()[0].tolist()])
            elif eval_type in ["semi_perso", "popularity", "inputfeatures"] :
                scores = np.asarray([proba_values])
            else :
                "error eval_type unknown"
            for k_val in k_val_list:
                ndcg = ndcg_score(groundtruth_array, scores, k=k_val)
                current_ndcg[k_val].append(ndcg)

    print('length dataset : '+str(num_batch_test))
    for keys in current_ndcg.keys():
        print("ndcg at "+ str(keys) +" is : "
              + str(sum(current_ndcg[keys])/float(len(current_ndcg[keys]))))
    for keys in current_recalls.keys():
        print("recall at "+ str(keys) +" is : "
              + str(sum(current_recalls[keys])/float(len(current_recalls[keys]))))
    for keys in current_precisions.keys():
        print("precision at "+ str(keys) +" is : "
              + str(sum(current_precisions[keys])/float(len(current_precisions[keys]))))

    # standard deviation estimation

    print("--- Evaluation running : standard deviation estimation ---")
    print(eval_type)

    max_loc = num_batch_test
    nb_iterations_eval_stddev = config["nb_iterations_eval_stddev"]
    nb_sub_iterations_eval_stddev = config["nb_sub_iterations_eval_stddev"]
    batch_size = int(len(total_test_dataset)/float(nb_sub_iterations_eval_stddev))
    batch_ndcg_list = {}
    batch_recall_list = {}
    batch_precision_list = {}
    for k_val in k_val_list:
        batch_ndcg_list[k_val] = []
        batch_recall_list[k_val] = []
        batch_precision_list[k_val] = []

    for iteration in range(nb_iterations_eval_stddev):
        torch.manual_seed(iteration)
        randInd = torch.randperm(max_loc)
        current_position = 0
        for i in range(nb_sub_iterations_eval_stddev):
            ending_position = min(current_position + batch_size, max_loc)
            for k_val in k_val_list:
                batch_recall = pd.DataFrame(current_recalls[k_val]).values[randInd[current_position : ending_position]]
                batch_recall_mean = sum(batch_recall)/float(len(batch_recall))
                batch_recall_list[k_val].append(batch_recall_mean[0])
                batch_precision = pd.DataFrame(current_precisions[k_val]).values[randInd[current_position : ending_position]]
                batch_precision_mean = sum(batch_precision)/float(len(batch_precision))
                batch_precision_list[k_val].append(batch_precision_mean[0])
                batch_ndcg = pd.DataFrame(current_ndcg[k_val]).values[randInd[current_position : ending_position]]
                batch_ndcg_mean = sum(batch_ndcg)/float(len(batch_ndcg))
                batch_ndcg_list[k_val].append(batch_ndcg_mean[0])
            current_position += batch_size

    print('length dataset : '+str(num_batch_test))
    for keys in batch_ndcg_list.keys():
        print("stddev ndcg at "+ str(keys) +" is : "
              + str(statistics.stdev(batch_ndcg_list[keys])))
    for keys in batch_recall_list.keys():
        print("stddev recall at "+ str(keys) +" is : "
              + str(statistics.stdev(batch_recall_list[keys])))
    for keys in batch_precision_list.keys():
        print("stddev precision at "+ str(keys) +" is : "
              + str(statistics.stdev(batch_precision_list[keys])))

def segment_pred(target_validation_estimated, centroid_, k = 10, cuda_name = torch.device(0)):
    use_cuda = config['use_cuda']
    n1, n2 = target_validation_estimated.size(0), centroid_.size(0)
    target_validation_norm_ = torch.sum(target_validation_estimated**2, dim=1)
    centroid_norm_ = torch.sum(centroid_**2, dim=1)
    centroid_norm_expand = centroid_norm_.expand(n1, n2).t()
    target_validation_norm_expand = target_validation_norm_.expand(n2, n1)
    product_ = centroid_.mm(target_validation_estimated.t())
    distance = - target_validation_norm_expand - centroid_norm_expand + 2 * product_
    idx = torch.topk(distance, k=k, dim=0)[1].float()
    if use_cuda:
        results = (idx+ torch.ones(k, target_validation_norm_.size(0)).to(device = cuda_name)).cpu().numpy()
    else:
        results = (idx+ torch.ones(k, target_validation_norm_.size(0))).numpy()
    return results

def generate_for_popularity_evaluation(dataset_path, embeddings_version="svd"):
    
    listd1d30streams = pd.read_parquet(dataset_path+"/user_features_train_"+embeddings_version+".parquet", engine = 'fastparquet')
    exploded_data = listd1d30streams[["user_index", "d1d30_songs"]].explode('d1d30_songs').set_index('d1d30_songs')
    grouped_data = exploded_data.groupby(['d1d30_songs']).size()
    popularity_df = pd.DataFrame(grouped_data / float(sum(grouped_data)))
    popularity_df.columns = ["proba"]
    list_proba = []
    for song_index in range(config["nb_songs"]):
        if song_index in popularity_df.index :
            list_proba.append(popularity_df.loc[song_index]["proba"])
        else :
            list_proba.append(0)
    
    return list_proba

