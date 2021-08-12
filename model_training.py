import os
import pandas as pd
import numpy as np
import torch
import torch.nn
import time
import pickle
import random
from model import RegressionTripleHidden
from options import config


def training(dataset_path, master_path, embeddings_version="svd", eval=True, model_save=True, model_filename=None):

    use_cuda = config['use_cuda']
    cuda_number = config['device_number']
    cuda = torch.device(cuda_number)
    target_dim = config['embeddings_dim']
    input_dim = config['input_dim']
    nb_epochs = config['nb_epochs']
    learning_rate = config['learning_rate']
    reg_param = config['reg_param']
    drop_out = config['drop_out']
    batch_size = config['batch_size']
    eval_every = config['eval_every']
    k_val = config['k_val']

    if not os.path.exists(master_path + "/" + model_filename + ".pt"):

        print("--- no model pre-existing for "+embeddings_version+" : training regression model running ---")

        # Load training dataset.
        training_set_size = int(len(os.listdir("{}/{}/train".format(master_path, embeddings_version))) / 2)
        train_xs = []
        train_ys = []
        for idx in range(training_set_size):
            train_xs.append(pickle.load(open("{}/{}/train/x_train_{}.pkl".format(master_path, embeddings_version, idx), "rb")))
            train_ys.append(pickle.load(open("{}/{}/train/y_train_{}.pkl".format(master_path, embeddings_version, idx), "rb")))
        total_dataset = list(zip(train_xs, train_ys))
        del(train_xs, train_ys)

        if eval:

            # Load validation dataset.

            validation_set_size = int(len(os.listdir("{}/{}/validation".format(master_path, embeddings_version))) / 3)
            validation_xs = []
            listened_songs_validation_ys = []
            for idx in range(validation_set_size):
                validation_xs.append(pickle.load(open("{}/{}/validation/x_{}.pkl".format(master_path, embeddings_version, idx), "rb")))
                listened_songs_validation_ys.append(pickle.load(open("{}/{}/validation/y_listened_songs_{}.pkl".format(master_path, embeddings_version, idx), "rb")))
            total_validation_dataset = list(zip(validation_xs, listened_songs_validation_ys))
            del(validation_xs, listened_songs_validation_ys)

            # Load song embeddings for evaluation

            song_embeddings_path = dataset_path + "/song_embeddings.parquet"
            song_embeddings = pd.read_parquet(song_embeddings_path, engine = 'fastparquet')
            list_features = ["feature_" + str(i) for i in range(len(song_embeddings["features_" + embeddings_version][0]))]
            song_embeddings[list_features] = pd.DataFrame(song_embeddings["features_" + embeddings_version].tolist(), index= song_embeddings.index)
            song_embeddings_values = song_embeddings[list_features].values
            song_embeddings_values_ = torch.FloatTensor(song_embeddings_values.astype(np.float32))

        if use_cuda:    
            regression_model = RegressionTripleHidden(input_dim = input_dim, output_dim = target_dim, drop_out = drop_out).cuda(device = cuda)
        else:
            regression_model = RegressionTripleHidden(input_dim = input_dim, output_dim = target_dim, drop_out = drop_out)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(regression_model.parameters(), lr = learning_rate, weight_decay=reg_param )

        print("training set size : "+str(training_set_size))
        print("validation set size : "+str(validation_set_size))
        print("input dimension : " + str(input_dim))
        print("regression model : "+ str(regression_model))
        print("training running")

        loss_train = []

        for nb in range(nb_epochs):
            print("nb epoch : "+str(nb))
            start_time_epoch = time.time()
            random.Random(nb).shuffle(total_dataset)
            a,b = zip(*total_dataset)
            num_batch = int(training_set_size / batch_size)
            if use_cuda:
                regression_model = regression_model.to(device = cuda)                
            for i in range(num_batch):
                optimizer.zero_grad()
                if use_cuda:
                    batch_features_tensor = torch.stack(a[batch_size*i:batch_size*(i+1)]).cuda(device = cuda)
                    batch_target_tensor = torch.stack(b[batch_size*i:batch_size*(i+1)]).cuda(device = cuda)
                else:
                    batch_features_tensor = torch.stack(a[batch_size*i:batch_size*(i+1)])
                    batch_target_tensor = torch.stack(b[batch_size*i:batch_size*(i+1)])
                output_tensor = regression_model(batch_features_tensor)
                loss = criterion(output_tensor, batch_target_tensor)
                loss.backward()
                optimizer.step()
                loss_train.append(loss.item())
            print('epoch ' + str(nb) +  " training loss : "+ str(sum(loss_train)/float(len(loss_train))))
            print("--- seconds ---" + str(time.time() - start_time_epoch))

            if nb != 0 and (nb % eval_every == 0 or nb == nb_epochs - 1):
                print('testing model')
                start_time_eval = time.time()
                reg = regression_model.eval()
                if use_cuda:
                    reg = reg.to(device=cuda)
                validation_set_size = len(total_validation_dataset)
                a,b = zip(*total_validation_dataset)
                num_batch_validation = int(validation_set_size / batch_size)
                current_precisions = []
                with torch.set_grad_enabled(False):
                    for i in range(num_batch_validation):
                        if use_cuda:
                            batch_features_tensor_validation = torch.stack(a[batch_size*i:batch_size*(i+1)]).cuda(device = cuda)
                        else:
                            batch_features_tensor_validation = torch.stack(a[batch_size*i:batch_size*(i+1)])
                        predictions_validation = reg(batch_features_tensor_validation)
                        groundtruth_validation = list(b[batch_size*i:batch_size*(i+1)])
                        predictions_songs_validation = torch.mm(predictions_validation.cpu(), song_embeddings_values_.transpose(0, 1))
                        recommendations_validation = (predictions_songs_validation.topk(k= k_val, dim = 1)[1]).tolist()
                        precisions = list(map(lambda x, y: len(set(x) & set(y))/float(min(len(x), k_val)), groundtruth_validation, recommendations_validation))
                        current_precisions.extend(precisions)
                print('epoch ' + str(nb) +  " precision test : "+ str(sum(current_precisions) / float(len(current_precisions))) )
                print("--- %s seconds ---" + str(time.time() - start_time_eval))
        print("--- training finished ---")

        if model_save:
            print("--- saving model ---")
            torch.save(regression_model.state_dict(), master_path + "/" + model_filename + ".pt")
            print(regression_model)
            print("--- model saved ---")

    else:
        print("--- there is already a model pre-existing for "+embeddings_version+" : no need to run training again ---")
