config = {
    # song
    'nb_songs': 50000,
    # user
    'embeddings_version': "svd",# can be "svd" for TT-SVD, or "mf" for UT-ALS in deezer data case
    'embeddings_dim': 128,#128 for TT-SVD, or 256 for UT-ALS
    # cuda setting
    'use_cuda': True,
    'device_number': 0,
    # model setting
    'input_dim': 2579, #2579 & 5139 for TT-SVD and UT-ALS train features respectively
    'nb_epochs': 130,
    'learning_rate': 0.00001,
    'batch_size': 512,
    'reg_param': 0,
    'drop_out': 0,
    # model training
    'eval_every': 10,
    'k_val': 50,
    #clustering for semi personalization strategy
    'nb_clusters': 1000,
    'max_iter': 20,
    'random_state': 0,
    # model evaluation
    'k_val_list': [50],
    'nb_iterations_eval_stddev': 2,
    'nb_sub_iterations_eval_stddev': 5,
    'indic_eval_evolution': 1000,
}

dataset_eval = ["validation", "test"]
