CFG = {
    "data": {
        "img_folder": "data/img_align_celeba", # change path according to image folder dataset (mtcnn, data_aug, etc)
        "mapping_file": "data/identity_CelebA.txt", # change path for correct mapping.txt file (mtcnn, data_aug, etc)
        "image_size": 160
    },
    "train": {
        "uses_MTCNN": False,
        "uses_augmentation": False,
        "uses_GAN": False, # if use_GAN is true, then uses_augmentation MUST BE False
        "is_finetuning": False,
        "is_feature_extracting": True,
        "num_workers": 20, # 0 if cuda:0
        "pin_memory": False, # True if cuda:0
        "batch_size": 128,
        "buffer_size": 1000,
        "epochs": 20,
        "optimizer": {
            "type": "adam",
            "lr": 0.01,
            "schedule": [25, 50, 75, 85],
        },
        "metrics": ["accuracy", "loss"],
        "subsample_positives": True,
        "num_positive": 1, 
    },
    "evaluate": {
        "classifier": "knn"
    },
    "model": {
        "model_type": "InceptionResNetv1",
        "model_weights": "vggface2",
        "margin": 0.5,
        "gamma": 0.1,
        "loss": "TripletLoss"
    }
}

# path for dataset + GAN
# 'data/all_images'
# 'data/identity_CelebA_all.txt'

# path for mtcnn dataset
# 'data/img_align_celeba_mtcnn'
# 'data/identity_CelebA.txt'
