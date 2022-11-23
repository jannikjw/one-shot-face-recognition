import torch
CFG = {
    "data": {
        "img_folder": "one-shot-face-recognition/data/img_align_celeba", # if using MTCNN, change file path to one-shot-face-recognition/data/img_align_celeba_mtcnn
        "mapping_file": "one-shot-face-recognition/data/identity_CelebA.txt",
        "image_size": 160,
        "use_full_train_data": False,
        "train_img_pp": 2,
    },
    "train": {
        "uses_MTCNN": False,
        "uses_augmentation": False,
        "uses_GAN": False,
        "is_finetuning": False,
        "is_feature_extracting": False,
        "device": 'cpu', # 'cuda:0'
        "num_workers": 2, # 0 if cuda:0
        "pin_memory": False, # True if cuda:0
        "batch_size": 10,
        "buffer_size": 1000,
        "epochs": 20,
        "optimizer": {
            "type": "adam",
            "lr": 0.1,
            "schedule": [25, 50, 75, 85],
        },
        "metrics": ["accuracy", "loss"]
    },
    "evaluate": {
        "classifier": "knn"
    },
    "model": {
        "model_type": "InceptionResNetv1",
        "model_weights": "vgg-face2",
        "margin": 0.5,
        "gamma": 0.1,
        "loss": "TripletLoss",
    }
}
