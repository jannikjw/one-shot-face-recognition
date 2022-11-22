CFG = {
    "data": {
        "path": "",
        "image_size": 160,
        "use_full_train_data": False,
        "train_img_pp": 2,
    },
    "train": {
        "uses_MTCNN": True,
        "uses_augmentation": False,
        "uses_GAN": False,
        "is_finetuning": False,
        "is_feature_extracting": False,
        "batch_size": 128,
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
