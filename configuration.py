class Configuration:
    batch_size = 32
    img_height = 180
    img_width = 180
    validation_split = 0.2
    epochs = 10
    models_location = './models'
    model_name = 'updated'
    data_location = './data'
    test_data_location = './test-data'
    # name of existing model, which should be updated (for update_model_only_new_data)
    existing_model_name = 'initial'
