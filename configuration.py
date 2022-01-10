class Configuration:
    # training configuration
    batch_size = 32
    validation_split = 0.2
    epochs = 10

    # data details
    img_height = 180
    img_width = 180

    # file locations
    models_location = './models'
    data_location = 'D:\\Studia\\magisterka\\sem2\\IUI\\data'
    test_data_location = 'D:\\Studia\\magisterka\\sem2\\IUI\\data\\test_data'
    update_data_location = 'D:\\Studia\\magisterka\\sem2\\IUI\\data\\subsets'

    # name of existing model, which should be updated (for update_model_only_new_data)
    base_model_name = 'base'
