from data_loader import *


def build_model(input_shape,n_classes,train_ds,test_ds):
    ## Optimize for performance
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    #from keras.engine import sequential
    resnet_model= tf.keras.Sequential()

    pretraind_model= tf.keras.applications.ResNet50(include_top=False,
                                                    input_shape=input_shape,
                                                    pooling='avg',
                                                    classes=n_classes,
                                                    weights="imagenet")

    for layer in pretraind_model.layers:
        layer.trainable=False

    resnet_model.add(pretraind_model)
    resnet_model.add(tf.keras.layers.Flatten())
    resnet_model.add(tf.keras.layers.Dense(512,activation="relu"))
    resnet_model.add(tf.keras.layers.Dense(n_classes))
    
    
    resnet_model.summary()
    
    return resnet_model
    
def compile_model(model,train_ds,test_ds):
    
    print(f'Model Address : {model_name}\nHyper parametrs are :\nEPOCHS = {EPOCHS}\nALPHA = {ALPHA}\nBATCH_SIZE = {BATCH_SIZE}\nES_PATIENCE = {ES_PATIENCE}\nLR_PATIENCE ={LR_PATIENCE}\nLR_FACTOR = {LR_FACTOR}') ##--Add all hyper parameters here so that it will be appended to log file
    
    
    model.compile(optimizer='adam',
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])
    
    history=model.fit(train_ds,
                     epochs=EPOCHS,
                     validation_data=test_ds,
                     batch_size=BATCH_SIZE)



    #import sys
    #filename=log_file_gen() ## calling blank log file here
    #temp=sys.stdout ## assigning sys.stdout to temparary variable
    #sys.stdout = open(filename, 'w')  ## re directing console output to log file to be stored
    
    
    #res_df = pd.DataFrame(history.history)
    #print(res_df)
    
    # Rearranging index
    #res_df.index = np.arange(1, len(res_df) + 1)
    
    # Display modified DataFrame
    #print(res_df)
    
    #sys.stdout = temp  ## re assigning sys.stdout to its original value
