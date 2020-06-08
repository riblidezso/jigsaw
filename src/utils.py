
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import  KFold
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model
from transformers import TFRobertaModel, TFAutoModel, AutoTokenizer, PretrainedConfig
from tqdm import tqdm
import tensorflow_addons as tfa



def connect_to_TPU():
    """Detect hardware, return appropriate distribution strategy"""
    try:
        # TPU detection. No parameters necessary if TPU_NAME environment variable is
        # set: this is always the case on Kaggle.
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
        strategy = tf.distribute.get_strategy()

    print("REPLICAS: ", strategy.num_replicas_in_sync)

    auto = tf.data.experimental.AUTOTUNE
    global_batch_size = BATCH_SIZE * strategy.num_replicas_in_sync

    return tpu, strategy, global_batch_size, auto


def create_model():
    print('Loading pretrained transformer and building model...')
    with strategy.scope():
        # this does not work somehow
        #model = tf.keras.models.load_model(KERAS_MODEL)
        # this one loads the enourmous decoder too, breaks low memory instance
        #transformer_layer = TFAutoModel.from_pretrained(MODEL)

        ### Load only config no weights ###
        config = PretrainedConfig.from_json_file('config.json')                
        transformer_layer = TFRobertaModel(config) 

        ### Make the cls model ###               
        model = build_model(transformer_layer)

        ### Load weights saved after creating the CLS model in a high mem 
        ### Instance
        #model.load_weights(KERAS_WEIGHTS)

        ### Fails in colab because of the grad copies memory footprint
        optimizer_transformer = tf.keras.optimizers.SGD(
            learning_rate=LR_TRANSFORMER)
        optimizer_head = tf.keras.optimizers.Adam(learning_rate=LR_HEAD)
        #optimizer_head = tfa.optimizers.AdamW(weight_decay=0.00001,
        #                                      learning_rate=LR_HEAD)

    model.summary()
    return model, optimizer_transformer, optimizer_head


def set_weights_low_memory_workaround():
    ### Load full pretrained model outside strategy scope ###
    transformer_layer = TFAutoModel.from_pretrained(MODEL)

    ### Make the cls model ###
    #model2 = build_model(transformer_layer)

    ### Assign weights 
    for tv1, tv2 in zip(model.layers[1].trainable_variables,
                        transformer_layer.trainable_variables):
        tv1.assign(tv2)



def build_model(transformer):
    inp = Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_word_ids")
    x = transformer(inp)[0][:, 0, :]
    x = Dropout(DROPOUT)(x)
    out = Dense(1, activation='sigmoid', name='custom')(x)
    model = Model(inputs=[inp], outputs=[out])
    
    return model




def define_losses_and_metrics():
    with strategy.scope():
        loss_object = tf.keras.losses.BinaryCrossentropy(
                reduction=tf.keras.losses.Reduction.NONE, from_logits=False)

        def compute_loss(targets, predictions):
            per_example_loss = loss_object(targets, predictions)
            loss =  tf.nn.compute_average_loss(
                per_example_loss, global_batch_size = global_batch_size)

            return loss

        train_accuracy_metrics = [
            tf.keras.metrics.AUC()
        ]

    return compute_loss, train_accuracy_metrics


























def create_dist_dataset(X, y=None, y2=None, training=False):
    dataset = tf.data.Dataset.from_tensor_slices(X)

    ### Add y if present ###
    if y is not None:
        dataset_y = tf.data.Dataset.from_tensor_slices(y)
    if y2 is not None:
        dataset_y2 = tf.data.Dataset.from_tensor_slices(y2)
        dataset_y = tf.data.Dataset.zip((dataset_y, dataset_y2))
    if y is not None:
        dataset = tf.data.Dataset.zip((dataset, dataset_y))

    ### Repeat if training ###
    if training:
        dataset = dataset.shuffle(len(X)).repeat()

    dataset = dataset.batch(global_batch_size).prefetch(auto)

    ### make it distributed  ###
    dist_dataset = strategy.experimental_distribute_dataset(dataset)

    return dist_dataset





















@tf.function
def train_step(data):
    transformer_trainable_variables = [ v for v in model.trainable_variables 
                                       if (('pooler' not in v.name)  and 
                                           ('custom' not in v.name))]
    cls_head_trainable_variables = [ v for v in model.trainable_variables 
                                     if 'custom'  in v.name]
    

    inputs, targets = data

    with tf.GradientTape(persistent=True) as g:
        predictions = model(inputs, training=True)
        loss = compute_loss(targets, predictions)
    gradients_transformer = g.gradient(loss, transformer_trainable_variables)
    gradients_cls_head = g.gradient(loss, cls_head_trainable_variables)
    del g
    
    # Transformer
    gradients_transformer = tf.distribute.get_replica_context(
        ).all_reduce('sum', gradients_transformer)
    gradients_transformer, _ = tf.clip_by_global_norm(gradients_transformer,
                                                      CLIP_NORM)
    optimizer_transformer.apply_gradients(
        zip(gradients_transformer, transformer_trainable_variables),
        experimental_aggregate_gradients=False)
    
    # CLS head
    #gradients_cls_head = tf.distribute.get_replica_context(
    #    ).all_reduce('sum', gradients_cls_head)
    #gradients_cls_head, _ = tf.clip_by_global_norm(gradients_cls_head,
    #                                                  CLIP_NORM)
    optimizer_head.apply_gradients(zip(gradients_cls_head, 
                                       cls_head_trainable_variables))

    for m,t,p in zip(train_accuracy_metrics, [targets], [predictions]):
        ### todo check for mask ###
        m.update_state(t, p)




def prediction_step(data):
    predictions = model(data, training=False)
    return predictions


@tf.function
def distributed_train_step(data):
    strategy.experimental_run_v2(train_step, args=(data,))


@tf.function
def distributed_prediction_step(data):
    predictions = strategy.experimental_run_v2(prediction_step, args=(data,))
    return strategy.experimental_local_results(predictions)


def predict(dataset, with_progbar=False, with_len=False):  
    iterator = dataset
    if with_progbar:
        if with_len:
            total = 0
            for _ in dataset:
                total+=1
            iterator = tqdm(dataset, total=total)
        else:
            iterator = tqdm(dataset)

    predictions = []
    for tensor in iterator:
        predictions.append(distributed_prediction_step(tensor))

    predictions = np.vstack(list(map(np.vstack,predictions)))

    return predictions


def train(total_steps, validate_every, verbose=1, tol=5):
    history = []
    best_weights=None
    step = 0
    if verbose==1:  pbar = tqdm(total=validate_every)

    for tensor in train_dist_dataset:
        distributed_train_step(tensor) 
        if verbose==1: pbar.update()

        if step % validate_every == 0:        
            ### Test loop with exact AUC ###
            test_metric = roc_auc_score(y_val, predict(val_dist_dataset))
     
            ### Print some info ###
            if verbose>=1:   
                metrics = [m.result().numpy() for m in train_accuracy_metrics]

                print("Step %d, val AUC: %.5f, train AUC:%.5f" % (
                     step, test_metric, metrics[0])) 
                
            ### Save history and update best model ###    
            history.append(test_metric)
            if history[-1]==max(history):
                if verbose>=1: print('Saving best model ...')
                best_weights = model.get_weights()

            ### Early stopping ###
            if len(history)>tol and (max(history) > max(history[-tol:])):
                if verbose>=1: print('Stopped early')
                break
                
            ### Reset (train) metrics and pbar ###
            for m in train_accuracy_metrics:
                m.reset_states()
            if verbose==1: pbar.reset()
            
        if step  == total_steps:
            break

        step+=1

    model.set_weights(best_weights)
    
    return max(history)
