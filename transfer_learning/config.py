# Configuration for the data
DATA_TRAIN_INDEX_PATH = "" # .jsonl file
DATA_EVAL_INDEX_PATH  = "" # .jsonl file

# Configuration for the large model
PATCH_STREAM = True                                             # Stream training / inference
PATCH_SIZE = 16                                                 # Patch Size
PATCH_LENGTH = 1024                                             # Patch Length
CHAR_NUM_LAYERS = 6                                             # Number of layers in the decoder
PATCH_NUM_LAYERS = 20                                           # Number of layers in the encoder
HIDDEN_SIZE = 1280                                              # Hidden Size

# Configuration for the training
BATCH_SIZE = 1         
LEARNING_RATE = 1e-5                                            # Non-Transfer: 1e-4, FT: 1e-5
NUM_EPOCHS = 32                                                 # Number of epochs to train for (if early stopping doesn't intervene)
EARLY_STOP_PATIENCE = 4
PATCH_SAMPLING_BATCH_SIZE = 0                                   # Batch size for patch during training

WEIGHT_DECAY = 1e-2                                             # AdamW
WARMUP_STEPS = 1000                                             # LR Scheduler
MAX_GRAD_NORM = 1.0      

LOAD_FROM_CHECKPOINT = False                                    # Whether to load weights from a checkpoint
WANDB_LOGGING = True                                            # Whether to log to wandb
WANDB_KEY = '' 

PRETRAINED_PATH = ""                                            # Path of pretrained weights
EXP_TAG = 'cguitar_'                                            # Experiment tag for name differentiation
NAME =  EXP_TAG + \
        "_p_size_" + str(PATCH_SIZE) + \
        "_p_length_" + str(PATCH_LENGTH) + \
        "_p_layers_" + str(PATCH_NUM_LAYERS) + \
        "_c_layers_" + str(CHAR_NUM_LAYERS) + \
        "_h_size_" + str(HIDDEN_SIZE) + \
        "_lr_" + str(LEARNING_RATE) + \
        "_batch_" + str(BATCH_SIZE)

WEIGHTS_PATH = "/content/NotaGen/model_weights/weights_notagen_" + NAME + ".pth"    # Path to save weights
LOGS_PATH    = "logs_notagen_"    + NAME + ".txt"                                   # Path to save logs
WANDB_NAME = NAME