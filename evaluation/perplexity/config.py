# Directory containing the input .abc files to be processed
INPUT_INTERLEAVED_FOLDER = ''                                      

# Configurations for small model (110M Parameters)
INFERENCE_WEIGHTS_PATH = ''

PATCH_STREAM = True               # Streaming mode                            
PATCH_SIZE = 16                    # Patch size
PATCH_LENGTH = 2048                # Patch Length
CHAR_NUM_LAYERS = 3                # Number of layers in the decoder (Character-Level Decoder)
PATCH_NUM_LAYERS = 12              # Number of layers in the encoder (Patch-Level Decoder)
HIDDEN_SIZE = 768                  # Hidden Size (n_embd)
PATCH_SAMPLING_BATCH_SIZE = 0     # Νo subsampling is performed
 