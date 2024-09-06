"""Defines parameters required for ttraining and inference"""

################################ transformers #################################
MODE            =   1  # 0: train, # 1 : eval, # 2 : both
BACKUP          =   "dec_only_notes_8"
START_ID        =   7500
########## Params ##############
EPOCHS          =   1000
VOCAB_SIZE      =   26412
FFN_HIDDEN      =   1024
MAX_SEQ_LENGTH  =   12
NUM_HEADS       =   2
DROP_PROB       =   0.3
NUM_LAYERS      =   2
D_MODEL         =   60
LEARNING_RATE   =   0.01
PATCH           =   1
STRIDE          =   1
TRAINING        =   ["CB"]
BATCH           =   890
CONVERGENCE     =   0.0005
TEMPERATURE     =   2

EOS             =   26405
BOS             =   26406
BARRE_NOTE      =   26407
#################################################################################
MEASURE         =   16