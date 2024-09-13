"""Defines parameters required for training and inference"""

################################ transformers #################################
MODE            =   2  # 0: train, # 1 : eval, # 2 : both
BACKUP          =   "dec_only_notes_16"
SAVE            =   "dec_only_notes_16A"
START_ID        =   15922
BOS_TRUE        =   1
########## Params ##############
EPOCHS          =   1500
VOCAB_SIZE      =   26420
FFN_HIDDEN      =   3072
MAX_SEQ_LENGTH  =   12
NUM_HEADS       =   12
DROP_PROB       =   0.3
NUM_LAYERS      =   12
D_MODEL         =   720
LEARNING_RATE   =   0.0001
PATCH           =   1
STRIDE          =   1
TRAINING        =   ["CB"]
BATCH           =   950
CONVERGENCE     =   0.0005
TEMPERATURE     =   0.7

EOS             =   26405
BOS             =   26406
BARRE_NOTE      =   26407
BEND_NOTE_1     =   26408
BEND_NOTE_2     =   26409
BEND_NOTE_3     =   26410
BEND_NOTE_4     =   26411
BEND_NOTE_5     =   26412
BEND_NOTE_6     =   26413
BEND_NOTE_7     =   26414
TREM_BAR_1      =   26415
TREM_BAR_2      =   26416
TREM_BAR_3      =   26417
TREM_BAR_4      =   26418
TREM_BAR_5      =   26419
#################################################################################
MEASURE         =   16
