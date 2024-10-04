"""Defines parameters required for training and inference"""

################################ transformers #################################
MODE            =   3  # 0: train, # 1 : eval, # 2 : both # 3 : load from
BACKUP          =   "dec_only_notes_2_Mac"
SAVE            =   "dec_only_notes_2A_Mac"
START_ID        =   9430
BOS_TRUE        =   1
TEST_CRITERIA   =   2
TEST_TRIES      =   10
########## Params ##############
EPOCHS          =   300
SAVE_EVERY      =   10
VOCAB_SIZE      =   26421
FFN_HIDDEN      =   2048
MAX_SEQ_LENGTH  =   12
NUM_HEADS       =   4
DROP_PROB       =   0.3
NUM_LAYERS      =   4
D_MODEL         =   720
LEARNING_RATE   =   0.0001
PATCH           =   1
STRIDE          =   1
TRAINING        =   ["CB"]
BATCH           =   182
CONVERGENCE     =   0.0005
TEMPERATURE     =   2.0

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
DEAD_NOTE       =   26420
#################################################################################
MEASURE         =   8
