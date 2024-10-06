"""Defines parameters required for training and inference"""

################################ transformers #################################
# 0: train, 
# 1 : eval
# 2 : both 
# 3 : load from file
MODE                =   1
BACKUP              =   "dec_only_notes_28"
SAVE                =   "dec_only_notes_28B"
START_ID            =   15900
BOS_TRUE            =   1
EOS_TRUE            =   0
TEST_CRITERIA       =   2
# 0 - Vanilla
# 1 - Disregard any note > BARRE_NOTE
# 2 - n bars of generation, n = TEST_TRIES
# 3 - Same as above + last note of every bar is fed to firt note of next bar
# 4 - Same as above + loop until no repeatitive notes > BOS 
TEST_TRIES          =   2
# 1 - Greedy Search
# 2 - Beam Seaarch
# 3 - Sampling
PREDICTION_CRITERIA =   3
########## Params ##############
EPOCHS          =   1500
SAVE_EVERY      =   100
VOCAB_SIZE      =   26421
FFN_HIDDEN      =   1024
MAX_SEQ_LENGTH  =   12
NUM_HEADS       =   16
DROP_PROB       =   0.3
NUM_LAYERS      =   24
D_MODEL         =   720
SCHEDULER       =   0
SCHEDULER_SIZE  =   30
LEARNING_RATE   =   0.0001
PATCH           =   1
STRIDE          =   1
TRAINING        =   ["CB"]
BATCH           =   880
CONVERGENCE     =   0.0005
TEMPERATURE     =   1.5

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
