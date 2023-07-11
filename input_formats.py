import numpy as np


############## CNN and LSTM Inputs formats (t' = 60) ########################

 # with no padding for multivariate

def input_2D_no_padding():
    return np.array([[0.64948039, 0.28236301],
                     [0.64880177, 0.28236301],
                     [0.64870708, 0.28236301],
                     [0.6485177, 0.28236301],
                     [0.6466239, 0.28193493],
                     [0.64484057, 0.28150685],
                     [0.64408304, 0.28056507],
                     [0.64373585, 0.28236301],
                     [0.64373585, 0.28236301],
                     [0.64359381, 0.28150685],
                     [0.6440357, 0.28107877],
                     [0.6444618, 0.28150685],
                     [0.64417773, 0.28193493],
                     [0.64294676, 0.28107877],
                     [0.64271003, 0.28150685],
                     [0.64364116, 0.28107877],
                     [0.64624514, 0.28056507],
                     [0.65017478, 0.28107877],
                     [0.65497242, 0.28150685],
                     [0.6601646, 0.28056507],
                     [0.66575133, 0.28107877],
                     [0.67045428, 0.28013699],
                     [0.67493628, 0.28013699],
                     [0.68020737, 0.28056507],
                     [0.68361622, 0.28056507],
                     [0.68795619, 0.28056507],
                     [0.68943967, 0.28013699],
                     [0.68999203, 0.28056507],
                     [0.68934498, 0.28013699],
                     [0.68721445, 0.28013699],
                     [0.68574675, 0.28056507],
                     [0.68610973, 0.28013699],
                     [0.68861902, 0.28107877],
                     [0.69240663, 0.28056507],
                     [0.69543672, 0.28056507],
                     [0.69818274, 0.28056507],
                     [0.70160737, 0.28056507],
                     [0.70533185, 0.28013699],
                     [0.70765176, 0.28013699],
                     [0.70935619, 0.28013699],
                     [0.70954557, 0.2797089],
                     [0.70899321, 0.27919521],
                     [0.70724144, 0.28013699],
                     [0.70525294, 0.28056507],
                     [0.70320132, 0.28013699],
                     [0.70116548, 0.2797089],
                     [0.69922433, 0.2797089],
                     [0.69852993, 0.2797089],
                     [0.69922433, 0.2797089],
                     [0.69983982, 0.2797089],
                     [0.69857728, 0.27919521],
                     [0.69621002, 0.27876712],
                     [0.6932115, 0.27919521],
                     [0.69093893, 0.2797089],
                     [0.68767212, 0.27919521],
                     [0.68551003, 0.2797089],
                     [0.68312699, 0.27919521],
                     [0.68170663, 0.2797089],
                     [0.68143834, 0.28013699],
                     [0.6823379, 0.27919521]])

 

# with pre padding for multivariate

def input_2D_pre_padding():
    return np.array([[0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.0, 0.0],
                     [0.64948039, 0.28236301],
                     [0.64880177, 0.28236301],
                     [0.64870708, 0.28236301],
                     [0.6485177, 0.28236301],
                     [0.6466239, 0.28193493],
                     [0.64484057, 0.28150685],
                     [0.64408304, 0.28056507],
                     [0.64373585, 0.28236301],
                     [0.64373585, 0.28236301],
                     [0.64359381, 0.28150685],
                     [0.6440357, 0.28107877],
                     [0.6444618, 0.28150685],
                     [0.64417773, 0.28193493],
                     [0.64294676, 0.28107877],
                     [0.64271003, 0.28150685],
                     [0.64364116, 0.28107877],
                     [0.64624514, 0.28056507],
                     [0.65017478, 0.28107877],
                     [0.65497242, 0.28150685],
                     [0.6601646, 0.28056507],
                     [0.66575133, 0.28107877],
                     [0.67045428, 0.28013699],
                     [0.67493628, 0.28013699],
                     [0.68020737, 0.28056507],
                     [0.68361622, 0.28056507],
                     [0.68795619, 0.28056507],
                     [0.68943967, 0.28013699],
                     [0.68999203, 0.28056507],
                     [0.68934498, 0.28013699],
                     [0.68721445, 0.28013699],
                     [0.68574675, 0.28056507],
                     [0.68610973, 0.28013699],
                     [0.68861902, 0.28107877],
                     [0.69240663, 0.28056507],
                     [0.69543672, 0.28056507],
                     [0.69818274, 0.28056507],
                     [0.70160737, 0.28056507],
                     [0.70533185, 0.28013699],
                     [0.70765176, 0.28013699],
                     [0.70935619, 0.28013699],
                     [0.70954557, 0.2797089],
                     [0.70899321, 0.27919521],
                     [0.70724144, 0.28013699],
                     [0.70525294, 0.28056507],
                     [0.70320132, 0.28013699],
                     [0.70116548, 0.2797089],
                     [0.69922433, 0.2797089],
                     [0.69852993, 0.2797089],
                     [0.69922433, 0.2797089],
                     [0.69983982, 0.2797089],
                     [0.69857728, 0.27919521],
                     [0.69621002, 0.27876712],
                     [0.6932115, 0.27919521],
                     [0.69093893, 0.2797089],
                     [0.68767212, 0.27919521],
                     [0.68551003, 0.2797089],
                     [0.68312699, 0.27919521],
                     [0.68170663, 0.2797089],
                     [0.68143834, 0.28013699],
                     [0.6823379, 0.27919521]])
# 

############## FFNN Inputs formats ########################

##### (t' = 137)



##### (t' = 60)

def input_dense():
    return np.array([0.0,0.0,0.69979247,0.27234589,0.69936636,0.27191781
,0.69924011,0.27191781,0.70009232,0.27234589,0.70066046,0.27191781
,0.70054999,0.27148973,0.70088141,0.27054795,0.70061312,0.27148973
,0.69976091,0.27054795,0.69957153,0.27097603,0.69857728,0.27097603
,0.69827743,0.27097603,0.69823008,0.27054795,0.69905073,0.27054795
,0.6995084,0.27011986,0.70083406,0.27011986,0.70111813,0.27011986
,0.70111813,0.27011986,0.70154424,0.27011986,0.7024438,0.27011986
,0.70274365,0.27011986,0.70316976,0.26917808,0.70387993,0.26917808
,0.70482684,0.26960616,0.70547389,0.26875,0.70684689,0.26960616
,0.70771489,0.26917808,0.70855132,0.26960616,0.70956135,0.26960616
,0.71022418,0.26875,0.71083967,0.26917808,0.71136046,0.26917808
,0.71240205,0.26960616,0.71292285,0.26875,0.71323849,0.26875
,0.71382241,0.26832192,0.71363303,0.26875,0.71334896,0.26875
,0.71375928,0.26875,0.71415382,0.26875,0.7139171,0.26832192
,0.71423273,0.26832192,0.71480087,0.26780822,0.71543214,0.26832192
,0.71576356,0.26875,0.71565309,0.26832192,0.71628435,0.26875
,0.71655264,0.26832192,0.71659999,0.26780822,0.71647374,0.26738014
,0.71680515,0.26780822,0.71656843,0.26738014,0.71664733,0.26695205
,0.71664733,0.26695205,0.71647374,0.26738014,0.71607919,0.26738014
,0.71576356,0.26738014,0.71529011,0.26780822,0.71480087,0.26780822
,0.71432742,0.26695205,0.71385397,0.26738014,0.71353834,0.26738014
,0.71282816,0.26738014,0.71290707,0.26695205,0.71254409,0.26738014
,0.71211798,0.26738014,0.71178657,0.26695205,0.7116761,0.26780822
,0.71140781,0.26601027,0.71126577,0.26652397,0.71129734,0.26695205
,0.71088701,0.26695205,0.7109817,0.26695205,0.71145515,0.26652397
,0.71186548,0.26652397,0.71306489,0.26652397,0.71325427,0.26738014
,0.7129702,0.26652397,0.71311223,0.26695205,0.71188126,0.26695205
,0.71117108,0.26695205,0.71050825,0.26738014,0.70964026,0.26652397
,0.70869335,0.26601027,0.70788849,0.26601027,0.70732034,0.26652397
,0.70653126,0.26695205,0.70605781,0.26601027,0.70610515,0.26652397
,0.70605781,0.26601027,0.70623141,0.26652397,0.70629453,0.26652397
,0.70670486,0.26652397,0.70700471,0.26652397,0.7077938,0.26601027
,0.7087407,0.26652397,0.7092615,0.26652397,0.70954557,0.26652397
,0.71044512,0.26652397,0.71068185,0.26558219,0.71110795,0.26601027
,0.71121843,0.26601027,0.71101326,0.26601027,0.71060294,0.26601027
,0.71027152,0.26558219,0.70921415,0.26601027,0.70945088,0.26652397
,0.70869335,0.26601027,0.70889852,0.26601027,0.70803052,0.26515411
,0.70842506,0.26652397,0.70878804,0.26601027,0.70889852,0.26601027
,0.70960869,0.26601027,0.71039778,0.26601027,0.71027152,0.26558219
,0.70984542,0.26515411,0.71008214,0.26558219,0.71036621,0.26558219
,0.71031887,0.26601027,0.71079232,0.26558219,0.71121843,0.26601027
,0.71140781,0.26601027,0.7121022,0.26558219,0.71259144,0.26558219
,0.71366459,0.26601027,0.71300176,0.26558219,0.71281238,0.26558219
,0.71233893,0.26601027,0.71268613,0.26558219,0.71273347,0.26601027
,0.71259144,0.26558219,0.71240205,0.26558219,0.71243362,0.26601027
,0.71273347,0.26601027,0.7130491,0.26601027])

