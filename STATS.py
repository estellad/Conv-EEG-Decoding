# Test AUC 10-fold CV for C2 C3
# Test AUC avg across 6 subjects for C2, C3, trained on C1
# Inaki Data:
# 10-fold CV C2: 85.06%
#            C3: ~87%
# C1C2: LDA no correction:  67.01%
#       LDA corrected: 79.06%
#       Baseline: 79.42%

# C1C3: LDA no correction: 65.77%
#       LDA corrected: 80.25%
#       Baseline: ~80%

# C2C3: LDA no correction: 79.93%
#       LDA corrected: 82.97%
#       Baseline: ~80%

import os
os.environ['PYTHONHASHSEED'] = '0'

import numpy as np
np.random.seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import random
random.seed(9001)

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Polygon

# trans_data12 = [i *100 for i in [0.923287892, 0.907181845, 0.915812966, 0.920867952, 0.922992122,
#                   0.515731112, 0.505093379, 0.512706919, 0.505783107, 0.504456706,
#                   0.532673567, 0.546649543, 0.564981995, 0.540454786, 0.548487825,
#                   0.726036015, 0.699011821, 0.703834762, 0.689339729, 0.709063983,
#                    0.737707919, 0.749973847, 0.736871012, 0.736844858, 0.725834292,
#                    0.684218685, 0.669749897, 0.660086813, 0.665073377, 0.657167218]]
#
# trans_data13 = [i*100 for i in [0.979806943, 0.978005431, 0.982549542, 0.979215402, 0.978543196,
#                    0.480690537, 0.489104859, 0.522378517, 0.509207161, 0.501381074,
#                    0.647296063, 0.642400568, 0.639077719, 0.644404424, 0.636262175,
#                    0.517693044, 0.499314022, 0.498783361, 0.505552536, 0.502342678,
#                    0.711143322, 0.716941308, 0.728410131, 0.734335266, 0.731944868,
#                    0.572665399, 0.567360625, 0.567360625, 0.602517266, 0.588829947]]
#
# trans_data23 = [i*100 for i in [0.983544406,	0.982603318, 0.983732623, 0.984162835, 0.983087306,
#                    0.675549872,	0.670383632, 0.657058824, 0.676777494, 0.656138107,
#                    0.572975852,	0.588994014, 0.570718344, 0.575157265, 0.58887987,
#                    0.737671818, 0.78587145,  0.74934638,  0.730113122, 0.771271777,
#                    0.766681925, 0.765486726, 0.769199471, 0.751398637, 0.757628929,
#                    0.616554899, 0.608672806, 0.625262736, 0.628615754, 0.621859674]]
trans_data12 = [i *100 for i in [0.918028555, 0.508754244, 0.546649543, 0.705457262, 0.737446386, 0.667259198]] # After taking subject-wise average for above
trans_data13 = [i *100 for i in [0.979624103, 0.50055243,  0.64188819,  0.504737128, 0.724554979, 0.579746772]]
trans_data23 = [i *100 for i in [0.983426098, 0.667181586, 0.579345069, 0.754854909, 0.762079137, 0.620193174]]

########## 10 fold ###############

# ten_fold2 =[i *100 for i in [0.974431818, 0.996685606, 0.998106061, 1,           1,           1,    0.998579545, 0.999047619, 0.999047619, 0.997152349,
#                       0.732954545, 0.716392513, 0.829059829, 0.786894587, 0.799430199, 0.825568182, 0.769318182, 0.834090909, 0.819318182, 0.704545455,
#                       0.805387205, 0.721212121, 0.656228956, 0.660942761, 0.724915825, 0.72020202,  0.761279461, 0.666666667, 0.803367003, 0.793939394,
#                       0.755488835, 0.785472973, 0.830046948, 0.834459459, 0.859984985, 0.849690373, 0.79545881,  0.866929429, 0.905405405, 0.882507508,
#                       0.79749831,  0.828571429, 0.867816092, 0.895199459, 0.868154158, 0.902711864, 0.876949153, 0.801355932, 0.889830508, 0.91220339,
#                       0.698366013, 0.740522876, 0.727868852, 0.737581699, 0.766666667, 0.728104575, 0.773529412, 0.742810458, 0.85147541,  0.819016393]] #mean: 82.81073664666668
#
# ten_fold3 =[i*100 for i in [0.999565217, 1,           1,           1,           1,    	   1,           1,           1,           1,           1,
#                       0.612068966, 0.656449553, 0.691717301, 0.760446571, 0.69284802,  0.827630317, 0.824752159, 0.806194125, 0.856321839, 0.865581098,
#                       0.725897436, 0.677692308, 0.682051282, 0.626410256, 0.627948718, 0.704405738, 0.618340164, 0.661794872, 0.62602459,  0.681282051,
#                       0.786525974, 0.808823529, 0.80952381,  0.86038961,  0.865400271, 0.934871099, 0.910984848, 0.880140693, 0.87761194,  0.915332429,
#                       0.779745861, 0.735725309, 0.773148148, 0.788781431, 0.890817901, 0.759645062, 0.80477474,  0.851837524, 0.859953704, 0.811728395,
#                       0.693059299, 0.721548822, 0.728619529, 0.721698113, 0.787061995, 0.807407407, 0.806734007, 0.857142857, 0.778301887, 0.820875421]] # mean: 81.08939032666667

ten_fold2 = [82.81073664666668]
ten_fold3 = [81.08939032666667]

Inaki_ten_fold2 = [85.06]
Inaki_ten_fold3 = [87]
######### 10 fold ##############

Inaki_LDA_no1 = [67.01]
Inaki_LDA_yes1 = [79.06]
Inaki_LDA_no2 = [65.77]
Inaki_LDA_yes2 = [80.25]
Inaki_LDA_no3 = [79.93]
Inaki_LDA_yes3 = [82.97]


# baselineC2 = [i*100 for i in [0.910166438, 0.887768546, 0.893952838, 0.894893926, 0.882982442,
#                               0.601310484, 0.627148769, 0.616696732, 0.607332343, 0.619986205,
#                               0.596963058, 0.592858402, 0.589257385, 0.577598147, 0.608874115,
#                               0.722785772, 0.733401484, 0.723519698, 0.700899059, 0.720872323,
#                               0.681687415, 0.70964536,  0.686682707, 0.698242494, 0.695313317,
#                               0.659130839, 0.670447499, 0.651353865, 0.660681067, 0.660164324]]
# baselineC3 = [i*100 for i in [0.938694846, 0.942620526, 0.939770375, 0.9513592,   0.936086688,
#                               0.569769821, 0.561994885, 0.564680307, 0.528299233, 0.568670077,
#                               0.496550325, 0.513316761, 0.498579545, 0.512403612, 0.512631899,
#                               0.776086563, 0.782143874, 0.790453263, 0.765162693, 0.77629365,
#                               0.634014851, 0.641109755, 0.63846506,  0.623664937, 0.629946089,
#                               0.527474727, 0.518816935, 0.530027024, 0.5280002,   0.528750876]]

baselineC2 = [i*100 for i in [0.893952838, 0.614494907, 0.593110221, 0.720295667, 0.694314259, 0.660355519]]  # After taking subject-wise average for above
baselineC3 = [i*100 for i in [0.941706327, 0.558682864, 0.506696429, 0.778028009, 0.633440138, 0.526613953]]

Inaki_base2 = [79.42]
Inaki_base3 = [80]

listdat12 =       Inaki_base2    + baselineC2         + trans_data12           +Inaki_LDA_no1        +Inaki_LDA_yes1    +ten_fold2        +Inaki_ten_fold2
listcategory = ['LDA_Baseline']+['CNN_Baseline']*6 +['CNN_No_Correction']*6+['LDA_No_Correction']+['LDA_Correction']+['CNN_10_fold'] +['LDA_10_fold']
listsubj_cat = ['no']+    ['s1']+['s2']+['s3']+['s4']+['s5']+['s6']  +  ['s1']+['s2']+['s3']+['s4']+['s5']+['s6']    +['no']*4

listdat13 =       Inaki_base3    + baselineC3         + trans_data13           +Inaki_LDA_no2        +Inaki_LDA_yes2    +ten_fold3      +Inaki_ten_fold3
listdat23 =       Inaki_base3    + baselineC3         + trans_data23           +Inaki_LDA_no3        +Inaki_LDA_yes3    +ten_fold3      +Inaki_ten_fold3



data12 = {'Mean Test AUC Score': listdat12, 'Type': listcategory, 'Subject': listsubj_cat}
data12df = pd.DataFrame(data=data12)

data13 = {'Mean Test AUC Score': listdat13, 'Type': listcategory, 'Subject': listsubj_cat}
data13df = pd.DataFrame(data=data13)

data23 = {'Mean Test AUC Score': listdat23, 'Type': listcategory, 'Subject': listsubj_cat}
data23df = pd.DataFrame(data=data23)


# E1E2
import seaborn as sns
palette_set = ['red','red','purple','purple','purple','orange','orange']
sns.catplot(x='Type', y='Mean Test AUC Score', jitter=False, data=data12df, palette=palette_set, zorder=1).set_xticklabels(rotation=40)

# calc medians
lda_bsmean2 = Inaki_base2[0]
bsmean2 = data12df.loc[data12df["Type"] == 'CNN_Baseline'].mean()
nocorrectionmean12 = data12df.loc[data12df["Type"] == 'CNN_No_Correction'].mean()
lda_nomean1 = Inaki_LDA_no1[0]
lda_yesmean1 = Inaki_LDA_yes1[0]
tenfoldmean2 = data12df.loc[data12df["Type"] == 'CNN_10_fold'].mean()
lda_10fmean2 = Inaki_ten_fold2[0]

x = plt.gca().axes.get_xlim()
lda_bs_location = (-0.3,0.3)
bs_location = (0.7, 1.3)
nocorrection_location = (1.7, 2.3)
lda_no_location = (2.7, 3.3)
lda_yes_location = (3.7, 4.3)
tenfold_location = (4.7, 5.3)
lda_10f_location = (5.7, 6.3)

plt.plot(lda_bs_location, len(lda_bs_location) * [lda_bsmean2], sns.xkcd_rgb["pale red"])
plt.plot(bs_location, len(bs_location) * [bsmean2], sns.xkcd_rgb["pale red"])
plt.plot(nocorrection_location, len(nocorrection_location) * [nocorrectionmean12], sns.xkcd_rgb["purple"])
plt.plot(lda_no_location, len(lda_no_location) * [lda_nomean1], sns.xkcd_rgb["purple"])
plt.plot(lda_yes_location, len(lda_yes_location) * [lda_yesmean1], sns.xkcd_rgb["purple"])

plt.plot(tenfold_location, len(tenfold_location) * [tenfoldmean2], sns.xkcd_rgb["orange"])
plt.plot(lda_10f_location, len(lda_10f_location) * [lda_10fmean2], sns.xkcd_rgb["orange"])

plt.title('E1E2 Pair')
sns.pointplot(x="Type", y="Mean Test AUC Score", hue='Subject', data=data12df, zorder=100, color='grey', hue_order=['s1', 's2', 's3', 's4', 's5', 's6'], ci=95)
plt.legend(loc='lower right')
plt.show()


# E1E3
import seaborn as sns
palette_set = ['red','red','purple','purple','purple','orange','orange']
sns.catplot(x='Type', y='Mean Test AUC Score', jitter=False, data=data13df, palette=palette_set, zorder=1).set_xticklabels(rotation=40)

# calc medians
lda_bsmean3 = Inaki_base3[0]
bsmean3 = data13df.loc[data13df["Type"] == 'CNN_Baseline'].mean()
nocorrectionmean13 = data13df.loc[data13df["Type"] == 'CNN_No_Correction'].mean()
lda_nomean2 = Inaki_LDA_no2[0]
lda_yesmean2 = Inaki_LDA_yes2[0]
tenfoldmean3 = data13df.loc[data13df["Type"] == 'CNN_10_fold'].mean()
lda_10fmean3 = Inaki_ten_fold3[0]

x = plt.gca().axes.get_xlim()
lda_bs_location = (-0.3,0.3)
bs_location = (0.7, 1.3)
nocorrection_location = (1.7, 2.3)
lda_no_location = (2.7, 3.3)
lda_yes_location = (3.7, 4.3)
tenfold_location = (4.7, 5.3)
lda_10f_location = (5.7, 6.3)

plt.plot(lda_bs_location, len(lda_bs_location) * [lda_bsmean3], sns.xkcd_rgb["pale red"])
plt.plot(bs_location, len(bs_location) * [bsmean3], sns.xkcd_rgb["pale red"])
plt.plot(nocorrection_location, len(nocorrection_location) * [nocorrectionmean13], sns.xkcd_rgb["purple"])
plt.plot(lda_no_location, len(lda_no_location) * [lda_nomean2], sns.xkcd_rgb["purple"])
plt.plot(lda_yes_location, len(lda_yes_location) * [lda_yesmean2], sns.xkcd_rgb["purple"])

plt.plot(tenfold_location, len(tenfold_location) * [tenfoldmean3], sns.xkcd_rgb["orange"])
plt.plot(lda_10f_location, len(lda_10f_location) * [lda_10fmean3], sns.xkcd_rgb["orange"])

plt.title('E1E3 Pair')
sns.pointplot(x="Type", y="Mean Test AUC Score", hue='Subject', data=data13df, zorder=100, color='grey', hue_order=['s1', 's2', 's3', 's4', 's5', 's6'], ci=95)
plt.legend(loc='lower right')
plt.show()



# E2E3
import seaborn as sns
palette_set = ['red','red','purple','purple','purple','orange','orange']
sns.catplot(x='Type', y='Mean Test AUC Score', jitter=False, data=data23df, palette=palette_set, zorder=1).set_xticklabels(rotation=40)

# calc medians
lda_bsmean3 = Inaki_base3[0]
bsmean3 = data23df.loc[data23df["Type"] == 'CNN_Baseline'].mean()
nocorrectionmean23 = data23df.loc[data23df["Type"] == 'CNN_No_Correction'].mean()
lda_nomean3 = Inaki_LDA_no3[0]
lda_yesmean3 = Inaki_LDA_yes3[0]
tenfoldmean3 = data23df.loc[data23df["Type"] == 'CNN_10_fold'].mean()
lda_10fmean3 = Inaki_ten_fold3[0]

x = plt.gca().axes.get_xlim()
lda_bs_location = (-0.3,0.3)
bs_location = (0.7, 1.3)
nocorrection_location = (1.7, 2.3)
lda_no_location = (2.7, 3.3)
lda_yes_location = (3.7, 4.3)
tenfold_location = (4.7, 5.3)
lda_10f_location = (5.7, 6.3)

plt.plot(lda_bs_location, len(lda_bs_location) * [lda_bsmean3], sns.xkcd_rgb["pale red"])
plt.plot(bs_location, len(bs_location) * [bsmean3], sns.xkcd_rgb["pale red"])
plt.plot(nocorrection_location, len(nocorrection_location) * [nocorrectionmean23], sns.xkcd_rgb["purple"])
plt.plot(lda_no_location, len(lda_no_location) * [lda_nomean3], sns.xkcd_rgb["purple"])
plt.plot(lda_yes_location, len(lda_yes_location) * [lda_yesmean3], sns.xkcd_rgb["purple"])

plt.plot(tenfold_location, len(tenfold_location) * [tenfoldmean3], sns.xkcd_rgb["orange"])
plt.plot(lda_10f_location, len(lda_10f_location) * [lda_10fmean3], sns.xkcd_rgb["orange"])

plt.title('E2E3 Pair')
sns.pointplot(x="Type", y="Mean Test AUC Score", hue='Subject', data=data23df, zorder=100, color='grey', hue_order=['s1', 's2', 's3', 's4', 's5', 's6'], ci=95)
plt.legend(loc='lower right')
plt.show()




