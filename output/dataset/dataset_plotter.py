import numpy as np
from matplotlib import pyplot as plt

# Load training data set
dataset = np.load('output/dataset/dataset.npy')
labels = np.load('output/dataset/labels.npy')
uniq_labels = np.unique(labels)

# zcr_feat, 0 
# rmse_feat, 1
# mfcc_feat, 2-14
# spectral_centroid_feat, 15 
# spectral_rolloff_feat,  16 
# spectral_bandwidth_feat  17

zcr = dict()
specc = dict()
specr = dict()
specb = dict()
rmse = dict()
mfcc = dict()
for label in uniq_labels:
    ind = np.where(labels==label)
    zcr[label[6:]] = dataset[ind,0][0,:]
    rmse[label[6:]] = dataset[ind,1][0,:]
    specc[label[6:]] = dataset[ind,15][0,:]
    specr[label[6:]] = dataset[ind,16][0,:]
    specb[label[6:]] = dataset[ind,17][0,:]
    mfcc[label[6:]] = dataset[ind,2:14].squeeze().mean(axis=0)

# for k,v in zcr.items():
#     plt.plot(v)
# plt.title("zero crossing rate")
# plt.legend(uniq_labels)
# plt.show()


fig = plt.figure()

ax1 = fig.add_subplot(231)
# g = sns.catplot(ax=ax1,data=zcr)
sns.stripplot(data=zcr, ax=ax1)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
plt.ylabel("zero crossing rate")

ax2 = fig.add_subplot(232)
# g = sns.catplot(ax=ax2,data=rmse)
sns.stripplot(data=rmse, ax=ax2)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
plt.ylabel("rmse")

ax3 = fig.add_subplot(233)
# g = sns.catplot(ax=ax3,data=specc)
sns.stripplot(data=specc, ax=ax3)
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
plt.ylabel("spectral centroid")


ax4 = fig.add_subplot(234)
# g = sns.catplot(ax=ax3,data=specc)
sns.stripplot(data=specr, ax=ax4)
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
plt.ylabel("spectral rolloff")


ax5 = fig.add_subplot(235)
# g = sns.catplot(ax=ax3,data=specc)
sns.stripplot(data=specb, ax=ax5)
ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45)
plt.ylabel("spectral bandwidth")


ax6 = fig.add_subplot(236)
# g = sns.catplot(ax=ax3,data=specc)
sns.stripplot(data=mfcc, ax=ax6)
ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45)
plt.ylabel("mfcc coeff mean")

plt.tight_layout()