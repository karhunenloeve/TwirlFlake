# These are the parameters for the neural network.
# The subnetworks are named by x,y,z.
# Differential indicates how many times the finite difference is applied to the layers.
# 6 Layers has pooling in the forth.

cnnmodel = dict(
    batch_size=128,
    image_size=(500, 3),
    validation_split=0.1,
    l1=0.001,
    l2=0.01,
    epochs=10 ** 5,
    learning_rate=1e-5,
    filters=64,
    kernel_size=3,
    pooling_size=3,
    activation="swish",
    padding="causal",
    dropout_rate=0.5,
    layers_x=42,
    layers_y=42,
    layers_z=42,
    differential_x=1,
    differential_y=1,
    differential_z=1,
    dropouts_x=[],
    dropouts_y=[],
    dropouts_z=[],
    name_x="raw",
    name_y="betti1",
    name_z="betti2",
    max_pool_x=[],
    max_pool_y=[],
    max_pool_z=[],
    avg_pool_x=[],
    avg_pool_y=[],
    avg_pool_z=[],
)

lstmmodel = dict(
    units=32,
    l1=0.001,
    l2=0.01,
    pooling_size=5,
    dropout_rate=0.5,
    return_state=False,
    go_backwards=False,
    stateful=False,
    differential=1,
    layers=22,
    name_a="CuDNNLSTM",
    max_pool=[],
    avg_pool=[],
)

paths = dict(
    general="/home/goku/Dokumente/siemens_powerplant_samples",
    files="/home/goku/Dokumente/siemens_powerplant_samples/powerplant",
    target="/home/goku/Dokumente/siemens_powerplant_samples/powerplant_slidingwindow",
    pershom="/home/goku/Dokumente/siemens_powerplant_samples/powerplant_pershom",
    silhouette="/home/goku/Dokumente/siemens_powerplant_samples/powerplant_silhouette",
    betticurve="/home/goku/Dokumente/siemens_powerplant_samples/powerplant_betticurve",
    heatkernels="/home/goku/Dokumente/siemens_powerplant_samples/powerplant_heatkernels",
    split_analysed="/home/goku/Dokumente/siemens_powerplant_samples/signaldaten_ordner_gruppiert_analysiert",
    split_ordered="/home/goku/Dokumente/siemens_powerplant_samples/signaldaten_ordner_gruppiert_aufgeräumt",
    split="/home/goku/Dokumente/siemens_powerplant_samples/signaldaten_ordner_gruppiert",
    data="/home/goku/Dokumente/TwirlFlake/data/",
    images="/home/goku/Dokumente/TwirlFlake/images/",
    test="/home/goku/Dokumente/TwirlFlake/test/",
)

# These are the categories of the signals.
# Can be completed but is not used for classifications.
# They are used for plotting analysis only.

pptcat = [
    "EKT20",
    "EKT30",
    "MBP",
    "MBR10",
    "MBA26",
    "MBA12",
    "MBY",
    "MBA11",
    "MBL",
    "MKC",
    "BBT",
    "BAT",
    "HBK10",
    "LBA15",
    "LAE20",
    "LBA10",
    "HAH10",
    "HAH20",
    "HAJ50",
    "HAJ60",
    "LAE10",
    "HAH15",
    "HAH10",
    "HAC30",
    "HAC20",
    "HAD30",
    "HAH50",
    "LAE01",
    "HAD50",
    "HAH80",
    "LBA15",
    "HAC50",
    "HAC10",
    "HAD80",
    "HAA10",
    "HAC60",
    "LBA80",
    "LAB60",
    "LAB30",
    "LAB90",
    "LCA40",
    "LCA60",
    "LCA30",
    "LCA70",
    "LBB40",
    "LAH",
    "LBB45",
    "LAF10",
    "LAF01",
    "LAF20",
    "LBC45",
    "10LBC40",
    "10MAA50",
    "10MAA10-20",
    "10LBA20",
    "10MAB10-20",
    "10LBB50",
    "10MAB50",
    "10MAC10-20",
    "10MAC40-45",
    "10MAC50",
    "10MAG",
    "10MKY",
    "10MKC",
    "10BAT",
    "PAB30",
    "PAB20",
    "LCB",
    "LCA10",
    "10LCA20",
    "LBA90",
    "LAC",
]
