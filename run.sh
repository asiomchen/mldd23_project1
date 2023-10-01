python train_clf.py -d data/activity_data/c1_ECFP_100nM_std.parquet -m models/GRUv3_ECFP_tola/epoch_150.pt
python train_clf.py -d data/activity_data/c1_ECFP_100nM_std.parquet -m models/GRUv3_ECFP_bolek/epoch_150.pt
python train_clf.py -d data/activity_data/c1_ECFP_100nM_std.parquet -m models/GRUv3_ECFP_lolek/epoch_150.pt
python train_clf.py -d data/activity_data/c1_ECFP_100nM_std.parquet -m models/GRUv3_ECFP_uszatek/epoch_150.pt

python train_clf.py -d data/activity_data/c1_klek_100nM_std.parquet -m models/GRUv3_klek_sonic/epoch_200.pt
python train_clf.py -d data/activity_data/c1_klek_100nM_std.parquet -m models/GRUv3_klek_tails/epoch_200.pt
python train_clf.py -d data/activity_data/c1_klek_100nM_std.parquet -m models/GRUv3_klek_eggman/epoch_200.pt
python train_clf.py -d data/activity_data/c1_klek_100nM_std.parquet -m models/GRUv3_klek_knuckles/epoch_200.pt
