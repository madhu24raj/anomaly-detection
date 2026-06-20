 DETECTION RESULTS
  Contacts evaluated : 50000
  True anomalous     : 950 (1.9%)
  Flagged by detector: 1053

  Precision          : 0.876
  Recall             : 0.972
  F1                 : 0.922
  AUROC              : 1.000
  Average Precision  : 0.990

  Confusion: TP=923 FP=130 FN=27 TN=48920

 Concatenating chunks...
  431,861,746 pings, 50,000 contacts, columns: ['entity_id', 'lat', 'lon', 'timestamp', 'vessel_type', 'sog_knots', 'is_anomalous', 'anomaly_type']
  Memory usage: 37.58 GB

Extracting features...
  Auto-detected ping interval : 5.0 min
  Auto-estimated contamination: 0.019 (1.9% of contacts flagged as structurally suspicious)
  Computing pairwise proximity across 8640 timestamps × 50000 contacts...
  Features cached → features_full.parquet
  Feature matrix: 50000 contacts × 29 features  [18856.1s]
  Using interval=5.0 min, contamination=0.019

Running detectors...
  [1/5] Score: speed teleport (rule-based)...
  [2/5] Score: transmission blackout (rule-based)...
  [3/5] Score: slow operations near sensitive areas (IF + LOF)...
  [4/5] Score: at-sea co-location (IF + LOF)...
  [5/5] Score: erratic close-range pursuit (IF + LOF)...
  Done  [18864.7s]


DETECTION RESULTS
  Contacts evaluated : 50000
  True anomalous     : 950 (1.9%)
  Flagged by detector: 1053

  Precision          : 0.876
  Recall             : 0.972
  F1                 : 0.922
  AUROC              : 1.000
  Average Precision  : 0.990

  Confusion: TP=923 FP=130 FN=27 TN=48920


RECALL BY ANOMALY TYPE
          anomaly_type     n  flagged  recall  mean_score_teleport  mean_score_blackout  mean_score_slow_stratum  mean_score_colocation  mean_score_erratic_pursuit
          ais_spoofing   200      200   1.000                1.000                0.000                    0.205                  0.122             0.356
       illegal_fishing   350      350   1.000                1.000                0.000                    0.316                  0.098             0.286
         transshipment   200      196   0.980                0.827                0.000                    0.342                  0.559             0.255
         dark_activity   150      136   0.907                0.000                0.737                    0.235                  0.178             0.091
aggressive_maneuvering    50       41   0.820                0.901                0.000                    0.329                  0.415             0.260
                  none 49050      130   0.003                0.000                0.000                    0.096                  0.094             0.078


TOP 20 CONTACTS BY ANOMALY SCORE
           ensemble_score  ensemble_mean  score_teleport  score_blackout  score_slow_stratum  score_colocation  score_erratic_pursuit  true_anomalous      true_type
entity_id
33432               1.200          0.330           0.000           1.000               0.335             0.071                  0.043   1  dark_activity
21262               1.200          0.270           0.000           1.000               0.067             0.032                  0.053   1  dark_activity
46815               1.200          0.284           0.000           1.000               0.069             0.054                  0.095   1  dark_activity
18632               1.200          0.259           0.000           1.000               0.022             0.044                  0.030   1  dark_activity
14332               1.200          0.321           0.000           1.000               0.331             0.059                  0.016   1  dark_activity
8208                1.200          0.297           0.000           1.000               0.069             0.114                  0.101   1  dark_activity
3973                1.200          0.421           0.000           1.000               0.454             0.379                  0.071   1  dark_activity
37394               1.200          0.324           0.000           1.000               0.392             0.015                  0.015   1  dark_activity
16412               1.200          0.312           0.000           1.000               0.074             0.223                  0.062   1  dark_activity
35079               1.200          0.260           0.000           1.000               0.041             0.010                  0.047   1  dark_activity
23605               1.200          0.290           0.000           1.000               0.062             0.100                  0.086   1  dark_activity
36953               1.200          0.280           0.000           1.000               0.030             0.086                  0.085   1  dark_activity
8550                1.200          0.294           0.000           1.000               0.200             0.023                  0.045   1  dark_activity
26295               1.200          0.293           0.000           1.000               0.096             0.115                  0.052   1  dark_activity
35210               1.200          0.291           0.000           1.000               0.114             0.048                  0.095   1  dark_activity
40141               1.200          0.273           0.000           1.000               0.098             0.019                  0.050   1  dark_activity
47528               1.200          0.295           0.000           1.000               0.140             0.029                  0.108   1  dark_activity
29640               1.200          0.251           0.000           1.000               0.026             0.011                  0.019   1  dark_activity
21087               1.200          0.270           0.000           1.000               0.059             0.038                  0.054   1  dark_activity
40883               1.200          0.306           0.000           1.000               0.200             0.086                  0.042   1  dark_activity


Scores written → scores_full.csv

Total runtime: 18865.2s
