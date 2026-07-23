# Formal Data Pipeline

The formal predictor input is based on the 0408 Regret step-level transition artifact. That artifact already implements the session split and consecutive same-item step aggregation used by the regret project.

Run:

```bash
cd "/root/autodl-tmp/0626/0626 Predictor"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./01_data/prepare_formal_data.sh
```

Default outputs:

- `01_data/processed/raw_rqkmeans/dense_item_features.npy`: dense-item feature matrix used as the embedding store.
- `01_data/processed/raw_rqkmeans/dense_item2sid.npy`: dense item id to semantic-ID path.
- `01_data/processed/raw_rqkmeans/dense2orig_item_id.npy`: dense item id to original item id.
- `01_data/processed/raw_rqkmeans/orig2dense_item_id.npy`: original item id to dense item id.
- `01_data/processed/formal_data_manifest.json`: the formal data manifest used by the training runbook.
- `01_data/logs/formal_preprocess.log`: one-shot preprocessing log.

By default the full train/val/test transition parquet is not materialized again. `FutureIterableDataset` reads the 0408 Regret transition parquet directly and converts each step to predictor fields at load time. This avoids creating a second multi-GB `future_data` copy.

Optional modes:

- `COPY_TRANSITIONS=1 ./01_data/prepare_formal_data.sh`: copy the 0408 transition root into `01_data/processed/regret_current_data` when there is enough disk.
- `BUILD_FUTURE_DATA=1 ./01_data/prepare_formal_data.sh`: explicitly materialize `01_data/processed/future_data/{train,val,test}`. This is not recommended on the current disk because the full converted train split is much larger than the compact source transition artifact.
