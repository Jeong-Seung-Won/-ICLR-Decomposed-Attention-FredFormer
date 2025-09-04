from types import SimpleNamespace

__all__ = ["MODEL_ZOO", "CFG_REGISTRY"]

from Foundation_Model_baseline import GRUForecast, LSTMForecast, DLinear
from Foundation_Model_ours_fredformer_based import FredFormer_ours
from Foundation_Model_Fredformer import FredFormer
from Foundation_Model_SparseTSF import SparseTSF
from Foundation_Model_ours_tensor_train import Ours_TT
from Foundation_Model_ours_cp import Ours_CP

MODEL_ZOO = {
    "ours_tt":               Ours_TT
}


ours_tt_cfg = SimpleNamespace(
    model="ours_tt", enc_in=6, seq_len=512, pred_len=90,
    d_model=45, cf_dim=128, cf_depth=2, cf_heads=4, cf_mlp=128,
    cf_head_dim=16, cf_drop=0.1, head_dropout=0.1, rank = [1, 3, 32, 1]
)

# ─── registry ───
CFG_REGISTRY = {
    "ours_tt":               ours_tt_cfg
}
