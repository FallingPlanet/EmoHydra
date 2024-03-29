import torch
from torch.utils.data import DataLoader, Dataset
from FallingPlanet.orbit.models.multimodal.Hydra import HydraTiny
from FallingPlanet.orbit.models.multimodal.Chimera import Chimera
from FallingPlanet.orbit.models.AuTransformer import FPATF_Tiny
from FallingPlanet.orbit.models.BertFineTuneForSequenceClassification import BertFineTuneTiny
from FallingPlanet.orbit.models.DeiTFineTuneForImageClassification import DeitFineTuneTiny
from FallingPlanet.orbit.models.QNetworks import DCQN
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm
from accelerate import Accelerator
import tensorboard
from FallingPlanet.orbit.utils.model_utils import log_model_to_tensorboard
log_dir = "torchlogs/DCQN_logs"  # Specify a directory for the logs
model_name = "DCQNGraph"  # Specify a name for the model in TensorBoard
model = DCQN(n_actions=4)
dummy_input = torch.randn(32,4,84,84)
log_model_to_tensorboard(model = model, dummy_input=dummy_input,log_dir=log_dir,model_name=model_name)