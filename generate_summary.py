from CVAE import CVAE_dropout
import config
from torchinfo import summary

model_CVAE_v1 = CVAE_dropout(img_size=512)
# model_CVAE_v1.name = "model_CVAE_v1"
summary(
    model_CVAE_v1,
    input_size=[config.BATCH_SIZE, 3, config.IMAGE_SIZE, config.IMAGE_SIZE],
)
