from kserve import Model, ModelServer,model_server, InferRequest, InferOutput, InferResponse
from cldm.hack import disable_verbosity, enable_sliced_attention
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from omegaconf import OmegaConf
from typing import Dict, Union
import cv2
from PIL import Image
import io
from io import BytesIO
import base64
import numpy as np
from run_inference import inference_single_image
import argparse
import torch

class Any_Door(Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.load_model()
        self.ready=True

    def load_model(self):
        save_memory = False
        disable_verbosity()
        if save_memory:
            enable_sliced_attention()
        config = OmegaConf.load('./configs/inference.yaml')
        model_ckpt =  config.pretrained_model
        model_config = config.config_file

        model = create_model(model_config ).cpu()
        model.load_state_dict(load_state_dict(model_ckpt, location='cuda'))
        model = model.cuda()
        self.ddim_sampler = DDIMSampler(model)
    
    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        fg_img = self.ori_img_input(payload["instances"][0]["fg_img_base64"])
        fg_mask = self.mask_input(payload["instances"][0]["fg_mask_base64"])
        bg_img = self.ori_img_input(payload["instances"][0]["bg_img_base64"])
        bg_mask = self.mask_input(payload["instances"][0]["bg_mask_base64"])
        gen_image = inference_single_image(fg_img, fg_mask, bg_img.copy(), bg_mask)
        Image.fromarray(gen_image.astype(np.uint8)).save('./examples/TestDreamBooth/GEN/test.png')
        torch.cuda.empty_cache()
        return {
            'img_base64':self.img_2_base64(Image.fromarray(gen_image.astype(np.uint8)))
        }

    def img_2_base64(self,image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue())
        return img_str


    
    def ori_img_input(self,str_base64):
        raw_img_data = base64.b64decode(str_base64)
        pil_image = Image.open(io.BytesIO(raw_img_data)).convert('RGB')
        # pil_image.save('results/tmp.png')
        image = np.array(pil_image)
        # Convert RGB to BGR
        # image = image[:, :, ::-1].copy()
        return image
    
    def mask_input(self,str_base64):
        raw_mask_data = base64.b64decode(str_base64)
        pil_mask = Image.open(io.BytesIO(raw_mask_data)).convert('HSV')
        mask = np.array(pil_mask)
        # Convert RGB to BGR
        mask = mask[:, :, -1].copy()
        mask = mask / 255
        return mask

parser = argparse.ArgumentParser(parents=[model_server.parser])
parser.add_argument(
    "--model_name", help="The name that the model is served under.", default="custom-model"
)

args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = Any_Door("aigc_anydoor")
    ModelServer().start([model]
                        )