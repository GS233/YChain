import torch
from transformers import  AutoModel,AutoTokenizer,AutoConfig
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from torch.mps import empty_cache
import torch

class LLM():
    name = ''
    def __init__(self,name):
        self.name = name

    def llm_type(self):
        return 
    def load_model(self):
        return
    def load_model_multi_device(model_name_or_path):
        return
    def call(self):
        return



class GLM(LLM):

    name = 'CHATGLM'
    
    max_token: int = 2048
    defult_temperature: float = 0.8
    top_p = 0.9
    history_len: int = 1024
    
    model: object = None
    tokenizer: object = None
    
    def __init__(self,name,max_token,history_len,defult_temperature,top_p):
        super().__init__(name)
        self.max_token = max_token
        self.history_len=history_len
        self.defult_temperature=defult_temperature
        self.top_p=top_p


    def llm_type(self) -> str:
        return self.name
            
    def load_model(self,model_name_or_path=None,device="cpu",quant='half'):
        model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)
        if device == 'cpu':
            self.model = AutoModel.from_pretrained(model_name_or_path, config=model_config, trust_remote_code=True).eval()
        else:
            if quant == 'full':
                self.model = AutoModel.from_pretrained(model_name_or_path, config=model_config, trust_remote_code=True).half().cuda().eval()
            if quant == 'half':
                self.model = AutoModel.from_pretrained(model_name_or_path, config=model_config, trust_remote_code=True).half().cuda().eval()

    def load_model_multi_device(model_name_or_path):
        # 暂时不需要
        return
    
    def call(self,prompt:str,history,stop: Optional[List[str]] = None):
        response, history = self.model.chat(
                    self.tokenizer,prompt,
                    history=history[-self.history_len:] if self.history_len > 0 else [],
                    max_length=self.max_token,temperature=self.defult_temperature,
                    top_p=self.top_p)
        return response, history