import threading
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from app.config.settings import MODEL_NAME


class ModelLoader:
    """
    负责加载 tokenizer 和 model。
    这里使用懒加载：第一次请求 /chat 时才真正加载模型。
    """

    def __init__(self):
        self.model_name = MODEL_NAME
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._loaded = False
        self._lock = threading.Lock()

    def load(self):
        """
        加载模型。
        加锁是为了避免多个请求同时进来时重复加载模型。
        """
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            print(f"[ModelLoader] Loading model: {self.model_name}")
            print(f"[ModelLoader] Device: {self.device}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model.to(self.device)
            self.model.eval()

            self._loaded = True
            print("[ModelLoader] Model loaded successfully.")

    def get(self):
        self.load()
        return self.tokenizer, self.model, self.device, self.model_name


model_loader = ModelLoader()
