import glob
import math
import torch
from utils import *
from config import *
from transformers import GPT2Config

class PerplexityCalculator:

    def __init__(self, model: NotaGenLMHeadModel, device: torch.device):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.patchilizer = Patchilizer()

    def compute(self, abc_text: str) -> float:
        # Encode the ABC notation into patches
        id_patches = self.patchilizer.encode_train(abc_text)
        # Create tensors
        file_bytes = torch.tensor(id_patches, dtype=torch.long, device=self.device)
        file_masks = torch.ones(file_bytes.size(0), dtype=torch.long, device=self.device)

        input_patches = file_bytes.unsqueeze(0) # (1, num_patches, patch_size)
        input_masks = file_masks.unsqueeze(0) # (1, num_patches)
        # Compute loss and perplexity
        with torch.no_grad():
            output = self.model(input_patches, input_masks)
            loss = output.loss
        num_tokens = int(input_masks.sum().item())
        nll = loss.item() * num_tokens    
        return nll, num_tokens

def main():
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load base model
    patch_cfg = GPT2Config(
        num_hidden_layers=PATCH_NUM_LAYERS,
        max_length=PATCH_LENGTH,
        max_position_embeddings=PATCH_LENGTH,
        n_embd=HIDDEN_SIZE,
        num_attention_heads=HIDDEN_SIZE // 64,
        vocab_size=1,
    )
    byte_cfg = GPT2Config(
        num_hidden_layers=CHAR_NUM_LAYERS,
        max_length=PATCH_SIZE + 1,
        max_position_embeddings=PATCH_SIZE + 1,
        n_embd=HIDDEN_SIZE,
        num_attention_heads=HIDDEN_SIZE // 64,
        vocab_size=128,
    )

    model = NotaGenLMHeadModel(encoder_config=patch_cfg, decoder_config=byte_cfg)
    checkpoint = torch.load(INFERENCE_WEIGHTS_PATH, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()

    calc = PerplexityCalculator(model, device)

    total_nll = 0.0
    total_tokens = 0
    files = sorted(glob.glob(f"{INPUT_INTERLEAVED_FOLDER}/*.abc"))
    for path in files:
        name = path.rsplit('/',1)[-1].rsplit('.',1)[0]
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()

        nll, num_tokens = calc.compute(text)
        ppl = math.exp(nll / num_tokens)

        # Print per song, tokens, perplexity
        print(f"{name}: tokens={num_tokens}, perplexity={ppl:.4f}")

        total_nll    += nll
        total_tokens += num_tokens

    # Global perplexity
    global_ppl = math.exp(total_nll / total_tokens)
    print(f"\nGlobal Perplexity over {len(files)} songs: {global_ppl:.4f}")

if __name__ == "__main__":
    main()