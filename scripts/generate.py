import torch
from music_generator.tokenization import REMITokenizer
from music_generator.model import CrossModalMusicTransformer

def main():
    tokenizer = REMITokenizer()
    model = CrossModalMusicTransformer(vocab_size=tokenizer.vocab_size)
    checkpoint = torch.load("music_generator_model.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    generated = model.generate(max_length=128)
    tokenizer.tokens_to_midi(generated, "generated_output.mid")
    print("Generated music saved as generated_output.mid")

if __name__ == "__main__":
    main()
