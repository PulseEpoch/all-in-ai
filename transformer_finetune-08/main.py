import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset
import logging
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from transformer_gpt2_07.main import GPT2Small
logging.basicConfig(level=logging.WARNING)
import time
import matplotlib.pyplot as plt
import argparse

class GPT2Classifier(GPT2Small):
    def __init__(self, config, num_classes, device=None):
        super().__init__(config, device)
        # replace lm_head with a new one
        self.lm_head = nn.Linear(config['n_embd'], num_classes)
        # freeze all layers
        for param in self.parameters():
            param.requires_grad = False
        # unfreeze the last layer
        for param in self.lm_head.parameters():
            param.requires_grad = True
        for param in self.blocks[-1].parameters():
            param.requires_grad = True
        for param in self.ln_f.parameters():
            param.requires_grad = True

    def forward(self, idx, labels=None):
        logits, _ = super().forward(idx)

        return logits


class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, split, tokenizer, max_length=None, pad_token_id=50256):
        logging.basicConfig(level=logging.WARNING)

        # load dataset from huggingface dataset
        dataset = load_dataset("dair-ai/emotion", split=split)

        self.data = []
        for item in dataset:
            self.data.append({"text": item["text"], "label": item["label"]})

        # Create label mapping
        all_labels = {item["label"] for item in self.data}
        self.label_map = {label: i for i,
                          label in enumerate(sorted(all_labels))}
        self.num_classes = len(self.label_map)

        # Pre-tokenize texts
        self.encoded_texts = [
            tokenizer.encode(item["text"]) for item in self.data
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # Truncate sequences if they are longer than max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # Pad sequences to the longest sequence
        self.encoded_texts = [
            encoded_text + [pad_token_id] *
            (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(
                self.label_map[self.data[index]["label"]], dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # Logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(
                device), target_batch.to(device)

            with torch.no_grad():
                # Logits of last output token
                logits = model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels ==
                                    target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            # New: track examples instead of tokens
            examples_seen += input_batch.shape[0]
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Calculate accuracy after each epoch
        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter)
        print(
            f"Training accuracy: {train_accuracy*100:.2f}% | Validation accuracy: {val_accuracy*100:.2f}%")

        # Collect and print validation predictions vs targets
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            # Use fresh iterator to ensure we start from beginning of val_loader
            for i, (input_batch, target_batch) in enumerate(iter(val_loader)):
                if i >= eval_iter:
                    break
                input_batch = input_batch.to(device)
                logits = model(input_batch)[:, -1, :]
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(target_batch.numpy())
        model.train()

        # Print first 5 examples
        print("Validation examples (Expected vs Actual):")
        for i in range(min(5, len(val_preds))):
            print(
                f"  Example {i+1}: Expected={val_targets[i]}, Actual={val_preds[i]}")

        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen

def predict_emotion(text, model, tokenizer, device, max_length=512):
    model.eval()
    encoded_text = tokenizer.encode(text, max_length=max_length, truncation=True)
    encoded_text = torch.tensor(encoded_text, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(encoded_text)[:, -1, :]
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_label = torch.argmax(probabilities, dim=-1).item()
    return predicted_label, probabilities.cpu().numpy()[0]


def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.",
             label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    # Invisible plot for aligning ticks
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig(f"{label}-plot.pdf")

def main(args):
    start_time = time.time()
    torch.manual_seed(123)
    # 配置参数
    config = {
        'vocab_size': 50257,
        'n_embd': 768,
        'n_layer': 12,
        'n_head': 12,
        'n_ctx': 1024,
        'dropout': 0.0,
        'bos_token_id': 50256,
        'eos_token_id': 50256,
        "qkv_bias": True
    }
    batch_size = 8
    epochs = args.epochs

    if args.device:
        if args.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this system")
        elif args.device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available on this system")
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = EmotionDataset('train', tokenizer)
    val_dataset = EmotionDataset('validation', tokenizer)
    num_classes = train_dataset.num_classes

    model = GPT2Classifier(config, num_classes, device)
    model.to(device)

    if args.mode == 'train':
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=5e-5, weight_decay=0.1)

        train_losses, val_losses, train_accs, val_accs, examples_seen = train(model, train_loader, val_loader, optimizer, device, epochs,
                            eval_freq=100, eval_iter=10)

        torch.save(model.state_dict(), 'emotion_classifier_weights.pth')
        print('Model weights saved to emotion_classifier_weights.pth')

        epochs_tensor = torch.linspace(0, epochs, len(train_losses))
        examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
        plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

        epochs_tensor = torch.linspace(0, epochs, len(train_accs))
        examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
        plot_values(epochs_tensor, examples_seen_tensor,
                    train_accs, val_accs, label="accuracy")

    elif args.mode == 'inference':
        if not args.model_path:
            print("Please specify a model path with --model-path")
            return

        try:
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            print(f"Successfully loaded model from {args.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return

        if args.text:
            predicted_label, probabilities = predict_emotion(args.text, model, tokenizer, device)
            label_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']  # 对应emotion数据集的标签
            print(f"Input text: {args.text}")
            print(f"Predicted emotion: {label_names[predicted_label]}")
            print("Emotion probabilities:")
            for i, prob in enumerate(probabilities):
                print(f"  {label_names[i]}: {prob:.4f}")
        else:
            print("Please provide text for inference with --text")

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Completed in {execution_time_minutes:.2f} minutes.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPT2 Emotion Classifier')
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], default='train',
                        help='Mode: train or inference')
    parser.add_argument('--model-path', type=str, default='emotion_classifier_weights.pth', help='Path to model weights (.pth) for inference')
    parser.add_argument('--text', type=str, help='Text to classify for inference')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'mps'], help='Device to use (cuda, cpu, or mps). Defaults to auto-detect.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs (default: 5)')
    args = parser.parse_args()
    main(args)