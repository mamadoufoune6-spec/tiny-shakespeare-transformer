import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

with open(Path(__file__).parent / "data/input.txt", "r", encoding="utf-8") as handle:
    text = handle.read()
    sorted_text = sorted(list(set(text)))
    char_to_int_dict = {ch: i for i, ch in enumerate(sorted_text)}
    int_to_char_dict = {i: ch for i, ch in enumerate(sorted_text)}



def encoder(text):
    # Convert the text to a list of integers using the char_to_int_dict
    return torch.tensor([char_to_int_dict[char] for char in text], dtype=torch.long)

def decoder(integer_representation):
    # Convert the tensor of integers back to text using the int_to_char_dict
    text = ""
    for integer in integer_representation.tolist():
        text += int_to_char_dict[integer]
    return text

assert decoder(encoder(text[:50])) == text[:50] 
data = encoder(text)

def split_data(integer_representation, train_ratio=0.8, val_ratio=0.1):
    total_length = len(integer_representation)
    train_cutoff = int(total_length * train_ratio)
    val_cutoff = int(total_length * (train_ratio + val_ratio))
    train_data = integer_representation[:train_cutoff]
    val_data = integer_representation[train_cutoff:val_cutoff]
    test_data = integer_representation[val_cutoff:]
    return train_data, val_data, test_data

train_data, val_data, test_data = split_data(data)
assert len(train_data) + len(val_data) + len(test_data) == len(data)

def slicing_func(train_data):
    T = 8
    inputs = train_data[:T]
    targets = train_data[1:T + 1]
    return inputs, targets
inputs, targets = slicing_func(train_data)

# I loop over slicing func and make a batch of inputs and targets
def create_batches(data, batch_size=4, seq_length=8):
    inputs = []
    targets = []
    for i in range(0, len(data) - seq_length, seq_length):
        input_seq = data[i:i + seq_length]
        target_seq = data[i + 1:i + seq_length + 1]
        inputs.append(input_seq)
        targets.append(target_seq)
        if len(inputs) == batch_size:
            yield torch.stack(inputs), torch.stack(targets)
            inputs = []
            targets = []
# Test the create_batches function
if __name__ == "__main__":
    print(repr(text[:100]))
    print("Vocabulary size:", len(sorted_text))
    print(data.shape, data.dtype)
    for batch_inputs, batch_targets in create_batches(train_data):
        print("Batch inputs:", batch_inputs)
        print("Batch targets:", batch_targets)
        print("Decoded inputs:", repr(decoder(batch_inputs[0])))
        print("Decoded targets:", repr(decoder(batch_targets[0])))
        break
