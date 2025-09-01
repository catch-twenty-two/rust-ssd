import re
import matplotlib.pyplot as plt
import sys

def parse_log(file_path, keyword):
    """
    Extract epochs and losses from log file lines containing the given keyword.
    Returns two lists: epochs and losses.
    """
    epochs = []
    losses = []

    with open(file_path, "r") as f:
        for line in f:
            if keyword in line:
                # Expecting format like: ...,E:<number>,...,L:<number>,...
                parts = line.strip().split(',')
                e_match = re.search(r"E:([0-9]+)", parts[1])
                l_match = re.search(r"L:([0-9.]+)", parts[3])
                if e_match and l_match:
                    epochs.append(int(e_match.group(1)))
                    losses.append(float(l_match.group(1)))
    return epochs, losses

# File path to your log
log_file = sys.argv[1]

# Parse training and validation logs
train_epochs, train_losses = parse_log(log_file, "Train")
valid_epochs, valid_losses = parse_log(log_file, "Valid")


# Plot both losses
plt.figure(figsize=(10, 6))
plt.plot(train_epochs, train_losses, label="Training Loss", marker='o')
plt.plot(valid_epochs, valid_losses, label="Validation Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()