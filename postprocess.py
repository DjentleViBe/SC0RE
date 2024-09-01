"""Post processing"""
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def plot(lossplot, filename):
    "Plots data and saves figure"
    plt.plot(lossplot)
    plt.suptitle('SupervisedRiffGen')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.savefig(filename, dpi = 200)

    return 0

def decoder_inference(decoder, dummy_in, embedding_layer, pos_enc, mask, seq_lim):
    """Transformer Decoder"""
    for e in range (2, seq_lim):
        embeddings = embedding_layer(dummy_in)
        output_eval = decoder(embeddings + pos_enc, mask)

        next_token_logits = output_eval[:, -1, :]
        probabilities = F.softmax(next_token_logits, dim=-1)

        # Select the next token (using greedy search here)
        next_token = torch.argmax(probabilities, dim=-1).unsqueeze(0)

        # generated_sequence = dummy_in
        dummy_in[0][e] = next_token
    print(dummy_in)
    return dummy_in
