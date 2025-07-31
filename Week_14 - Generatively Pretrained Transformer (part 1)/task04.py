from torch.utils.data import TensorDataset
import torch
def main():
    
    filepath = "../DATA/shakespeare.txt"
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    
    
    first_thousand = text[:1000]
    first_thousand_vocab = ''.join(sorted(set(text)))
    index_to_letter_thousand = dict(enumerate(first_thousand_vocab))
    thousand_letter_to_index = {v:k for k,v in index_to_letter_thousand.items()}
    encoded_thousand = [(thousand_letter_to_index[letter]) for letter in first_thousand]
    
    result = torch.tensor(text)
    print(result.shape)
    print(result)
    
    

if __name__ == "__main__":
    main()