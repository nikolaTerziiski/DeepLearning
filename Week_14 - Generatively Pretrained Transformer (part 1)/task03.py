from torch.utils.data import TensorDataset

def main():
    
    filepath = "../DATA/shakespeare.txt"
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
        
    vocab = ''.join(sorted(set(text)))
    
    index_to_letter = dict(enumerate(vocab))
    
    letter_to_index = {v:k for k,v in index_to_letter.items()}
    hello = 'hi therre'

 
    ecnoded_hello = [(letter_to_index[letter]) for letter in hello]
    print(f"Encoding the text {hello}: {ecnoded_hello}")
    decoded_hello = [(index_to_letter[number]) for number in ecnoded_hello]
    print(''.join(decoded_hello))
    
    

if __name__ == "__main__":
    main()