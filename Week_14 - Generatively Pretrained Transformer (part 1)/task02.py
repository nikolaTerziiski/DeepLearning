def main():
    
    filepath = "../DATA/shakespeare.txt"
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
        
    vocab = ''.join(sorted(set(text)))
    print(f"Vocabulary: \n {vocab}")
    print(f'Vocabulary size: {len(vocab)}')


if __name__ == "__main__":
    main()