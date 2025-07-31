def main():
    filepath = "../DATA/shakespeare.txt"
    with open(filepath, "r") as f:
        text = f.read()
        
    print(f"Total length (in characters): {len(text):,}")

    print("\nFirst 1000 characters:")
    print(text[:1000])

if __name__ == "__main__":
    main()
