from models.table_transformer import TableTransformerDetector

def main():
    detector = TableTransformerDetector()

    print("Model ready:", detector.is_ready())
    print("Last error:", detector.get_last_error())

if __name__ == "__main__":
    main()