from rich.console import Console
from core.engine import process_user_input

console = Console()

def main():
    while True:
        user_input = console.input("You: ")
        if user_input.lower() == "exit":
            break
        result = process_user_input(user_input, ...)
        console.print(result)

if __name__ == "__main__":
    main()
