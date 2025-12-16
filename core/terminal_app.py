from rich.console import Console
from core.engine import process_user_input

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.rule import Rule
from rich.text import Text
from rich.align import Align

from main import process_user_input

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
