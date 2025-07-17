from rich.console import Console
console = Console()

def info(msg: str):
    console.print(f"[INFO] {msg}", style="bold cyan")

def success(msg: str):
    console.print(f"[SUCCESS] {msg}", style="bold green")

def error(msg: str):
    console.print(f"[ERROR] {msg}", style="bold red")
