"""
app.py - Interactive command-line interface for the Spam Classifier
Usage:
    python app.py                  # interactive mode
    python app.py --demo           # run with built-in demo messages
    python app.py --msg "your msg" # classify a single message inline
"""

import argparse
import sys
from predict import predict

BANNER = r"""
  ____  ____   _    __  __    ____ _        _    ____ ____ ___ _____ ___ _____ ____
 / ___||  _ \ / \  |  \/  |  / ___| |      / \  / ___/ ___|_ _|  ___|_ _| ____|  _ \\
 \___ \| |_) / _ \ | |\/| | | |   | |     / _ \ \___ \___ \| || |_   | ||  _| | |_) |
  ___) |  __/ ___ \| |  | | | |___| |___ / ___ \ ___) |__) | ||  _|  | || |___|  _ <
 |____/|_| /_/   \_\_|  |_|  \____|_____/_/   \_\____/____/___|_|   |___|_____|_| \_\\
"""

DEMO_MESSAGES = [
    "Congratulations! You've won a £1,000 Walmart gift card. Go to http://bit.ly/xxx to claim now.",
    "Hey, are we still on for lunch tomorrow at 1pm?",
    "URGENT! Your mobile number has won our £2000 prize. Call 09061743810 to claim.",
    "Can you send me the notes from today's lecture?",
    "FREE entry to a weekly competition. Text WIN to 80086 now. T&Cs apply.",
    "I'll be home late tonight, don't wait up for dinner.",
    "Your account is about to be suspended. Verify immediately at http://phish.example.com",
    "What time does the movie start?",
]


def print_result(message: str, result: dict):
    """Pretty-print a single prediction result."""
    label      = result['label']
    confidence = result['confidence']
    spam_prob  = result['spam_prob']
    ham_prob   = result['ham_prob']

    bar_len   = 30
    spam_fill = int(bar_len * spam_prob / 100)
    ham_fill  = bar_len - spam_fill

    if label == 'SPAM':
        verdict = f"🚨  SPAM  ({confidence}% confidence)"
        bar     = f"\033[91m{'█' * spam_fill}\033[0m{'░' * ham_fill}"
    else:
        verdict = f"✅  HAM   ({confidence}% confidence)"
        bar     = f"\033[92m{'█' * ham_fill}\033[0m{'░' * spam_fill}"

    print("\n" + "─" * 60)
    print(f"  Message   : {message[:80]}{'...' if len(message) > 80 else ''}")
    print(f"  Verdict   : {verdict}")
    print(f"  Spam prob : {spam_prob:5.1f}%  [{bar}]  Ham: {ham_prob:.1f}%")
    print("─" * 60)


def interactive_mode():
    """Run a continuous interactive classification session."""
    print(BANNER)
    print("  SMS / Email Spam Classifier  |  type 'quit' to exit\n")
    while True:
        try:
            msg = input("  Enter message: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if msg.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break
        if not msg:
            print("  (empty message — please type something)")
            continue

        try:
            result = predict(msg)
            print_result(msg, result)
        except FileNotFoundError as e:
            print(f"\n[ERROR] {e}")
            sys.exit(1)


def demo_mode():
    """Classify a set of built-in demo messages."""
    print(BANNER)
    print("  DEMO MODE — classifying sample messages\n")
    for msg in DEMO_MESSAGES:
        try:
            result = predict(msg)
            print_result(msg, result)
        except FileNotFoundError as e:
            print(f"\n[ERROR] {e}")
            sys.exit(1)


def single_mode(message: str):
    """Classify a single message passed via --msg flag."""
    try:
        result = predict(message)
        print_result(message, result)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Spam Classifier — classify SMS or email messages as spam or ham.'
    )
    parser.add_argument('--demo', action='store_true',
                        help='Run demo mode with sample messages')
    parser.add_argument('--msg',  type=str, default=None,
                        help='Classify a single message passed as a string')
    args = parser.parse_args()

    if args.msg:
        single_mode(args.msg)
    elif args.demo:
        demo_mode()
    else:
        interactive_mode()


if __name__ == '__main__':
    main()
