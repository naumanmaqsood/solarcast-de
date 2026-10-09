"""Command line entry point.

    python run.py train        download history (cached), train, evaluate
    python run.py forecast     fetch the live weather forecast and predict the next days
    python run.py dashboard    write output/dashboard.html and open it
    python run.py all          forecast + dashboard
    python run.py serve        live dashboard on http://127.0.0.1:8000 that keeps updating
    python run.py site         forecast + static site in site/ (what GitHub Pages publishes)
    python run.py ask "..."    ask the AI assistant one question
    python run.py chat         interactive chat with the AI assistant
"""
import argparse
import json
import webbrowser


def _assistant(action) -> None:
    import anthropic
    from solarcast import agent
    try:
        action(agent)
    except TypeError as e:  # the SDK raises this when no credentials are configured
        if "authentication" not in str(e):
            raise
        print("No Anthropic credentials found. Set ANTHROPIC_API_KEY to use the assistant.")
    except anthropic.AuthenticationError:
        print("The Anthropic API rejected the credentials. Check ANTHROPIC_API_KEY.")
    except anthropic.APIConnectionError:
        print("Could not reach the Anthropic API. Check the internet connection.")
    except anthropic.AnthropicError as e:
        print(f"Assistant unavailable: {e}")
        print("Set ANTHROPIC_API_KEY to use the assistant.")


def main() -> None:
    parser = argparse.ArgumentParser(description="SolarCast DE")
    parser.add_argument("command", choices=["train", "forecast", "dashboard", "all", "serve", "site", "ask", "chat"])
    parser.add_argument("question", nargs="?", help='question for "ask"')
    parser.add_argument("--refresh", action="store_true", help="re-download the training history")
    parser.add_argument("--no-open", action="store_true", help="do not open the dashboard in the browser")
    parser.add_argument("--port", type=int, default=8000, help='port for "serve"')
    parser.add_argument("--interval", type=float, default=10, help='minutes between data refreshes for "serve" and "site"')
    args = parser.parse_args()

    if args.command == "train":
        from solarcast.data import load_history
        from solarcast.model import train_and_evaluate
        metrics = train_and_evaluate(load_history(refresh=args.refresh))
        print(json.dumps({k: v for k, v in metrics.items() if k != "monthly_mae_gw"}, indent=2))

    if args.command in ("forecast", "all", "site"):
        from solarcast.forecast import make_forecast
        for day in make_forecast()["days"]:
            print(f"{day['date']}: {day['energy_gwh']['p50']:.0f} GWh "
                  f"(P10-P90 {day['energy_gwh']['p10']:.0f}-{day['energy_gwh']['p90']:.0f}), "
                  f"peak {day['peak_gw']} GW at {day['peak_time']}")

    if args.command in ("dashboard", "all"):
        from solarcast.dashboard import build_dashboard
        path = build_dashboard()
        print(f"Dashboard written to {path}")
        if not args.no_open:
            webbrowser.open(path.as_uri())

    if args.command == "site":
        from solarcast.dashboard import build_site
        print(f"Site written to {build_site(args.interval)}")

    if args.command == "serve":
        from solarcast.server import serve
        serve(args.port, args.interval, on_ready=None if args.no_open else webbrowser.open)

    if args.command == "ask":
        if not args.question:
            parser.error('ask needs a question, e.g. python run.py ask "When should I charge my EV tomorrow?"')
        _assistant(lambda agent: print(agent.ask(args.question)))

    if args.command == "chat":
        _assistant(lambda agent: agent.chat())


if __name__ == "__main__":
    main()
