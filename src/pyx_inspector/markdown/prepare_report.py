import argparse
import pickle
from pathlib import Path
from jinja2 import Template


    
def main():
    parser = argparse.ArgumentParser(description="Prepare a markdown report from a pickle file")
    parser.add_argument("results_pkl", type=str, help="Path to the pickle file containing results")
    parser.add_argument("-o", "--outfile", type=str, required="True", help="Path to the output markdown file")
    args = parser.parse_args()

    results_pkl = Path(args.results_pkl)
    outfile = Path(args.outfile)

    if not results_pkl.exists():
        raise FileNotFoundError(f"Pickle file {results_pkl} not found")

    template_file = Path(__file__).resolve().parent / "templates" / "report_template.md"

    with open(results_pkl, "rb") as f:
        data = pickle.load(f)

    with open(template_file, "r") as f:
        template = Template(f.read())
        report = template.render(**data)

    with open(outfile, "w") as f:
        f.write(report)

    print(f"Report written to {outfile}")
    

    

    
     

