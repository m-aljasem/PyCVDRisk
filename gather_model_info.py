
import inspect
import importlib
import pkgutil
import re
import sys
import pandas as pd
from cvd_risk.core.base import RiskModel
import cvd_risk.models

def get_model_classes():
    """Discover all RiskModel subclasses in the package."""
    models = []
    
    # helper to check if a class is a concrete RiskModel
    def is_concrete_risk_model(obj):
        return (
            inspect.isclass(obj) 
            and issubclass(obj, RiskModel) 
            and obj is not RiskModel
        )

    # Walk through the packages
    path = cvd_risk.models.__path__
    prefix = cvd_risk.models.__name__ + "."

    for module_info in pkgutil.walk_packages(path, prefix):
        try:
            module = importlib.import_module(module_info.name)
            for name, obj in inspect.getmembers(module):
                if is_concrete_risk_model(obj):
                    #Avoid importing imported base classes from other modules
                    if obj.__module__ == module.__name__:
                         models.append(obj)
        except ImportError as e:
            print(f"Could not import {module_info.name}: {e}")

    # Remove duplicates based on class name
    unique_models = {m.__name__: m for m in models}
    return list(unique_models.values())

def parse_docstring(docstring):
    """Parse docstring to extract description, parameters, and population."""
    if not docstring:
        return "", "", "", ""

    lines = [line.strip() for line in docstring.split('\n')]
    
    # Short description is usually the first non-empty line
    description = ""
    for line in lines:
        if line:
            description = line
            break

    # Search for duration
    duration = "Unknown"
    duration_match = re.search(r'(\d+)-year', docstring, re.IGNORECASE)
    if duration_match:
        duration = duration_match.group(1) + " years"
    elif "lifetime" in docstring.lower():
        duration = "Lifetime"
    
    # Search for population/validated for
    population = "General population" # Default
    pop_match = re.search(r'validated for.*?(ages.*?)(\.|\n)', docstring, re.IGNORECASE | re.DOTALL)
    if not pop_match:
         pop_match = re.search(r'population.*?(ages.*?)(\.|\n)', docstring, re.IGNORECASE | re.DOTALL)
    
    if pop_match:
        population = pop_match.group(1).strip()
    
    # Count parameters
    # Look for Parameters section
    param_count = 0
    in_params = False
    
    for line in lines:
        line_clean = line.strip()
        if line_clean == 'Parameters':
            in_params = True
            continue
            
        if in_params:
            if set(line_clean) == {'-'} and len(line_clean) > 3:
                # This is the underline for Parameters, ignore it
                continue
            if line_clean.startswith('Returns') or line_clean.startswith('Examples') or line_clean.startswith('Notes'):
                in_params = False
                break
            
            # Count parameters
            if ':' in line_clean:
                 # Ensure it's not a nested thing or list item, simplistic check
                 # Usually "param : type"
                 if not line_clean.startswith('#'):
                     param_count += 1

            
    return description, population, param_count, duration

def main():
    models = get_model_classes()
    data = []

    for model_cls in models:
        try:
            # Instantiate to access attributes if needed, though class attr is better
            # Using class attributes directly
            model_name = getattr(model_cls, 'model_name', model_cls.__name__)
            
            docstring = model_cls.__doc__
            
            # Fallback to module docstring if class docstring is missing or generic
            if not docstring or "Abstract base class" in docstring:
                 module_name = model_cls.__module__
                 if module_name in sys.modules:
                     module_doc = sys.modules[module_name].__doc__
                     if module_doc:
                         docstring = module_doc
            
            desc, pop, params, duration = parse_docstring(docstring)
            
            # Refine params if 0, check __init__ or calculate ?
            # Inspecting calculate signature might be misleading as it takes 'patient'
            # So docstring parsing is best effort.
            
            data.append({
                "Model Name": model_name,
                "Short Description": desc,
                "Target Population": pop,
                "Number of Parameters": params,
                "Duration": duration
            })
        except Exception as e:
            print(f"Error processing {model_cls.__name__}: {e}")

    df = pd.DataFrame(data)
    df = df.sort_values("Model Name")
    
    
    # Manual markdown formatting to avoid tabulate dependency
    headers = df.columns
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        print("| " + " | ".join(str(val) for val in row) + " |")
    
    # Also save to file
    with open("model_summary_table.md", "w") as f:
        f.write("# CVD Risk Models Summary\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for _, row in df.iterrows():
            f.write("| " + " | ".join(str(val) for val in row) + " |\n")

if __name__ == "__main__":
    main()
