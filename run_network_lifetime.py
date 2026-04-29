#!/usr/bin/env python
"""Wrapper to run network lifetime evaluation with warnings suppressed."""

import warnings
import os

# Suppress numpy warnings about experimental MINGW build
warnings.filterwarnings('ignore', category=RuntimeWarning)
os.environ['PYTHONWARNINGS'] = 'ignore'

# Now import and run
if __name__ == "__main__":
    import sys
    from evaluate_network_lifetime import main
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Network Lifetime Evaluation')
    parser.add_argument('--runs', type=int, default=100, help='Number of runs per method (default: 100)')
    parser.add_argument('--plot', action='store_true', default=True, help='Generate plots')
    parser.add_argument('--no-plot', dest='plot', action='store_false', help='Skip plot generation')
    
    args = parser.parse_args()
    
    # Run main evaluation
    try:
        results = main(num_runs=args.runs, plot=args.plot)
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
