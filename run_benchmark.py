import argparse

from npbench.infrastructure import (Benchmark, generate_framework, LineCount,
                                    Test, utilities as util)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b",
                        "--benchmark",
                        type=str,
                        nargs="?",
                        required=True)
    parser.add_argument("-f",
                        "--framework",
                        type=str,
                        nargs="?",
                        default="numpy")
    parser.add_argument("-p",
                        "--preset",
                        choices=['S', 'M', 'L', 'paper'],
                        nargs="?",
                        default='S')
    parser.add_argument("-m", "--mode", type=str, nargs="?", default="main")
    parser.add_argument("-v",
                        "--validate",
                        type=util.str2bool,
                        nargs="?",
                        default=True)
    parser.add_argument("-r", "--repeat", type=int, nargs="?", default=10)
    parser.add_argument("-t",
                        "--timeout",
                        type=float,
                        nargs="?",
                        default=200.0)
    parser.add_argument("-s",
                        "--save-strict-sdfg",
                        type=util.str2bool,
                        nargs="?",
                        default=False)
    parser.add_argument("-l",
                        "--load-strict-sdfg",
                        type=util.str2bool,
                        nargs="?",
                        default=False)
    parser.add_argument('--golden-cache', help='Reuse input/NumPy result bundles in this directory')
    parser.add_argument('--require-golden', action='store_true', help='Fail on a cache miss; never run the reference')
    parser.add_argument('--prepare-golden', action='store_true', help='Prepare the golden bundle and exit')
    parser.add_argument('--lifecycle', action='store_true', help='Record initialization, fresh-process and same-process host-to-host calls')
    parser.add_argument('--fresh-process-runs', type=int, default=1)
    parser.add_argument('--implementation', help='Select a single lifecycle implementation label')
    parser.add_argument('--run-dir', help='Parent directory for isolated lifecycle runs')
    args = vars(parser.parse_args())
    if args['repeat'] < 1 or args['fresh_process_runs'] < 0 or args['timeout'] <= 0:
        parser.error('Repeat/timeout must be positive and fresh-process-runs nonnegative')
    if args['require_golden'] and not (args['golden_cache'] or args['lifecycle'] or args['prepare_golden']):
        parser.error('--require-golden needs --golden-cache or --lifecycle')
    if args['implementation'] and not args['lifecycle']:
        parser.error('--implementation currently requires --lifecycle')
    if args['prepare_golden']:
        from npbench.infrastructure.golden import load_or_create
        import json
        _, event = load_or_create(Benchmark(args['benchmark']), args['preset'], generate_framework('numpy'),
                                  args['golden_cache'] or '.cache/goldens', required=args['require_golden'])
        print(json.dumps(event, indent=2))
        raise SystemExit(0)
    if args['lifecycle']:
        from npbench.infrastructure.lifecycle import run
        raise SystemExit(run(args))

    # print(args)

    bench = Benchmark(args["benchmark"])
    frmwrk = generate_framework(args["framework"],
                                save_strict=args["save_strict_sdfg"],
                                load_strict=args["load_strict_sdfg"])
    numpy = generate_framework("numpy")
    lcount = LineCount(bench, frmwrk, numpy)
    lcount.count()
    test = Test(bench, frmwrk, numpy)
    test.run(args["preset"], args["validate"], args["repeat"], args["timeout"],
             golden_cache=args["golden_cache"], require_golden=args["require_golden"])
