"""One-shot launch remains disabled without independently supplied root release."""
import argparse,json
from authority import read_release,admit_once
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--approved-release-sha256',required=True)
    parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    if args.preflight:
        try:read_release(args.approved_release_sha256)
        except (OSError,ValueError,KeyError):
            print(json.dumps({'status':'DISABLED_NO_ROOT_RELEASE','model_work':False}));return 2
        print(json.dumps({'status':'RELEASE_VALID_NOT_LAUNCHED','model_work':False}));return 0
    admission=admit_once(args.approved_release_sha256)
    from production_run import controller
    from setup_budget import Budget
    result=controller(Budget(admission['started_monotonic']))
    print(json.dumps(result,sort_keys=True));return 0 if result['audit_completed'] else 1
if __name__=='__main__':raise SystemExit(main())

