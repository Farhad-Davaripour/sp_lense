from driver import admitted_main_seconds as v2_admit, launch as v2_launch
V2_MAIN_SECONDS=v2_admit(V2_SETUP_STARTED,rate=6.77,cap=12,reserve=1.5,maximum=5400,prior_incurred_units=V2_PRIOR_INCURRED)
print('CURATED_V2_MAIN_SECONDS',V2_MAIN_SECONDS,flush=True)
V2_DRIVER=v2_launch(V2_ROOT,V2_ROOT/'model',V2_MAIN_SECONDS)
print('CURATED_V2_DRIVER_STARTED',json.dumps(V2_DRIVER),flush=True)
