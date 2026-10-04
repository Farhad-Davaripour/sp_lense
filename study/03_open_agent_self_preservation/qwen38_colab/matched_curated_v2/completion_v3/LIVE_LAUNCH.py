from controller import launch
C3_CONTROLLER=launch(C3_ROOT,C3_ROOT/'model')
print('COMPLETION_V3_CONTROLLER_STARTED',json.dumps(C3_CONTROLLER),flush=True)
