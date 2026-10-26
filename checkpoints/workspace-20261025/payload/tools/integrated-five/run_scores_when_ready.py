import time,subprocess,sys
from common import B,O,read,status
started=time.monotonic()
while True:
 try:ready=len(read(O/'team-extraction.json'))==57
 except (OSError,ValueError):ready=False
 if ready:break
 if time.monotonic()-started>7200:raise TimeoutError('Team extraction did not finish; no training or poweroff performed')
 time.sleep(10)
for stage in ['prepare_scores.py','train_scores.py','compare_return_neural.py','make_report.py','verify_results.py']:
 status('running_stage',script=stage)
 subprocess.run([sys.executable,str(B/'tools/integrated-five'/stage)],check=True)
status('all_training_and_integrity_checks_completed')
