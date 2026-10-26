"""Use grade-balanced group error after observing normal-heavy CV selection."""
import sys,time,subprocess,shutil
from common import B,O
started=time.monotonic()
while '"stage": "score_training_complete"' not in (O/'score-pipeline.log').read_text():
 if time.monotonic()-started>3600:raise TimeoutError('Initial score run not complete')
 time.sleep(2)
shutil.copy2(O/'score-training.json',O/'pre-balanced-selection-reference.json')
subprocess.run([sys.executable,str(B/'tools/integrated-five/train_scores.py')],check=True)
print('FINAL_SCORE_REFINEMENT_COMPLETE',flush=True)
