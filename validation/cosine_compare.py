"""Author-only exact-source comparison; not part of the production PR."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET


root = Path.cwd()
receipts = root / 'receipts'
scope = sys.argv[1]
outcomes = {}


def run_suite(label, checkout, *args):
  junit = receipts / f'{label}.xml'
  command = [sys.executable, '-m', 'pytest', '-q', *args,
             f'--junitxml={junit}']
  with (receipts / f'{label}.log').open('w') as log:
    result = subprocess.run(command, cwd=root / checkout, stdout=log,
                            stderr=subprocess.STDOUT, check=False)
  output = (receipts / f'{label}.log').read_text()
  print(output, flush=True)
  tree = ET.parse(junit)
  cases = list(tree.iter('testcase'))
  outcomes[label] = {
      'command': command, 'returncode': result.returncode,
      'tests': len(cases),
      'failures': [c.attrib['name'] for c in cases if c.find('failure') is not None],
      'errors': [c.attrib['name'] for c in cases if c.find('error') is not None],
      'skips': [c.attrib['name'] for c in cases if c.find('skipped') is not None],
  }
  return result.returncode


for checkout in ('head', 'baseline'):
  source = subprocess.check_output(
      [sys.executable, '-c',
       'import jax,optax; print(jax.__version__); print(optax.__file__)'],
      cwd=root / checkout, text=True,
  )
  print(checkout, source, flush=True)
  assert str(root / checkout / 'optax') in source, source

head_status = run_suite('head', 'head', scope, '-n', '2')
base_status = run_suite('baseline', 'baseline', scope, '-n', '2')
shutil.copyfile(root / 'head/optax/losses/_regression_test.py',
                root / 'baseline/optax/losses/_regression_test.py')
negative_status = run_suite(
    'negative', 'baseline', 'optax/losses/_regression_test.py',
    '-k', 'CosineLowPrecisionTest',
)
(receipts / 'summary.json').write_text(json.dumps(outcomes, indent=2))
assert head_status == 0, outcomes['head']
assert base_status == 0, outcomes['baseline']
assert negative_status == 1 and not outcomes['negative']['errors']
required_failures = {
    'test_values_and_gradients_match_float64_reference0',
    'test_values_and_gradients_match_float64_reference1',
    'test_values_and_gradients_match_float64_reference2',
    'test_values_and_gradients_match_float64_reference3',
    'test_values_and_gradients_match_float64_reference8',
    'test_values_and_gradients_match_float64_reference9',
    'test_values_and_gradients_match_float64_reference10',
    'test_values_and_gradients_match_float64_reference11',
    'test_float16_parameters_take_real_sgd_steps',
    'test_low_precision_epsilon_does_not_underflow_before_clipping',
}
assert required_failures <= set(outcomes['negative']['failures'])
assert outcomes['negative']['tests'] == 41, outcomes['negative']
