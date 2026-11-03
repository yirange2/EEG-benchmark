"""Test the actual task adapters against native LSL inlets."""
from pathlib import Path
import ast
import json
import socket
import struct
import sys
import subprocess
from types import SimpleNamespace
import unittest
import uuid

from pylsl import StreamInlet, local_clock, resolve_byprop

REPO = Path(__file__).resolve().parent
TASKS = {
    'eoec': ('01-EOEC/EOEC_task.py', [11, 12]),
    'ssvep': ('02-SSVEP/ssvep_task.py', [12, 15, 20, 1]),
    'n170': ('03-Face/n170_task.py', [1, 40, 41, 80, 101, 140, 141, 180, 201, 202]),
    'flanker': ('04-Flanker/flanker_task.py', [11, 12, 21, 22, 111, 211, 112, 212, 121, 221, 122, 222]),
}

def task_class(path, phase, enabled, udp=False, port=12340):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_experiment')
    cls = next(n for n in fn.body if isinstance(n, ast.ClassDef) and n.name == 'TriggerPort')
    ns = {'args': SimpleNamespace(send_lsl_markers=enabled, send_parallel_triggers=False,
                                 send_galea_markers=udp, galea_marker_port=port,
                                 lsl_stream_name='GaleaPilotTest-' + uuid.uuid4().hex),
          'Path': Path, 'sys': sys, '__file__': str(path), 'program_phase': phase}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), ns)
    return ns['TriggerPort']

class MarkerIntegration(unittest.TestCase):
    def test_udp_import_without_lsl_dependency(self):
        subprocess.run([
            sys.executable, '-c',
            "import sys; sys.modules['pylsl'] = None; "
            "from galea_pilot_markers import GaleaUdpMarkers; "
            "markers = GaleaUdpMarkers(12340); markers.send(0); markers.close()"
        ], cwd=REPO, check=True, capture_output=True, text=True)

    def test_native_galea_udp_protocol(self):
        for name,(relative,codes) in TASKS.items():
            with self.subTest(task=name), socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as receiver:
                receiver.bind(('127.0.0.1',0))
                receiver.settimeout(1)
                port=receiver.getsockname()[1]
                trigger=task_class(REPO/relative,'main',False,True,port)()
                try:
                    for code in codes:
                        trigger.send(code)
                        trigger.reset()
                        payload,_=receiver.recvfrom(64)
                        self.assertEqual(len(payload),8)
                        self.assertEqual(struct.unpack('>d',payload)[0],float(code))
                    receiver.settimeout(0.02)
                    with self.assertRaises(socket.timeout):
                        receiver.recvfrom(64)
                finally:
                    trigger.close()

    def test_original_task_codes_and_timestamps(self):
        for name, (relative, codes) in TASKS.items():
            with self.subTest(task=name):
                trigger = task_class(REPO / relative, 'main', True)()
                stream_name = trigger.lsl.outlet.get_info().name()
                found = resolve_byprop('name', stream_name, timeout=3)
                self.assertEqual(len(found), 1, f'LSL discovery failed for {name}')
                inlet = StreamInlet(found[0])
                try:
                    inlet.open_stream(timeout=3)
                    self.assertTrue(trigger.lsl.outlet.wait_for_consumers(3))
                    before = local_clock()
                    for code in codes:
                        trigger.send(code)
                        trigger.reset()
                    after = local_clock()
                    received = []
                    for expected in codes:
                        sample, stamp = inlet.pull_sample(timeout=3)
                        self.assertIsNotNone(sample)
                        self.assertEqual(sample, [expected])
                        self.assertGreaterEqual(stamp, before)
                        self.assertLessEqual(stamp, after)
                        received.append(sample[0])
                    trigger.close()
                    self.assertEqual(inlet.pull_sample(timeout=0.05), (None, None))
                    print(json.dumps({'task':name,'received_codes':received,'timestamp_check':'PASS'}))
                finally:
                    inlet.close_stream()

    def test_disabled_output_and_practice(self):
        for name, (relative, _) in TASKS.items():
            disabled = task_class(REPO / relative, 'main', False)()
            self.assertIsNone(disabled.lsl)
            self.assertEqual(disabled.mode, 'none')
            disabled.send(11)
            disabled.reset()
            disabled.close()
            if name != 'eoec':
                practice = task_class(REPO / relative, 'practice', True)()
                self.assertIsNone(practice.lsl)
                self.assertEqual(practice.mode, 'none')
                practice_udp = task_class(REPO / relative, 'practice', False, True)()
                self.assertIsNone(practice_udp.galea_udp)

if __name__ == '__main__':
    unittest.main(verbosity=2)
