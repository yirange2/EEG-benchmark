# Galea markers, fullscreen presentation, and Windows audio

The main tasks can send the original event codes to the original Galea GUI
over localhost UDP and/or to an LSL marker stream. Existing parallel-port
triggers remain available. Software marker outputs are optional and disabled
unless requested. Practice programs send no markers and save no task data.

## Original Galea GUI

1. Connect Galea and start a GUI session.
2. Select the **Marker** widget. Leave Receive IP at `127.0.0.1`, set Receive
   Port to `12340`, and click **Start Receiver**. It should become Stop Receiver.
3. Start the data stream/recording before beginning the task.
4. Launch a main task, for example:

   ```console
   python 02-SSVEP/ssvep_main.py --participant pilot01 --run 01 --send-galea-markers
   ```

5. After the task completes, stop the GUI recording and compare its EEG Marker
   channel with the saved task event codes. Interrupted trials can have an EEG
   onset marker without a completed row in the task table.

UDP uses an eight-byte network-order float64 (`>d`), matching the Galea Unity
SDK Marker.AddStreamMarker protocol. It sends only nonzero event codes; reset
samples are not events. The default is the original Galea GUI's port 12340,
not the generic OpenBCI GUI port. Override with `--galea-marker-port PORT` if
the receiver is configured differently. UDP has no acknowledgement: console
output alone does not verify recording. These changes do not add support to
Meridian or stream EEG from the task itself.

## Optional LSL output

Install `requirements-lsl.txt` into the task environment, then add
`--send-lsl-markers`. The stream is named `GaleaPsychoPyMarkers`, with type
`Markers` and one int32 channel. Set `--lsl-stream-name NAME` to override.
Record it together with the EEG stream using an LSL recorder; starting the
marker outlet does not itself save an XDF file. UDP and LSL can be enabled
together. UDP-only use does not require pylsl.

Stimulus markers use the task's existing flip callbacks. Response markers
retain their existing task semantics. Software event timestamps and EEG
marker sample placement do not measure the time light reaches the eyes;
validate display latency independently when precise timing is required.

## Display and audio

Main and practice tasks open fullscreen by default on display index 0. Use
`--windowed` for a window or `--screen-index N` to select another display.
Smoke tests stay windowed. Timing metadata records the actual drawable and
framebuffer sizes. Degree-based tasks update monitor pixel dimensions for
fullscreen presentation, while retaining the repository's existing physical
monitor defaults and geometry overrides.

Actual pixels are not a visual-angle calibration. Measure/configure display
width and viewing distance for the apparatus; a virtual desktop additionally
needs its angular geometry and compositor timing validated. The IMMERSE
pilot's uncalibrated Varjo geometry is deliberately not a default here.

EOEC uses the current Windows default output when no device is specified.
Use `--audio-device "DEVICE NAME"` to select an output available to PsychoPy.
The selection applies only to this process. Other platforms retain their
existing default behavior unless an override is supplied.

## Validation

Run each task's existing unit tests from its own task directory. With the
optional LSL dependency installed, run `python -m unittest test_galea_markers`
from the repository root to exercise the task adapters against local UDP
receivers and native LSL inlets. These tests verify event-code delivery and
practice suppression; they do not validate physiological signal quality or
physical stimulus latency.
