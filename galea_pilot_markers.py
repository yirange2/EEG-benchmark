"""Optional local Galea UDP and LSL event-marker outputs."""
import socket

class LslMarkers:
    def __init__(self, task, stream_name="GaleaPsychoPyMarkers"):
        try:
            from pylsl import StreamInfo, StreamOutlet, local_clock
        except ImportError as exc:
            raise RuntimeError(
                "LSL output requires pylsl; install requirements-lsl.txt"
            ) from exc
        self._local_clock = local_clock
        info = StreamInfo(stream_name, "Markers", 1, 0, "int32",
                          f"galea-psychopy-{socket.gethostname()}-{task}")
        desc = info.desc()
        desc.append_child_value("task", task)
        desc.append_child_value("psychopy_version", "2026.2.4")
        desc.append_child_value("event_semantics", "Original task trigger codes; no reset samples")
        desc.append_child_value("timing", "LSL local_clock at the task send callback; stimulus callbacks use callOnFlip; response events follow the original task")
        desc.append_child("channels").append_child("channel").append_child_value("label", "event_code")
        sync = desc.append_child("synchronization")
        sync.append_child_value("offset_mean", "0")
        sync.append_child_value("can_drop_samples", "false")
        self.outlet = StreamOutlet(info, chunk_size=1, max_buffered=360)
        print(f"LSL marker stream: {stream_name} ({task}). Record this with the Galea EEG stream.", flush=True)

    def send(self, code):
        code = int(code)
        if code != 0:
            self.outlet.push_sample([code], timestamp=self._local_clock(), pushthrough=True)

    def close(self):
        # Keep the outlet alive until the process exits so queued tail events
        # are not discarded during task cleanup.
        pass


class GaleaUdpMarkers:
    """Match OpenBCI Unity SDK 0.4.0 Marker.AddStreamMarker: network-order double."""
    def __init__(self, port, host="127.0.0.1"):
        import struct
        if host != "127.0.0.1":
            raise ValueError("Pilot UDP markers are restricted to the local GUI")
        if not 1 <= int(port) <= 65535:
            raise ValueError("Marker UDP port must be between 1 and 65535")
        self.destination = (host, int(port))
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.pack = struct.Struct(">d").pack
        print(f"Galea GUI markers: UDP {host}:{port}. Enable the GUI Marker widget on this port; UDP has no receipt acknowledgement.", flush=True)

    def send(self, code):
        if int(code) != 0:
            self.socket.sendto(self.pack(float(code)), self.destination)

    def close(self):
        self.socket.close()
