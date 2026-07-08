import time

import serial

import numpy as np


STEPS = 1200

PHYSICAL_NODES = 8
VIRTUAL_NODES = 4

STATES = 2 * VIRTUAL_NODES * PHYSICAL_NODES

DIMENSION = STATES + 4

TOTAL_FLOATS = STEPS * DIMENSION
TOTAL_BYTES = TOTAL_FLOATS * 4

PORT = '/dev/cu.usbmodem14103'
BAUD_RATE = 921600

SINK_FILE = f'recordings-{time.time()}-rossler-reservoir-2'


if __name__ == '__main__':
    with serial.Serial(PORT, BAUD_RATE, timeout=10) as ser:
        print(f'Listening on {PORT} at {BAUD_RATE} baud...')

        buffer = bytearray()
        while True:
            char = ser.read(1)
            if not char:
                continue

            buffer += char
            if len(buffer) >= 4 and buffer[-4:] == b'SYNC':
                print('Sync word detected. Receiving data...')
                break

        raw_data = ser.read(TOTAL_BYTES)

        if len(raw_data) < TOTAL_BYTES:
            raise BufferError(
                f'Error: Received only {len(raw_data)} out of {TOTAL_BYTES} bytes.'
            )

        print('Data received. Decoding...')
        recordings = np.frombuffer(raw_data, dtype=np.float32)
        recordings = recordings.reshape((STEPS, DIMENSION))

        np.savez(SINK_FILE, recordings=recordings)

        print(recordings)
