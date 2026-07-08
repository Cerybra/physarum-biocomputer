import time
import logging

import serial

import numpy as np

from sklearn.datasets import load_digits


# Protocol constants.
STRT = b'STRT'
RQST = b'RQST'
RCVD = b'RCVD'
NACK = b'NACK'
FNSH = b'FNSH'

TIMEOUT = 2.0

ROWS = 8
COLS = 8


def _flush_buffers(comm: serial.Serial):
    comm.reset_input_buffer()
    comm.reset_output_buffer()


def _checksum_bytes(data: bytes) -> int:
    return sum(data) % (2**32)


def _wait_for_signal(comm: serial.Serial, signal: bytes, timeout=TIMEOUT):
    deadline = time.time() + timeout
    buffer = bytearray()

    while time.time() < deadline:
        b = comm.read(1)

        if not b:
            continue

        buffer += b

        if len(buffer) > len(signal):
            buffer = buffer[1:]

        if buffer == signal:
            return True

    return False


def send_array(comm: serial.Serial, array: np.ndarray):
    array = np.ascontiguousarray(array, dtype=np.float32)
    data = array.tobytes()

    header = ROWS.to_bytes(4, 'little') + COLS.to_bytes(4, 'little')
    payload = STRT + header + data

    checksum = _checksum_bytes(payload).to_bytes(4, 'little')

    comm.write(payload + checksum)
    comm.flush()


def receive_array(comm: serial.Serial):
    if not _wait_for_signal(comm, STRT, timeout=TIMEOUT):
        return None

    header = comm.read(8)
    if len(header) < 8:
        return None

    rows = int.from_bytes(header[:4], 'little')
    cols = int.from_bytes(header[4:8], 'little')

    if rows != 8 or cols != 8:
        return None

    array = comm.read(rows * cols * 4)
    if len(array) < rows * cols * 4:
        return None

    payload = STRT + header + array
    checksum = comm.read(4)

    if len(checksum) < 4:
        return None

    received_checksum = int.from_bytes(checksum, 'little')

    if _checksum_bytes(payload) != received_checksum:
        return None

    return np.frombuffer(array, dtype=np.float32).reshape((rows, cols))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    digits = load_digits()

    images, labels = digits['data'].reshape(-1, 8, 8), digits['target'].reshape(-1, 1)
    images = ((images / images.max(axis=(1, 2), keepdims=True)) > 0.5).astype(np.float32)

    images_sent = []
    labels_sent = []
    responses_received = []

    filepath = 'digits-responses-3.npz'

    try:
        with serial.Serial('/dev/cu.usbmodem14103', 115200, timeout=0.5) as serial_port:
            for index, image in enumerate(images):
                if not _wait_for_signal(serial_port, RQST):
                    logging.warning(f'Timeout waiting for RQST for image {index + 1}')

                    continue

                for attempt in range(3):

                    _flush_buffers(serial_port)

                    try:
                        send_array(serial_port, image)
                    except Exception as exception:
                        logging.error(f'Send error: {exception}')
                        time.sleep(0.2)

                        continue

                    # Wait for RCVD or NACK.
                    if _wait_for_signal(serial_port, RCVD):
                        logging.info(f'Sent image {index + 1}/{len(images)} and received RCVD.')

                        response = receive_array(serial_port)

                        if response is not None:
                            images_sent.append(image.copy())
                            labels_sent.append(labels[index, 0])
                            responses_received.append(response)

                            logging.info(f'Received response for image {index + 1}')
                        else:
                            logging.error(f'Failed to receive response for image {index + 1}, will retry.')

                            continue

                        break
                    elif _wait_for_signal(serial_port, NACK):
                        logging.warning(f'NACK for image {index + 1}, retrying (attempt {attempt + 2})...')
                        time.sleep(0.2)

                        continue
                    else:
                        logging.warning(f'No RCVD/NACK for image {index + 1}, retrying (attempt {attempt + 2})...')
                        time.sleep(0.2)
                else:
                    logging.error(f'Failed to send image {index + 1} after 3 attempts. Resynchronizing...')
                    _flush_buffers(serial_port)

            # Graceful shutdown: send FNSH and wait for RCVD.
            serial_port.write(FNSH)
            serial_port.flush()

            logging.info('All data sent. FNSH signal transmitted.')

            if _wait_for_signal(serial_port, RCVD, timeout=5):
                logging.info('MCU acknowledged FNSH. Shutdown complete.')
            else:
                logging.warning('No RCVD for FNSH. MCU may not have shut down cleanly.')

    except serial.SerialException as error:
        logging.critical(f'Serial port error: {error}')

    except Exception as error:
        logging.critical(f'Unexpected error: {error}')

    results = {
        'data': np.array(images_sent),
        'label': np.array(labels_sent),
        'response': np.array(responses_received),
    }

    np.savez(filepath, **results)
