#!/usr/bin/env python3
"""USB IMU logger (Python 3.9+).

Install: python -m pip install pyserial
Ports:   python record_imu.py --list-ports
Example (ONLY if firmware sends time in ms, acceleration in g,
gyroscope in deg/s and magnetometer in microtesla):
  python record_imu.py --port COM3 --time-unit ms --mag-unit uT

Required Arduino output: one comma-separated line per sample, terminated
by newline, in EXACT order: time,ax,ay,az,gx,gy,gz,mx,my,mz.
An optional header must be: imu_time,ax,ay,az,gx,gy,gz,mx,my,mz
Arduino firmware must read and stream the sensors; this script cannot do
that itself. Configure sample rate on Arduino, not in this logger.

CSV output units: seconds, g (9.8 m/s^2), deg/s, gauss.
No filtering, gravity removal, axis conversion or Vicon synchronization.
Device timestamps are preserved (converted to seconds), NOT replaced by
PC arrival times. Clock resets/non-increasing times stop acquisition.
Duration is PC elapsed recording time, default 600 seconds; 0 = until Ctrl+C.
Outputs are in a new unique directory, never overwrite previous trials.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import tempfile
import time
from datetime import datetime, timezone

HEADER = ['imu_time', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz']


def parse_sample(line, time_unit, acc_unit, gyro_unit, mag_unit):
    parts = line.strip().split(',')
    if [p.strip() for p in parts] == HEADER:
        return None
    if len(parts) != 10:
        raise ValueError('Expected exactly 10 comma-separated numbers')
    values = [float(p) for p in parts]
    if not all(math.isfinite(v) for v in values):
        raise ValueError('NaN or infinity')
    values[0] *= {'s': 1, 'ms': 0.001, 'us': 0.000001}[time_unit]
    if acc_unit == 'm/s2':
        values[1:4] = [v / 9.8 for v in values[1:4]]
    if gyro_unit == 'rad/s':
        values[4:7] = [math.degrees(v) for v in values[4:7]]
    factor = {'G': 1, 'uT': 0.01}[mag_unit]
    values[7:10] = [v * factor for v in values[7:10]]
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--list-ports', action='store_true')
    parser.add_argument('--port', help='For example COM3 or /dev/ttyACM0')
    parser.add_argument('--baud', type=int, default=115200)
    parser.add_argument('--time-unit', choices=['s', 'ms', 'us'])
    parser.add_argument('--mag-unit', choices=['G', 'uT'])
    parser.add_argument('--acc-unit', choices=['g', 'm/s2'], default='g')
    parser.add_argument('--gyro-unit', choices=['deg/s', 'rad/s'], default='deg/s')
    parser.add_argument('--duration', type=float, default=600)
    parser.add_argument('--out', type=Path, default=Path('imu_recordings'))
    args = parser.parse_args(argv)
    try:
        import serial
        from serial.tools import list_ports
    except ImportError:
        print('Install dependency: python -m pip install pyserial')
        return 1
    if args.list_ports:
        ports = list(list_ports.comports())
        for port in ports:
            print(f'{port.device}: {port.description}')
        if not ports:
            print('No serial ports found. Check USB cable and Arduino connection.')
        return 0
    if not args.port or not args.time_unit or not args.mag_unit:
        parser.error('--port, --time-unit and --mag-unit are required; check firmware units')
    if not math.isfinite(args.duration) or args.duration < 0 or args.baud <= 0:
        parser.error('duration must be finite and >= 0; baud must be positive')
    args.out.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=datetime.now().strftime('trial_%Y%m%d_%H%M%S_'),
                                  dir=str(args.out)))
    metadata = dict(vars(args), out=str(folder.resolve()),
                    start_utc=datetime.now(timezone.utc).isoformat(),
                    output_units={'time': 's', 'acceleration': 'g (9.8 m/s2)',
                                  'angular_velocity': 'deg/s', 'magnetic_field': 'G'},
                    note='No coordinate conversion, filtering or Vicon sync applied.')
    count = rejected = 0
    first_t = last_t = None
    status = 'completed'
    error = None
    pending = b''
    start = time.monotonic()
    last_valid = last_flush = last_report = start
    print(f'Output: {folder.resolve()}\nClose Arduino Serial Monitor/Plotter first.')
    try:
        with (folder / 'imu.csv').open('w', newline='', encoding='utf-8') as output, \
             (folder / 'raw_serial.bin').open('wb') as raw, \
             (folder / 'reception.csv').open('w', newline='', encoding='utf-8') as reception, \
             (folder / 'rejected.log').open('w', encoding='utf-8') as bad:
            writer = csv.writer(output)
            writer.writerow(HEADER)
            arrivals = csv.writer(reception)
            arrivals.writerow(['sample_index', 'imu_time_s', 'pc_elapsed_s'])
            with serial.Serial(args.port, args.baud, timeout=0.2) as device:
                # Do not clear input: opening some boards may reset them;
                # startup lines are preserved in raw_serial.bin/rejected.log.
                start = last_valid = last_flush = last_report = time.monotonic()
                print('RECORDING. Check that sample count increases, then start Vicon. Ctrl+C stops.')
                try:
                    while args.duration == 0 or time.monotonic() - start < args.duration:
                        chunk = device.read(min(max(device.in_waiting, 1), 8192))
                        now = time.monotonic()
                        raw.write(chunk)
                        pending += chunk
                        while b'\n' in pending:
                            line, pending = pending.split(b'\n', 1)
                            try:
                                row = parse_sample(line.decode('utf-8'), args.time_unit,
                                                   args.acc_unit, args.gyro_unit, args.mag_unit)
                            except (ValueError, UnicodeError) as exc:
                                rejected += 1
                                bad.write(f'{now-start:.6f}: {exc}: {line!r}\n')
                                if rejected <= 3:
                                    print(f'Skipped line: {line[:150]!r}')
                                continue
                            if row is None:
                                continue
                            if last_t is not None and row[0] <= last_t:
                                raise RuntimeError('Device timestamp reset/repeated/decreased. '
                                                   'Check firmware clock and start a new trial.')
                            writer.writerow(row)
                            arrivals.writerow([count, row[0], f'{now-start:.6f}'])
                            count += 1
                            first_t = row[0] if first_t is None else first_t
                            last_t = row[0]
                            last_valid = now
                        if len(pending) > 65536:
                            raise RuntimeError('No newline in >64 KiB; check Arduino output format.')
                        if now - last_valid > 10:
                            raise RuntimeError('No valid sample for 10 seconds. Check port, baud, '
                                               'firmware and 10-column format; inspect rejected.log.')
                        if now - last_flush >= 1:
                            for file in (output, raw, reception, bad):
                                file.flush()
                            last_flush = now
                        if now - last_report >= 2:
                            hz = (count-1)/(last_t-first_t) if count > 1 else 0
                            print(f'{now-start:.1f}s | samples={count} | '
                                  f'mean device rate={hz:.2f} Hz | rejected={rejected}')
                            last_report = now
                finally:
                    for file in (output, raw, reception, bad):
                        file.flush()
                        os.fsync(file.fileno())
    except KeyboardInterrupt:
        status = 'stopped_by_user'
    except (serial.SerialException, OSError, RuntimeError) as exc:
        status, error = 'error', str(exc)
        print(f'ERROR: {error}')
    finally:
        metadata.update(status=status, error=error, samples=count, rejected_lines=rejected,
                        first_device_time_s=first_t, last_device_time_s=last_t,
                        incomplete_tail_bytes=len(pending),
                        end_utc=datetime.now(timezone.utc).isoformat())
        (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print(f'Saved {count} samples: {folder.resolve() / "imu.csv"}')
    return 1 if status == 'error' or count == 0 else 0


if __name__ == '__main__':
    raise SystemExit(main())
