# Data Collection Protocol

Goal: record our own phone data so it matches the UCI HAR format and can be fed to `preprocess.to_windows`.

## Setup (everyone, identical)
- **App:** one CSV-exporting sensor app for the whole team (Sensor Logger or phyphox). Same app, same settings, same version.
- **Sensors:** accelerometer **including gravity** (not "linear acceleration") + gyroscope.
- **Rate:** >= 50 Hz (set 50 Hz if the app allows; higher is fine, we resample).
- **Phone placement:** at the waist (belt clip or front trouser pocket), screen facing out (away from body), top of phone pointing up. Same placement every session.
- **Subjects:** `p1`–`p5` only. Train = p1–p3, test = p4–p5. **Never use real names** in file names, log sheets or plots.

## Activities
Six activities, **2–3 min each**: `walking`, `walking_upstairs`, `walking_downstairs`, `sitting`, `standing`, `lying`.

## Recording procedure
1. Start recording, put the phone in place, begin the activity.
2. Record 2–3 min continuously, stop, export CSV.
3. **Trim the first 5 s and last 5 s** (placement / reaching for the phone).
4. One file per person per activity, named `p1_walking_01.csv` (`<person>_<activity>_<take>.csv`).
5. Fill in the log sheet.

## Log sheet
| person | phone model | date | activity | duration (s) | notes |
|--------|-------------|------|----------|--------------|-------|
| p1 | | YYYY-MM-DD | walking | | |

## Upload
Raw (untrimmed or trimmed, as exported) files go to Drive: `HAR-data/own/raw/`. Do not edit raw files after upload.

## Convert to UCI format
Target columns: `timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z` — timestamp in seconds (float), acc in **g** (including gravity), gyro in **rad/s**.

1. **Accel units:** if the app exports m/s², divide by 9.80665 to get g. (Some apps already give g — check: standing still, |acc| should be ~1, not ~9.8.)
2. **Gyro units:** must be rad/s. If the app gives deg/s, multiply by pi/180.
3. **Timestamp:** convert to seconds as a float (e.g. ms or ns since start -> divide by 1e3 / 1e9).
4. **Axis matching:** phone axes may not match UCI axes. While **standing** (phone at waist, upright), find which UCI axis reads ~1 g (UCI: `acc_x`, the vertical axis) and which phone axis reads ~1 g. Swap columns so the vertical phone axis becomes `acc_x`; flip the sign (multiply by -1) if it reads -1 g instead of +1 g. Apply the same swap/flip to the matching gyro axes. Repeat for the other axes so right-handedness is kept.
5. **Rename** columns to `timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z` and drop everything else.
6. **Validate** with `validation.validate_csv(df, cfg)`; fix anything it reports.
7. Save to Drive `HAR-data/own/processed/` (same file name as the raw file).
