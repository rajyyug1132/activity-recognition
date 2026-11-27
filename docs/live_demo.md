# Live demo: phone → laptop over Wi-Fi

The **Live** page (`app/pages/1_Live.py`, in the app's sidebar) classifies what the phone is doing right now. It reads sensor data from the free [phyphox](https://phyphox.org) app through phyphox's *remote access* web server. It needs no cable and no app of our own.

> **Status:** the replay mode is tested. The phone path has a unit test against a fake phyphox server (`tests/test_live.py`) but **has not been tested on a real phone yet**. Try it before the review.

## Why a custom experiment

No built-in phyphox experiment records the accelerometer and the gyroscope together ("Acceleration with g" and "Gyroscope" are separate). `docs/har_live.phyphox` is a minimal experiment that records both at 50 Hz into these buffers:

| Buffer | Unit | Live page converts to |
|---|---|---|
| `acc_time`, `accX/Y/Z` | s, m/s² (with gravity) | `acc_x..z` in g (÷ 9.80665) |
| `gyr_time`, `gyrX/Y/Z` | s, rad/s | `gyro_x..z` in rad/s, resampled onto `acc_time` |

## Phone setup (once)

1. Install **phyphox** (Android or iOS).
2. Get `docs/har_live.phyphox` onto the phone (email it to yourself, or put it in Drive) and open the file with phyphox. It then appears under the category **Activity Recognition**. Save it to the collection when phyphox asks.
   - Fallback without the file: build it in the [phyphox editor](https://phyphox.org/editor/) with an accelerometer and a gyroscope input. Use the buffer names in the table above, then load it with the editor's QR code.

## Each demo

1. Put the phone and the laptop on the **same Wi-Fi**. A phone hotspot with the laptop joined to it works, and avoids campus networks that block device-to-device traffic.
2. Open the **HAR live (acc + gyro)** experiment. Open the ⋮ menu, choose **Allow remote access** and switch it on. Phyphox shows an address such as `http://192.168.0.42:8080`.
3. On the laptop: `streamlit run app/streamlit_app.py` from the repo root. Open **Live** in the sidebar.
4. Choose **Phone (Phyphox)**, type the address (`192.168.0.42:8080` or just `192.168.0.42`) and press **Start**. The page clears the phone's buffers and starts the measurement itself (`/control?cmd=clear`, then `cmd=start`).
5. Clip the phone at the waist, the same way as in `data_collection_protocol.md`: upright, screen facing out. The first prediction appears after about 2.6 s, then one every second.
6. **Stop** stops the measurement on the phone.

### Axis check (do this the first time)

UCI's vertical axis is `acc_x`, but the phone's vertical axis is its `y`. `src/live.py` remaps the axes with `AXES = [(1, 1), (0, 1), (2, -1)]`: phone y → x, phone x → y, phone −z → z. The same mapping applies to the gyro. Stand still with the phone at the waist, then open `http://<phone-ip>:8080/get?accY` in a browser. It should read about **+9.8**. If another axis reads ±9.8 instead, change `AXES` so that axis feeds UCI x with the matching sign.

## Without a phone: replay mode

Choose **Replay sample** and press **Start**. The page plays `data/sample_synthetic.csv` back in real time, looping, through the same buffer → window → model path. Use this to rehearse the demo or to show the page when Wi-Fi fails. The replay CSV is not included in the repository and no script generates it; provide a compatible CSV at that path. If it is missing, the page explains why replay is unavailable and leaves the phone source usable.

## How it works

- `PhyphoxSource.read()` polls `GET /get?accX=<t>|acc_time&…&acc_time=<t>&gyrX=<t>|gyr_time&…`. Each call returns only the samples newer than the last timestamp seen (phyphox's threshold queries, see the [remote interface docs](https://phyphox.org/docs/remote-interface/)).
- `LiveClassifier` keeps the last 5 s of samples. It runs the existing `to_windows` (resample to 50 Hz, filters, gravity split) and `normalize`, then classifies the **newest** 128-sample window with `models/cnn_uci.keras`. The buffer is 5 s rather than 2.56 s, so the 0.3 Hz gravity filter has enough context.
- The page polls once a second (`st.fragment(run_every=1)`). The top card shows the current activity and its confidence. The strip shows the last 30 predictions, oldest on the left.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Could not start: timed out" | Laptop can't reach the phone. Check that both are on the same network, remote access is on, and the IP is correct. Try the hotspot. |
| "Lost contact with the phone" | Wi-Fi dropped. The page keeps retrying every second. |
| Stuck on "Collecting…" | The measurement isn't running on the phone. Press ▶ in phyphox, or Stop and Start on the page. |
| Always "Lying" or always one wrong class | Axes are mapped wrong. Do the axis check above. |
