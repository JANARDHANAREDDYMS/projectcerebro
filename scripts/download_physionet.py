import wfdb
import os
from tqdm import tqdm

MOTOR_IMAGERY_RUNS = [4, 6, 8, 10, 12, 14]
SUBJECTS = list(range(1, 110))
CORRUPTED = [88, 92, 100, 104, 106]
SAVE_DIR = "../data/physionet"
os.makedirs(SAVE_DIR, exist_ok=True)

def download_subject(subject_id):
    if subject_id in CORRUPTED:
        print(f"⏭️  Skipping S{subject_id:03d} — known corrupted")
        return

    subject_str = f"S{subject_id:03d}"
    save_path = os.path.join(SAVE_DIR, subject_str)
    os.makedirs(save_path, exist_ok=True)

    for run in MOTOR_IMAGERY_RUNS:
        run_str = f"S{subject_id:03d}R{run:02d}"
        edf_file = os.path.join(save_path, f"{run_str}.edf")

        if os.path.exists(edf_file):
            print(f"⏭️  Already exists: {run_str}")
            continue

        try:
            # correct API — pn_dir is the PhysioNet database name
            wfdb.dl_database(
                'eegmmidb',
                dl_dir=save_path,
                records=[run_str],
                annotators=None
            )
            print(f"✅ {run_str} downloaded")
        except Exception as e:
            # fallback — try rdrecord which streams directly
            try:
                record = wfdb.rdrecord(
                    run_str,
                    pn_dir=f'eegmmidb/{subject_str}'
                )
                # save locally
                wfdb.wrsamp(
                    record_name=run_str,
                    fs=record.fs,
                    units=record.units,
                    sig_name=record.sig_name,
                    p_signal=record.p_signal,
                    fmt=record.fmt,
                    write_dir=save_path
                )
                print(f"✅ {run_str} downloaded via rdrecord")
            except Exception as e2:
                print(f"❌ {run_str} failed: {e2}")

if __name__ == "__main__":
    print(f"Downloading PhysioNet EEGMMIDB — motor imagery runs only")
    print(f"Subjects: {len(SUBJECTS) - len(CORRUPTED)} | Runs per subject: {len(MOTOR_IMAGERY_RUNS)}")
    for subject_id in tqdm(SUBJECTS):
        download_subject(subject_id)
    print("✅ PhysioNet download complete!")