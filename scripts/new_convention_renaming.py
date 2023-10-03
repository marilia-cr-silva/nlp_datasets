import os

renaming_map = {
    "sa_01": "pc_01",
    "sa_03": "pc_02",
    "sa_04": "pc_03",
    "sa_05": "pc_04",
    "sa_06": "pc_05",
    "sa_07": "pc_06",
    "sa_11": "pc_07",
    "sa_16": "pc_08",
    "sa_17": "pc_09",
    "sa_02": "ed_01",
    "sa_10": "ed_02",
    "sa_14": "ed_03",
    "sa_15": "ed_04",
    "sa_19": "ed_05",
    "sa_08": "ua_01",
    "sa_09": "ua_02",
    "sa_12": "ua_03",
    "sa_13": "ua_04",
    "sa_18": "ua_05",
    "sa_20": "ua_06",
}

files = os.listdir()

audit_files = set(files)

for old, renamed in renaming_map.items():
    current_files = [f for f in files if old in f]

    correct_prefix = set([f[:5] for f in current_files])

    print(f"correct_prefix={correct_prefix}")
    assert len(correct_prefix) == 1, "There must only be one prefix"

    prefix = correct_prefix.pop()
    assert prefix == old, "Prefix does not match expected old file"

    for f in current_files:
        new_name = renamed + f[5:]

        os.rename(f, new_name)
        print(f"renamed `{f}` to `{new_name}`")

        audit_files.remove(f)

print(f"Remaining files:\n{audit_files}")

remaining_datasets = {s[:5] for s in audit_files}
print(f"Remaining datasets:\n{remaining_datasets}")
remaining_tasks = {s[:2] for s in audit_files}
print(f"Remaining tasks:\n{remaining_tasks}")
