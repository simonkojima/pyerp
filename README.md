# scab-python
python library for Event-Related Potentials (ERPs) analysis


# Instruction
```python
eeg_files = list()
eeg_files.append(["NAME_OF_FILE_WITHOUT_EXTENTION", "TASK_NAME"])
```

```python
marker = list()
marker.append([101, "nontarget", "word1"])
marker.append([102, "nontarget", "word2"])
marker.append([103, "nontarget", "word3"])
marker.append([104, "nontarget", "word4"])
marker.append([105, "nontarget", "word5"])
marker.append([106, "nontarget", "word6"])
marker.append([111, "target", "word1"])
marker.append([112, "target", "word2"])
marker.append([113, "target", "word3"])
marker.append([114, "target", "word4"])
marker.append([115, "target", "word5"])
marker.append([116, "target", "word6"])
marker.append([210, "new-trial"])
```