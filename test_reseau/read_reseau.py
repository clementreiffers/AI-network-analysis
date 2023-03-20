import pyshark

cap = pyshark.FileCapture('cpature_test.cap')
print(cap[0])